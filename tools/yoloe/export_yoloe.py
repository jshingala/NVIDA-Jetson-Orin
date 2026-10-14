"""Bake Loci's object list into YOLOE-26 and build a TensorRT engine (FP16, or INT8).

YOLOE is an open-vocabulary detector: it finds objects named in plain text.
Exporting freezes the names into the weights, so the engine is a normal YOLO
model at run time (no text encoder on the board after this step).

    python3 export_yoloe.py --image ~/test.jpg          # FP16 -> <weights>.engine
    python3 export_yoloe.py --int8 ~/calib               # INT8 -> <weights>-int8.engine

INT8 calibrates on the images in the given folder (no labels needed; 300+
frames of the real room recommended, e.g. from `loci_capture --save`).
Also writes classes.txt next to the weights.
The first run downloads the checkpoint and a ~254 MB text encoder into the
current directory, so run it from a scratch folder, not the repo.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLOE

CLASSES = [
    "keys", "wallet", "phone", "glasses", "mug",
    "remote control", "backpack", "laptop", "water bottle", "headphones",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yoloe-26s-seg.pt")
    p.add_argument("--image", help="photo to sanity-check the prompts before exporting")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--int8", metavar="DIR", help="build an INT8 engine calibrated on the images in DIR")
    args = p.parse_args()

    model = YOLOE(args.weights)
    model.set_classes(CLASSES)

    if args.image:
        result = model.predict(args.image, imgsz=args.imgsz, conf=0.25, verbose=False)[0]
        print(f"{len(result.boxes)} objects in {args.image}:")
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            print(f"  {result.names[int(box.cls)]:15s} {float(box.conf):.2f}  ({x1},{y1})-({x2},{y2})")
        result.save(filename="yoloe_check.jpg")
        print("annotated image: yoloe_check.jpg")

    if args.int8:
        engine = export_int8(model, Path(args.weights), Path(args.int8).expanduser().resolve(), args.imgsz)
    else:
        engine = Path(model.export(format="engine", quantize=16, imgsz=args.imgsz, batch=1, device=0))
    (engine.parent / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    print(f"engine: {engine}\nclasses: {engine.parent / 'classes.txt'}")


def export_int8(model, weights, calib_dir, imgsz):
    """Ultralytics writes <weights>.engine for every precision, so keep the FP16 one aside while it runs."""
    images = [f for f in calib_dir.iterdir() if f.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}]
    print(f"INT8 calibration on {len(images)} images in {calib_dir}")
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    data = Path("calib.yaml")
    data.write_text(f"path: {calib_dir}\ntrain: .\nval: .\nnames:\n{names}")

    target = weights.with_suffix(".engine")
    backup = target.with_suffix(".engine.fp16")
    if target.exists():
        target.rename(backup)
    try:
        built = Path(model.export(format="engine", quantize=8, data=str(data), imgsz=imgsz, batch=1, device=0))
        int8 = built.with_name(f"{weights.stem}-int8.engine")
        shutil.move(built, int8)
    finally:
        if backup.exists():
            backup.rename(target)
    return int8


if __name__ == "__main__":
    main()
