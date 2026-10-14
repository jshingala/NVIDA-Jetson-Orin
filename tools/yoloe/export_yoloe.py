"""Bake Loci's object list into YOLOE-26 and build a TensorRT FP16 engine.

YOLOE is an open-vocabulary detector: it finds objects named in plain text.
Exporting freezes the names into the weights, so the engine is a normal YOLO
model at run time (no text encoder on the board after this step).

    python3 export_yoloe.py --image ~/test.jpg

Writes <weights>.onnx, <weights>.engine and classes.txt next to the weights.
The first run downloads the checkpoint and a ~254 MB text encoder into the
current directory, so run it from a scratch folder, not the repo.
"""

import argparse
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

    engine = Path(model.export(format="engine", half=True, imgsz=args.imgsz, batch=1, device=0))
    (engine.parent / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    print(f"engine: {engine}\nclasses: {engine.parent / 'classes.txt'}")


if __name__ == "__main__":
    main()
