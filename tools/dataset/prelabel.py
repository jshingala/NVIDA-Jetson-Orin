"""Draft labels for Loci's own-object dataset with a large YOLOE, for review in Roboflow.

Runs YOLOE-26l with the object names as text prompts on every image, removes
duplicate boxes (same rules as perception/src/nms.cpp) and writes a folder that
Roboflow imports with the boxes already drawn:

    OUT/<name>.jpg + OUT/<name>.txt   (YOLO format: cls cx cy w h, normalised)
    OUT/classes.txt, OUT/data.yaml

The drafts WILL contain mistakes (zero-shot YOLOE mixed up headphones and
glasses in this room), so review every image in Roboflow before training.

    cd ~/yolo/yoloe && python3 ~/loci/tools/dataset/prelabel.py --images ~/loci_work/data/dataset_jpg --out ~/loci_work/data/dataset_prelabeled
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

# Dataset v2 (Oct 15, 2026): small items Loci finds, larger items, then landmarks.
# Mattress and comforter are labelled as "bed" (they sit inside its box).
CLASSES = [
    "keys", "wallet", "phone", "glasses", "mug", "headphones", "calculator",
    "wireless charger", "extension board", "pillow", "mouse pad",
    "bed", "night stand", "desk",
]


def suppress_duplicates(boxes, iou_thresh=0.5, contain_thresh=0.6):
    """boxes: list of (x1, y1, x2, y2, score, cls). Per-class NMS plus containment, best score first."""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for b in boxes:
        duplicate = False
        for k in kept:
            if k[5] != b[5]:
                continue
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            union = area_b + area_k - inter
            smaller = min(area_b, area_k)
            if (union > 0 and inter / union > iou_thresh) or (smaller > 0 and inter / smaller > contain_thresh):
                duplicate = True
                break
        if not duplicate:
            kept.append(b)
    return kept


def yolo_lines(boxes, width, height):
    """YOLO label lines (class cx cy w h, normalised to 0..1)."""
    lines = []
    for x1, y1, x2, y2, _, cls in boxes:
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=str(Path.home() / "loci_work/data/dataset_jpg"))
    p.add_argument("--out", default=str(Path.home() / "loci_work/data/dataset_prelabeled"))
    p.add_argument("--weights", default="yoloe-26l-seg.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--classes", help="comma-separated names to keep (default: all of CLASSES)")
    p.add_argument("--preview", type=int, default=12, help="also save this many annotated previews")
    args = p.parse_args()

    from ultralytics import YOLOE  # imported here so the helpers above stay testable without it

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else CLASSES
    images = sorted(f for f in Path(args.images).expanduser().iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    preview_dir = out.parent / (out.name + "_preview")
    if args.preview:
        preview_dir.mkdir(exist_ok=True)

    model = YOLOE(args.weights)
    model.set_classes(classes)
    counts, empty = Counter(), 0
    for i, f in enumerate(images):
        r = model.predict(str(f), conf=args.conf, iou=0.5, imgsz=640, verbose=False)[0]
        raw = [(*map(float, b.xyxy[0]), float(b.conf), int(b.cls)) for b in r.boxes]
        kept = suppress_duplicates(raw)
        h, w = r.orig_shape
        shutil.copy2(f, out / f.name)
        (out / f"{f.stem}.txt").write_text("\n".join(yolo_lines(kept, w, h)) + ("\n" if kept else ""))
        counts.update(classes[b[5]] for b in kept)
        empty += not kept
        if i < args.preview:
            import cv2
            img = cv2.imread(str(f))
            for x1, y1, x2, y2, s, c in kept:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{classes[c]} {s:.2f}", (int(x1), max(int(y1) - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(str(preview_dir / f.name), img)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(images)} images")

    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(classes))
    (out / "data.yaml").write_text(f"nc: {len(classes)}\nnames:\n{names}")
    print(f"\n{len(images)} images, {sum(counts.values())} draft boxes, {empty} images with no boxes -> {out}")
    for name in classes:
        print(f"  {name:15s} {counts.get(name, 0)}")
    if args.preview:
        print(f"previews: {preview_dir}")


if __name__ == "__main__":
    main()
