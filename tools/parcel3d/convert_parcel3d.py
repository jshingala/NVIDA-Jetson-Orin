"""Convert Parcel3D (Detectron2-style COCO) to YOLO format.

Dry run by default: prints one sample label per split and counts missing images.
With --go it moves each rgb.png to yolo/images/{train,val,test}/<uuid>.png,
writes yolo/labels/.../<uuid>.txt (cls cx cy w h, normalised) and data.yaml.

    python3 convert_parcel3d.py        # check
    python3 convert_parcel3d.py --go   # convert
"""

import json, os, sys, collections
base = "/media/jenil/D D/parcel3d/extracted/parcel3d"
out = "/media/jenil/D D/parcel3d/yolo"
go = "--go" in sys.argv
for split, ysplit in {"train": "train", "validation": "val", "test": "test"}.items():
    d = json.load(open(f"{base}/parcel3d_{split}.json"))
    anns = collections.defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    ok = missing = 0
    for im in d["images"]:
        uuid = os.path.basename(os.path.dirname(im["file_name"]))
        src = f"{base}/{split}/{uuid}/rgb.png"
        if not os.path.exists(src):
            missing += 1
            continue
        W, H = im["width"], im["height"]
        lines = []
        for a in anns[im["id"]]:
            x, y, w, h = a["bbox"]
            if a["bbox_mode"] == 0:
                w, h = w - x, h - y
            elif a["bbox_mode"] != 1:
                sys.exit(f"unexpected bbox_mode {a['bbox_mode']}")
            lines.append(f'{a["category_id"]} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}')
        if not go:
            if ok == 0:
                print(split, "sample:", im["file_name"], "->", lines)
            ok += 1
            continue
        os.makedirs(f"{out}/images/{ysplit}", exist_ok=True)
        os.makedirs(f"{out}/labels/{ysplit}", exist_ok=True)
        with open(f"{out}/labels/{ysplit}/{uuid}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        os.rename(src, f"{out}/images/{ysplit}/{uuid}.png")
        ok += 1
    print(split, "| images:", len(d["images"]), "| ok:", ok, "| missing:", missing,
          "| bbox_mode:", dict(collections.Counter(a["bbox_mode"] for a in d["annotations"])))
if go:
    with open(f"{out}/data.yaml", "w") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "names:\n  0: normal box\n  1: damaged box\n")
    print("wrote", f"{out}/data.yaml")
