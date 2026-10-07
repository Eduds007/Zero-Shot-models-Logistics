"""
Hybrid annotator for the SAM3-labeled unannotated LOCO images: keep SAM3's
boxes only for the classes where SAM3 beats the production model zero-shot
(small_load_carrier, stillage, pallet_truck), and take forklift + pallet from
the production model, which SAM3 mostly misses / mislabels (SAM3-only
labels cost forklift -71% and pallet -32%, 3 seeds).

Usage:
    python hybrid_relabel_sam3_loco.py --n 5000 --tag _wh235
"""

import argparse
import os

import cv2

import db
from copy_paste_augment import _write_yolo_labels
from two_model_gate import predict_raw

PRODUCTION = os.path.join(db.PROJECT_ROOT, "runs", "detect", "production", "best.pt")
SAM3_CLASSES = {"small_load_carrier", "stillage", "pallet_truck"}
CONF = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--tag", default="", help='label-set suffix, e.g. "_wh235"')
    args = parser.parse_args()
    registry = db.load_class_registry()
    idx = {c["name"]: c["class_idx"] for c in db.loco_classes(registry)}
    keep_sam3 = {idx[n] for n in SAM3_CLASSES}
    from_prod = set(idx.values()) - keep_sam3

    src = os.path.join(db.DATASET_ROOT, "loco_unannotated_full", f"sam3_labeled_{args.n}{args.tag}")
    out = src + "_hybrid"
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    if not os.path.exists(os.path.join(out, "images")):
        os.symlink(os.path.join(src, "images"), os.path.join(out, "images"))

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(src, "labels")))
    paths = [os.path.join(src, "images", s + ".jpg") for s in stems]
    dets = predict_raw(PRODUCTION, paths, conf=CONF)
    counts = {}
    for s, p in zip(stems, paths):
        boxes = []
        with open(os.path.join(src, "labels", s + ".txt")) as f:
            for line in f:
                c, *xywh = line.split()
                if int(c) in keep_sam3:
                    boxes.append((int(c), *map(float, xywh)))
        h, w = cv2.imread(p).shape[:2]
        for (x0, y0, x1, y1), _s, c in zip(*dets[p]):
            if int(c) in from_prod:
                boxes.append((int(c), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
        for b in boxes:
            counts[b[0]] = counts.get(b[0], 0) + 1
        _write_yolo_labels(os.path.join(out, "labels", s + ".txt"), boxes)
    names = {v: k for k, v in idx.items()}
    print(f"HYBRID LABELS DONE: {out}; boxes: { {names[k]: v for k, v in sorted(counts.items())} }", flush=True)


if __name__ == "__main__":
    main()
