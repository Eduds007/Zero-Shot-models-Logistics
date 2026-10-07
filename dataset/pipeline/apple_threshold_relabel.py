"""
Per-class adaptive pseudo-label thresholds in the spirit of APPLE (Song et
al., "Overcoming Catastrophic Forgetting for Multi-Label Class-Incremental
Learning", WACV 2024): instead of one fixed confidence cut (0.5), each
class's threshold is set so the pseudo-labels have as many objects per
image as the human-annotated LOCO train split does. Classes whose cut is
too strict end up under-labeled, and every missed object is trained as
background.

APPLE adapts its threshold step by step during training under the
assumption that the per-image object count is the same across sessions;
here that assumption holds for unannotated LOCO images from the same
(train) warehouses, so the threshold is solved once, as the score of the
K-th most confident candidate, K = target density x number of images.

Same annotator per class as hybrid_relabel_sam3_loco.py (SAM3 for
small_load_carrier/stillage/pallet_truck, the production model for
forklift/pallet), so the threshold is the only difference from it.

Usage (after sam3_label_loco_unannotated.py --scored-conf 0.1):
    python apple_threshold_relabel.py --n 5000 --tag _wh235
"""

import argparse
import os

import cv2
import numpy as np

import db
from build_training_batch import LOCO_SRC
from copy_paste_augment import _write_yolo_labels
from hybrid_relabel_sam3_loco import PRODUCTION, SAM3_CLASSES
from two_model_gate import predict_raw

PRODUCTION_FLOOR = 0.05
SAM3_FLOOR = 0.1  # the --scored-conf the SAM3 candidates were kept down to
MAX_THRESHOLD = 0.99  # APPLE's epsilon_max


def target_density(class_idx):
    """{class_idx: objects per image} over the annotated LOCO train split."""
    lbl_dir = os.path.join(LOCO_SRC, "labels", "train")
    files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    counts = dict.fromkeys(class_idx, 0)
    for f in files:
        with open(os.path.join(lbl_dir, f)) as fh:
            for line in fh:
                c = int(line.split()[0])
                if c in counts:
                    counts[c] += 1
    return {c: n / len(files) for c, n in counts.items()}


def solve_threshold(scores, k, floor):
    """Score of the k-th most confident candidate, clamped to [floor, MAX_THRESHOLD]."""
    if k <= 0:
        return MAX_THRESHOLD
    if k >= len(scores):
        return floor
    return float(min(MAX_THRESHOLD, max(floor, np.sort(scores)[::-1][k - 1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--tag", default="_wh235")
    args = parser.parse_args()
    registry = db.load_class_registry()
    idx = {c["name"]: c["class_idx"] for c in db.loco_classes(registry)}
    names = {v: k for k, v in idx.items()}
    sam3_idx = {idx[n] for n in SAM3_CLASSES}
    prod_idx = set(idx.values()) - sam3_idx

    root = os.path.join(db.DATASET_ROOT, "loco_unannotated_full")
    src = os.path.join(root, f"sam3_labeled_{args.n}{args.tag}_scored")
    out = os.path.join(root, f"sam3_labeled_{args.n}{args.tag}_apple")
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    if not os.path.exists(os.path.join(out, "images")):
        os.symlink(os.path.join(src, "images"), os.path.join(out, "images"))

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(src, "labels")))
    paths = [os.path.join(src, "images", s + ".jpg") for s in stems]

    # candidates[stem] = [(cls, cx, cy, w, h, score), ...]
    candidates = {s: [] for s in stems}
    for s in stems:
        with open(os.path.join(src, "labels", s + ".txt")) as fh:
            for line in fh:
                c, cx, cy, w, h, score = line.split()
                if int(c) in sam3_idx:
                    candidates[s].append((int(c), float(cx), float(cy), float(w), float(h), float(score)))
    dets = predict_raw(PRODUCTION, paths, conf=PRODUCTION_FLOOR)
    for s, p in zip(stems, paths):
        h, w = cv2.imread(p).shape[:2]
        for (x0, y0, x1, y1), score, c in zip(*dets[p]):
            if int(c) in prod_idx:
                candidates[s].append((int(c), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                                      (x1 - x0) / w, (y1 - y0) / h, float(score)))

    density = target_density(idx.values())
    thresholds = {}
    for c in idx.values():
        scores = np.array([b[5] for boxes in candidates.values() for b in boxes if b[0] == c])
        k = round(density[c] * len(stems))
        thresholds[c] = solve_threshold(scores, k, SAM3_FLOOR if c in sam3_idx else PRODUCTION_FLOOR)
        kept = int((scores >= thresholds[c]).sum())
        at_half = int((scores >= 0.5).sum())
        print(f"{names[c]:<20} target {density[c]:.2f}/img -> threshold {thresholds[c]:.3f} "
              f"({kept / len(stems):.2f}/img kept; {at_half / len(stems):.2f}/img at the fixed 0.5)", flush=True)

    for s in stems:
        boxes = [b[:5] for b in candidates[s] if b[5] >= thresholds[b[0]]]
        _write_yolo_labels(os.path.join(out, "labels", s + ".txt"), boxes)
    print(f"APPLE LABELS DONE: {out}; thresholds: { {names[c]: round(t, 3) for c, t in thresholds.items()} }",
          flush=True)


if __name__ == "__main__":
    main()
