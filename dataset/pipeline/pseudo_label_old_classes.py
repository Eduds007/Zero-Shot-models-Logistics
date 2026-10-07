"""
Self-training step: a teacher model pseudo-labels the already-known classes
in a training batch's non-LOCO images (the robot's own frames), merging them
with whatever labels those images already have.

Why: pipeline images only carry boxes for the new class (SAM3 labels iw_hub
and nothing else), but 81 of the 83 approved iw_hub frames also contain
forklifts, pallet trucks and pallets. Unlabeled, every training round teaches
the model that those objects are background -- the "missing annotation"
cause of forgetting in incremental detection. Self-training in the sense of
Mayershofer et al., "Improving visual object detection using synthetic
self-training" (Logistics Journal, 2021).

Only images that don't come from the official LOCO split are touched (LOCO
images already have ground truth for every LOCO class). Their label files in
the batch are usually symlinks into approved_pending_training/, so the
symlink is replaced by a real file -- the approved labels are never modified.
Teacher boxes overlapping an existing label (IoU > 0.5) are dropped, so the
teacher can't relabel the new object as an old class.

Usage:
    python pseudo_label_old_classes.py --batch-dir training_batches/x \
        --teacher ../../runs/detect/production/best.pt [--conf 0.5]
"""

import argparse
import os

import numpy as np

import db
from build_training_batch import LOCO_SRC
from copy_paste_augment import _read_yolo_labels, _write_yolo_labels
from two_model_gate import _iou_matrix


def _to_xyxy(boxes, w, h):
    return np.array([[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                     for _c, cx, cy, bw, bh in boxes]).reshape(-1, 4)


def pseudo_label(batch_dir, teacher, conf=0.5, classes=None, imgsz=960):
    from ultralytics import YOLO

    registry = db.load_class_registry()
    classes = set(classes if classes is not None else (c["class_idx"] for c in db.loco_classes(registry)))
    model = YOLO(teacher)
    # By file name, not by resolving symlinks: dataset/yolo's own images are
    # symlinks into the raw LOCO tree, so a realpath check misses them.
    loco_files = {f for split in ("train", "val") for f in os.listdir(os.path.join(LOCO_SRC, "images", split))}
    added, touched = {}, 0
    for split in ("train", "val"):
        img_dir = os.path.join(batch_dir, "images", split)
        lbl_dir = os.path.join(batch_dir, "labels", split)
        targets = [f for f in sorted(os.listdir(img_dir))
                   if f.endswith(".jpg") and f not in loco_files and not f.startswith("synth_")]
        for i in range(0, len(targets), 16):
            chunk = targets[i:i + 16]
            preds = model.predict([os.path.join(img_dir, f) for f in chunk], conf=conf, imgsz=imgsz, verbose=False)
            for fname, pred in zip(chunk, preds):
                h, w = pred.orig_shape
                lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
                existing = _read_yolo_labels(lbl_path)
                ex_xyxy = _to_xyxy(existing, w, h)
                new = []
                for box, c in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.cls.cpu().numpy().astype(int)):
                    if c not in classes:
                        continue
                    if len(ex_xyxy) and _iou_matrix(box[None], ex_xyxy).max() > 0.5:
                        continue
                    x0, y0, x1, y1 = box
                    new.append((int(c), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
                    added[int(c)] = added.get(int(c), 0) + 1
                if os.path.islink(lbl_path):
                    os.unlink(lbl_path)
                _write_yolo_labels(lbl_path, existing + new)
                touched += 1
    names = {c["class_idx"]: c["name"] for c in registry}
    print(f"pseudo-labeled {touched} non-LOCO images; boxes added: "
          f"{ {names[k]: v for k, v in sorted(added.items())} }")
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    pseudo_label(args.batch_dir, args.teacher, args.conf)


if __name__ == "__main__":
    main()
