"""
Pseudo-labels for objects a training image isn't human-annotated for, so
they aren't trained as background (the "missing annotation" cause of
forgetting in incremental detection). Used by build_training_batch.py:

  - approved pipeline-class frames: only the pipeline class is
    human-labeled; the LOCO classes come from the production model.
  - environment frames (db.environment_frames): no human labels at all; the
    LOCO classes come from the production model, no pipeline class.
  - the LOCO bank: unannotated LOCO images from the official TRAIN
    warehouses (2, 3, 5), labeled once by SAM3. Each LOCO class takes its
    boxes from the annotator class_registry.json names in
    "pseudo_label_source" -- "sam3" or "production" -- the hybrid that lost
    the least AP over 3 seeds (see CATASTROPHIC_FORGETTING_RESULTS.md).

Every pseudo-label uses one fixed confidence, PSEUDO_LABEL_CONF = 0.5 (no
per-class / adaptive thresholds: tested, they add noisy boxes).
"""

import os

import cv2
import numpy as np

import db
from copy_paste_augment import _read_yolo_labels
from promote_if_better import PRODUCTION_WEIGHTS

PSEUDO_LABEL_CONF = 0.5
LOCO_BANK_N = 5000
LOCO_BANK_SUBSETS = [2, 3, 5]  # the official LOCO train warehouses -- never 1 or 4 (the val/test ones)
LOCO_BANK_DIR = os.path.join(db.DATASET_ROOT, "loco_unannotated_full", f"sam3_labeled_{LOCO_BANK_N}_wh235")
OVERLAP_IOU = 0.5


def production_available():
    return os.path.exists(PRODUCTION_WEIGHTS)


def production_boxes(paths, keep_classes):
    """{path: [(cls, cx, cy, w, h), ...]} from the production model at
    PSEUDO_LABEL_CONF, only for keep_classes. Empty lists if there's no
    production model yet (very first round)."""
    if not paths or not keep_classes or not production_available():
        return {p: [] for p in paths}
    from two_model_gate import predict_raw

    dets = predict_raw(PRODUCTION_WEIGHTS, paths, conf=PSEUDO_LABEL_CONF)
    out = {}
    for p in paths:
        h, w = cv2.imread(p).shape[:2]
        out[p] = [(int(c), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
                  for (x0, y0, x1, y1), _s, c in zip(*dets[p]) if int(c) in keep_classes]
    return out


def _xyxy(boxes):
    return np.array([[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                     for _c, cx, cy, bw, bh in boxes]).reshape(-1, 4)


def add_without_overlap(existing, teacher):
    """existing + the teacher boxes that don't overlap an existing box
    (IoU > OVERLAP_IOU), so the teacher can't relabel a human-labeled
    object as another class."""
    from two_model_gate import _iou_matrix

    ex = _xyxy(existing)
    kept = [b for b in teacher if not len(ex) or _iou_matrix(_xyxy([b]), ex).max() <= OVERLAP_IOU]
    return existing + kept


def ensure_loco_bank():
    """Labels the LOCO bank with SAM3 once (~25 min); later rounds reuse it."""
    labels = os.path.join(LOCO_BANK_DIR, "labels")
    if os.path.isdir(labels) and len(os.listdir(labels)) >= LOCO_BANK_N - len(LOCO_BANK_SUBSETS):
        return
    import json

    import sam3_label_loco_unannotated as s3

    s3.download_and_extract()
    with open(os.path.join(db.DATASET_ROOT, "annotations", "loco-all-v1.json")) as f:
        annotated = {img["file_name"] for img in json.load(f)["images"]}
    s3.label(s3.sample(LOCO_BANK_N, annotated, LOCO_BANK_SUBSETS), LOCO_BANK_DIR, db.load_class_registry())


def loco_bank(registry):
    """[(image_path, stem, boxes)] for the LOCO bank, with each LOCO class
    taken from its pseudo_label_source annotator."""
    ensure_loco_bank()
    sources = {c["class_idx"]: c.get("pseudo_label_source", "production") for c in db.loco_classes(registry)}
    from_sam3 = {c for c, s in sources.items() if s == "sam3"}
    from_prod = {c for c, s in sources.items() if s == "production"}
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(LOCO_BANK_DIR, "labels")))
    paths = [os.path.realpath(os.path.join(LOCO_BANK_DIR, "images", s + ".jpg")) for s in stems]
    prod = production_boxes(paths, from_prod)
    out = []
    for s, p in zip(stems, paths):
        sam3 = [b for b in _read_yolo_labels(os.path.join(LOCO_BANK_DIR, "labels", s + ".txt")) if b[0] in from_sam3]
        out.append((p, s, sam3 + prod[p]))
    return out
