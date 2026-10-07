"""
Trains the pipeline-class specialist of the two-model system.

Production's LOCO model is never retrained again -- its 5 classes can't be
forgotten if their weights never change. Every source=pipeline class
(iw_hub today, whatever comes next) lives in one small separate model,
retrained from scratch (COCO-pretrained YOLOv8s) each round on ALL approved
pipeline data so far. Retraining from scratch on everything is what keeps
earlier pipeline classes from being forgotten, and it stays cheap because
this model only ever sees the pipeline classes' own images plus negatives.

Class indices stay the global class_registry.json ones, so merging its
detections with the LOCO model's needs no remapping -- the LOCO classes just
never appear in its labels. Official LOCO *train* images are mixed in as
pure negatives (empty labels) so it learns what isn't a pipeline class; LOCO
*val* is never touched here, since two_model_gate.py measures the
specialist's false-positive rate on it.

Usage:
    python train_specialist.py --run-name specialist_v1
"""

import argparse
import json
import os
import random
import time

import db
from build_training_batch import LOCO_SRC, VAL_FRACTION, _label_path_for, _symlink_pair
from copy_paste_augment import augment

SPECIALIST_BASE = os.path.join(db.PROJECT_ROOT, "weights", "yolov8s.pt")
BATCH_ROOT = os.path.join(db.PIPELINE_ROOT, "specialist_batches")
YOLO_RUNS_DIR = os.path.join(db.PROJECT_ROOT, "runs", "detect", "yolo_runs")
NEGATIVE_RATIO = 3
PASTES_PER_SOURCE = 3
SEED = 0


def build_batch(batch_name, negative_ratio=NEGATIVE_RATIO, pastes_per_source=PASTES_PER_SOURCE, seed=SEED):
    random.seed(seed)
    conn = db.connect()
    registry = db.load_class_registry()
    pipeline_classes = db.pipeline_classes(registry)
    if not pipeline_classes:
        raise SystemExit("no source=pipeline classes in class_registry.json -- nothing to train")

    batch_dir = os.path.join(BATCH_ROOT, batch_name)
    if os.path.exists(batch_dir):
        raise SystemExit(f"{batch_dir} already exists -- pick a new --run-name or delete it first")

    split = {"train": [], "val": []}
    for cls in pipeline_classes:
        pool = [(row_id, path) for row_id, path in db.approved_pool(conn, cls["name"]) if os.path.exists(path)]
        random.shuffle(pool)
        n_val = max(1, round(len(pool) * VAL_FRACTION)) if len(pool) > 1 else 0
        split["val"].extend(pool[:n_val])
        split["train"].extend(pool[n_val:])

    loco_img_dir = os.path.join(LOCO_SRC, "images", "train")
    loco_all = sorted(f for f in os.listdir(loco_img_dir) if f.endswith(".jpg"))
    n_neg = {s: negative_ratio * len(split[s]) for s in split}
    negatives = random.sample(loco_all, min(len(loco_all), n_neg["train"] + n_neg["val"]))
    neg_split = {"train": negatives[:n_neg["train"]], "val": negatives[n_neg["train"]:]}

    counts = {}
    for s in ("train", "val"):
        for _row_id, img_path in split[s]:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            _symlink_pair(img_path, os.path.join(batch_dir, "images", s, os.path.basename(img_path)),
                          os.path.join(batch_dir, "labels", s, stem + ".txt"), src_lbl=_label_path_for(img_path))
        for fname in neg_split[s]:
            stem = os.path.splitext(fname)[0]
            _symlink_pair(os.path.join(loco_img_dir, fname), os.path.join(batch_dir, "images", s, fname),
                          os.path.join(batch_dir, "labels", s, stem + ".txt"))  # empty label = pure negative
        counts[s] = {"pipeline": len(split[s]), "negatives": len(neg_split[s])}

    counts["synthetic"] = {}
    for cls in pipeline_classes:
        counts["synthetic"][cls["name"]] = augment(batch_dir, cls["name"], pastes_per_source)

    data_yaml = os.path.join(batch_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {batch_dir}\ntrain: images/train\nval: images/val\n")
        f.write(f"names: {json.dumps({c['class_idx']: c['name'] for c in registry})}\n")
    with open(os.path.join(batch_dir, "manifest.json"), "w") as f:
        json.dump({"batch_name": batch_name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "counts": counts, "negative_ratio": negative_ratio,
                   "pastes_per_source": pastes_per_source, "seed": seed}, f, indent=2)
    print(f"specialist batch {batch_name}: {counts}")
    return data_yaml


def train(data_yaml, run_name, epochs=100, imgsz=960):
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    os.makedirs(os.path.dirname(SPECIALIST_BASE), exist_ok=True)
    attempt_download_asset(SPECIALIST_BASE)
    YOLO(SPECIALIST_BASE).train(
        data=data_yaml, epochs=epochs, patience=20, batch=-1, imgsz=imgsz,
        project=YOLO_RUNS_DIR, name=run_name, exist_ok=True,
    )
    return os.path.join(YOLO_RUNS_DIR, run_name, "weights", "best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    parser.add_argument("--pastes-per-source", type=int, default=PASTES_PER_SOURCE)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    data_yaml = build_batch(args.run_name, args.negative_ratio, args.pastes_per_source)
    weights = train(data_yaml, args.run_name, args.epochs)
    print(f"specialist trained: {weights}")


if __name__ == "__main__":
    main()
