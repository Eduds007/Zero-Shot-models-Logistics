"""
Assembles one fine-tune round's dataset, always starting the mix fresh from
the pristine LOCO-only baseline images (never a previous fine-tune's data)
so drift never compounds across rounds -- same anti-catastrophic-forgetting
idea already validated in build_mixed_robot_dataset.py, generalized from
"1 robot class" to "every source=pipeline class in class_registry.json":

  - For each pipeline class, sample up to REPLAY_BUFFER_SIZE_PER_CLASS
    images from ALL of that class's approved rows to date (not just the
    newest batch) -- capped-per-class replay buffer, re-sampled every round
    so the model doesn't forget older sessions/lighting/angles either.
  - Sample an equal-sized batch of official LOCO images (1:1 with the total
    pipeline-class side, generalizing build_mixed_robot_dataset.py's 1:1
    video/LOCO ratio to N classes) as correct negatives + to keep the
    fine-tune's effective distribution anchored to LOCO's.
  - Every row considered this round (review_status='approved' AND
    incorporated_in_batch IS NULL) gets stamped with this batch's name --
    NOT to exclude it from future rounds (it stays eligible for replay
    sampling forever), just so auto_train_trigger.py's threshold check
    knows this backlog has already been acted on once.

Requires dataset/yolo/images/{train,val} + labels/{train,val} (the official
LOCO YOLO conversion) to exist locally -- see convert_loco_to_yolo.py.

Usage:
    python build_training_batch.py --batch-name loco_finetune_pipeline_v1
"""

import argparse
import json
import os
import random
import time

import db

REPLAY_BUFFER_SIZE_PER_CLASS = 150
LOCO_SRC = os.path.join(db.DATASET_ROOT, "yolo")
BATCH_ROOT = os.path.join(db.PIPELINE_ROOT, "training_batches")
VAL_FRACTION = 0.2
SEED = 0


def _symlink_pair(src_img, dst_img, dst_lbl, src_lbl=None):
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
    if not os.path.exists(dst_img):
        os.symlink(os.path.abspath(src_img), dst_img)
    if not os.path.exists(dst_lbl):
        if src_lbl and os.path.exists(src_lbl):
            os.symlink(os.path.abspath(src_lbl), dst_lbl)
        else:
            open(dst_lbl, "w").close()  # correct negative: no boxes of this pipeline class


def _label_path_for(image_path):
    """approved_pending_training/<class>/images/x.jpg -> .../labels/x.txt"""
    images_dir = os.path.dirname(image_path)
    labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


def build_batch(batch_name, replay_cap=REPLAY_BUFFER_SIZE_PER_CLASS, seed=SEED):
    random.seed(seed)
    conn = db.connect()
    registry = db.load_class_registry()
    pipeline_classes = db.pipeline_classes(registry)
    if not pipeline_classes:
        raise SystemExit("no source=pipeline classes in class_registry.json -- nothing to fine-tune on")

    batch_dir = os.path.join(BATCH_ROOT, batch_name)
    counts = {"pipeline": {"train": 0, "val": 0}, "loco": {"train": 0, "val": 0}}
    ids_to_stamp = []

    # Assign each approved pipeline-class image to exactly one split up front --
    # sampling/splitting independently per split (as an earlier version of this
    # function did) let the same image land in both train and val.
    pipeline_split = {"train": [], "val": []}
    for cls in pipeline_classes:
        pool = db.approved_pool(conn, cls["name"])
        pool = [(row_id, path) for row_id, path in pool if os.path.exists(path)]
        random.shuffle(pool)
        sample = pool[:replay_cap]
        n_val = max(1, round(len(sample) * VAL_FRACTION)) if len(sample) > 1 else 0
        pipeline_split["val"].extend(sample[:n_val])
        pipeline_split["train"].extend(sample[n_val:])

    for split in ("train", "val"):
        for row_id, img_path in pipeline_split[split]:
            lbl_path = _label_path_for(img_path)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            dst_img = os.path.join(batch_dir, "images", split, os.path.basename(img_path))
            dst_lbl = os.path.join(batch_dir, "labels", split, stem + ".txt")
            _symlink_pair(img_path, dst_img, dst_lbl, src_lbl=lbl_path)
            counts["pipeline"][split] += 1
            ids_to_stamp.append(row_id)

        loco_img_dir = os.path.join(LOCO_SRC, "images", split)
        loco_lbl_dir = os.path.join(LOCO_SRC, "labels", split)
        loco_all = sorted(f for f in os.listdir(loco_img_dir) if f.endswith(".jpg"))
        loco_sample = random.sample(loco_all, min(len(pipeline_split[split]), len(loco_all)))
        for fname in loco_sample:
            stem = os.path.splitext(fname)[0]
            dst_img = os.path.join(batch_dir, "images", split, fname)
            dst_lbl = os.path.join(batch_dir, "labels", split, stem + ".txt")
            _symlink_pair(os.path.join(loco_img_dir, fname), dst_img, dst_lbl,
                          src_lbl=os.path.join(loco_lbl_dir, stem + ".txt"))
            counts["loco"][split] += 1

    incorporated = conn.execute(
        "SELECT DISTINCT id FROM frames WHERE review_status = 'approved' AND incorporated_in_batch IS NULL"
    ).fetchall()
    db.mark_incorporated(conn, [r[0] for r in incorporated], batch_name)

    class_names = [c["name"] for c in registry]  # index == class_idx by construction of class_registry.json
    data_yaml_path = os.path.join(batch_dir, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {batch_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names: {json.dumps({c['class_idx']: c['name'] for c in registry})}\n")

    manifest = {
        "batch_name": batch_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "counts": counts,
        "replay_cap_per_class": replay_cap,
        "class_names": class_names,
    }
    with open(os.path.join(batch_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote batch '{batch_name}' to {batch_dir}")
    print(f"counts: {counts}")
    print(f"data yaml: {data_yaml_path}")
    return data_yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-name", default=f"batch_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--replay-cap", type=int, default=REPLAY_BUFFER_SIZE_PER_CLASS)
    args = parser.parse_args()
    build_batch(args.batch_name, replay_cap=args.replay_cap)


if __name__ == "__main__":
    main()
