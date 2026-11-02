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
  - Sample a batch of official LOCO images sized at loco_ratio : 1 against
    the total pipeline-class side (generalizing build_mixed_robot_dataset.py's
    1:1 video/LOCO ratio to N classes -- default here is higher than 1:1;
    see below) as correct negatives + to keep the fine-tune's effective
    distribution anchored to LOCO's.

  loco_ratio default is 10.0, not the originally-validated 1:1: a real
  pipeline_finetune_v1 run at 1:1 (90 LOCO train images) regressed on ALL
  5 LOCO classes against the full 2277-image official val set, caught and
  blocked by promote_if_better.py. 10:1 is the ratio of the batches every
  later experiment (and the hybrid recipe) was built on.
  - Every row considered this round (review_status='approved' AND
    incorporated_in_batch IS NULL) gets stamped with this batch's name --
    NOT to exclude it from future rounds (it stays eligible for replay
    sampling forever), just so auto_train_trigger.py's threshold check
    knows this backlog has already been acted on once.

On top of that, every class the image isn't human-annotated for is
pseudo-labeled (pseudo_labels.py), and the batch gets more in-domain
images -- the recipe that lost the least LOCO AP over 3 seeds
(g_loco_train_pseudo in baselines.py, CATASTROPHIC_FORGETTING_RESULTS.md):
  - approved pipeline-class frames: the boxes of every pipeline class the
    capture was approved for (one row per class holds only its own class),
    LOCO classes from the production model;
  - copy-paste augmentation of each pipeline class onto LOCO replay images;
  - up to ENV_FRAMES_CAP environment frames (captured frames SAM3 rejected
    for every pipeline class): LOCO classes from the production model;
  - the LOCO train images not used as replay or val, with pseudo-labels
    instead of their human labels: each LOCO class from its
    class_registry.json pseudo_label_source (SAM3 boxes cached in
    loco_oracle/sam3, or the production model).

Robot frames from a class's test_capture_days never enter a batch.

The val split never touches the LOCO val set: promote_if_better.py scores
on it, so using it here to pick best.pt / early-stop leaks the test into
model selection. LOCO val images are drawn from the LOCO *train* split
instead (not used in this batch's train split) -- the pristine baseline
saw those, so val metrics are only informative logs; train a fixed
schedule and use last.pt (auto_train_trigger.py does). assert_no_test_leak()
refuses any batch holding a test image.

Requires dataset/yolo/images/train + labels/train (the official LOCO YOLO
conversion) to exist locally -- see convert_loco_to_yolo.py.

Usage:
    python build_training_batch.py --batch-name loco_finetune_pipeline_v1
"""

import argparse
import json
import os
import random
import re
import time

import copy_paste_augment
import db
import pseudo_labels

REPLAY_BUFFER_SIZE_PER_CLASS = 150
LOCO_RATIO = 10.0
LOCO_SRC = os.path.join(db.DATASET_ROOT, "yolo")
BATCH_ROOT = os.path.join(db.PIPELINE_ROOT, "training_batches")
VAL_FRACTION = 0.2
SEED = 0

# Plain random.sample() over the LOCO pool draws images proportional to each
# class's natural frequency in LOCO -- fine for common classes, but forklift/
# pallet_truck are rare enough overall that even a large random draw at high
# loco_ratio can still under-represent them, which is what actually drove the
# forgetting plateau seen at loco_ratio 6.0-10.0 (see catastrophic-forgetting
# test notes). MIN_IMAGES_PER_LOCO_CLASS guarantees every LOCO class at least
# this many images in the train split (scaled by VAL_FRACTION for val) before
# the remaining loco_ratio budget is filled by unconstrained random sampling.
MIN_IMAGES_PER_LOCO_CLASS = 40

ENV_FRAMES_CAP = 1000
COPY_PASTE_PER_SOURCE = 3
LOCO_TEST_SUBSETS = {"1", "4"}  # the official LOCO val warehouses
LOCO_VAL_PATHS = os.path.join(db.DATASET_ROOT, "loco_val_paths.txt")
LOCO_SAM3_LABELS = os.path.join(db.DATASET_ROOT, "loco_oracle", "sam3", "labels")


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


def _classes_in_label(label_path):
    classes = set()
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if parts:
                    classes.add(int(parts[0]))
    return classes


def _ts_key(name):
    """LOCO file names differ between sources ('31834_1579163807.91.jpg' in
    the YOLO conversion, '1579163807.91.jpg' in the raw tree) -- compare on
    the timestamp part."""
    return os.path.splitext(os.path.basename(name))[0].split("_")[-1]


def assert_no_test_leak(batch_dir):
    """Raises if any train/val image is a LOCO val image, comes from a LOCO
    val warehouse, is one of a pipeline class's fixed test images, or is a
    robot frame from a held-out capture day."""
    with open(LOCO_VAL_PATHS) as f:
        test_keys = {_ts_key(line.strip()) for line in f if line.strip()}
    test_files = set()
    for cls in db.pipeline_classes():
        for rel in cls.get("test_sets", []):
            d = os.path.join(db.PROJECT_ROOT, rel, "images")
            if os.path.isdir(d):
                test_files.update(os.listdir(d))
    held_days = db.held_out_capture_days()
    for split in ("train", "val"):
        d = os.path.join(batch_dir, "images", split)
        for f in os.listdir(d):
            name = f.removeprefix("env_")  # environment frames are written as env_<capture>.jpg
            m = re.search(r"subset-(\d)", os.path.realpath(os.path.join(d, f))) or re.search(r"subset-(\d)", f)
            if (_ts_key(f) in test_keys or name in test_files or (m and m.group(1) in LOCO_TEST_SUBSETS)
                    or db.capture_day(name) in held_days):
                raise RuntimeError(f"test image leaked into {split}: {f}")
    print("leak check passed: no LOCO-val warehouse / pipeline test image / held-out capture day in train or val")


def _write_pseudo_labeled(batch_dir, img_path, stem, boxes, split="train"):
    dst_img = os.path.join(batch_dir, "images", split, stem + ".jpg")
    dst_lbl = os.path.join(batch_dir, "labels", split, stem + ".txt")
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
    if not os.path.lexists(dst_img):
        os.symlink(os.path.abspath(img_path), dst_img)
    if os.path.lexists(dst_lbl):
        os.remove(dst_lbl)  # may be a symlink into approved_pending_training/ -- never write through it
    copy_paste_augment._write_yolo_labels(dst_lbl, boxes)


def _stratified_loco_sample(loco_img_dir, loco_lbl_dir, loco_all, target_count, min_per_class, loco_class_indices):
    """Guarantees >= min_per_class images per LOCO class (rarest classes
    satisfied first, so common classes don't crowd out the budget), then
    fills the rest of target_count with plain random sampling for overall
    diversity. Total returned can exceed target_count when the per-class
    minimums require it -- representation correctness wins over hitting the
    ratio exactly."""
    images_by_class = {c: [] for c in loco_class_indices}
    for fname in loco_all:
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(loco_lbl_dir, stem + ".txt")
        for c in _classes_in_label(label_path):
            if c in images_by_class:
                images_by_class[c].append(fname)

    selected = []
    selected_set = set()
    for c in sorted(loco_class_indices, key=lambda c: len(images_by_class[c])):
        pool = [f for f in images_by_class[c] if f not in selected_set]
        random.shuffle(pool)
        take = pool[:min_per_class]
        selected.extend(take)
        selected_set.update(take)

    remaining_budget = max(0, target_count - len(selected))
    remaining_pool = [f for f in loco_all if f not in selected_set]
    random.shuffle(remaining_pool)
    selected.extend(remaining_pool[:remaining_budget])

    return selected


def _add_loco_train_pseudo(batch_dir, registry, loco_used):
    """Every LOCO train image not already in the batch (and with cached SAM3
    labels), labeled per class by its pseudo_label_source instead of by
    LOCO's human labels -- the old model's view of the old classes, which
    preserves them better than the human labels do (oracle_annotator.py)."""
    sources = {c["class_idx"]: c.get("pseudo_label_source", "production") for c in db.loco_classes(registry)}
    from_sam3 = {c for c, s in sources.items() if s == "sam3"}
    from_prod = {c for c, s in sources.items() if s == "production"}
    img_dir = os.path.join(LOCO_SRC, "images", "train")
    rest = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".jpg") and f not in loco_used and
            os.path.exists(os.path.join(LOCO_SAM3_LABELS, f"lunann_train_{os.path.splitext(f)[0]}.txt"))]
    paths = [os.path.join(img_dir, f) for f in rest]
    prod = pseudo_labels.production_boxes(paths, from_prod)
    for f, path in zip(rest, paths):
        stem = os.path.splitext(f)[0]
        sam3 = [b for b in copy_paste_augment._read_yolo_labels(
            os.path.join(LOCO_SAM3_LABELS, f"lunann_train_{stem}.txt")) if b[0] in from_sam3]
        _write_pseudo_labeled(batch_dir, path, stem, sam3 + prod[path])
    return len(rest)


def build_batch(batch_name, replay_cap=REPLAY_BUFFER_SIZE_PER_CLASS, loco_ratio=LOCO_RATIO, seed=SEED,
                 min_images_per_loco_class=MIN_IMAGES_PER_LOCO_CLASS, env_frames_cap=ENV_FRAMES_CAP,
                 use_loco_train_pseudo=True):
    random.seed(seed)
    conn = db.connect()
    registry = db.load_class_registry()
    pipeline_classes = db.pipeline_classes(registry)
    if not pipeline_classes:
        raise SystemExit("no source=pipeline classes in class_registry.json -- nothing to fine-tune on")
    loco_class_indices = {c["class_idx"] for c in db.loco_classes(registry)}
    held_days = db.held_out_capture_days(registry)
    approved_by_capture = db.approved_labels_by_capture(conn)

    batch_dir = os.path.join(BATCH_ROOT, batch_name)
    counts = {"pipeline": {"train": 0, "val": 0}, "loco": {"train": 0, "val": 0}}
    ids_to_stamp = []
    loco_used = set()

    # Assign each approved pipeline-class image to exactly one split up front --
    # sampling/splitting independently per split (as an earlier version of this
    # function did) let the same image land in both train and val.
    pipeline_split = {"train": [], "val": []}
    classes_in_train = []
    for cls in pipeline_classes:
        pool = db.approved_pool(conn, cls["name"])
        pool = [(row_id, path) for row_id, path in pool
                if os.path.exists(path) and db.capture_day(path) not in held_days]
        random.shuffle(pool)
        sample = pool[:replay_cap]
        n_val = max(1, round(len(sample) * VAL_FRACTION)) if len(sample) > 1 else 0
        pipeline_split["val"].extend(sample[:n_val])
        pipeline_split["train"].extend(sample[n_val:])
        if sample[n_val:]:
            classes_in_train.append(cls["name"])

    for split in ("train", "val"):
        for row_id, img_path in pipeline_split[split]:
            lbl_path = _label_path_for(img_path)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            dst_img = os.path.join(batch_dir, "images", split, os.path.basename(img_path))
            dst_lbl = os.path.join(batch_dir, "labels", split, stem + ".txt")
            _symlink_pair(img_path, dst_img, dst_lbl, src_lbl=lbl_path)
            counts["pipeline"][split] += 1
            ids_to_stamp.append(row_id)

        loco_img_dir = os.path.join(LOCO_SRC, "images", "train")
        loco_lbl_dir = os.path.join(LOCO_SRC, "labels", "train")
        loco_all = sorted(f for f in os.listdir(loco_img_dir) if f.endswith(".jpg") and f not in loco_used)
        loco_target = round(len(pipeline_split[split]) * loco_ratio)
        split_min_per_class = (min_images_per_loco_class if split == "train"
                                else max(1, round(min_images_per_loco_class * VAL_FRACTION)))
        loco_sample = _stratified_loco_sample(
            loco_img_dir, loco_lbl_dir, loco_all,
            target_count=min(loco_target, len(loco_all)),
            min_per_class=split_min_per_class,
            loco_class_indices=loco_class_indices,
        )
        loco_used.update(loco_sample)
        for fname in loco_sample:
            stem = os.path.splitext(fname)[0]
            dst_img = os.path.join(batch_dir, "images", split, fname)
            dst_lbl = os.path.join(batch_dir, "labels", split, stem + ".txt")
            _symlink_pair(os.path.join(loco_img_dir, fname), dst_img, dst_lbl,
                          src_lbl=os.path.join(loco_lbl_dir, stem + ".txt"))
            counts["loco"][split] += 1

    # --- pseudo-labels for the LOCO classes on approved pipeline frames
    approved = [(split, path) for split in ("train", "val") for _id, path in pipeline_split[split]]
    teacher = pseudo_labels.production_boxes([p for _s, p in approved], loco_class_indices)
    for split, path in approved:
        stem = os.path.splitext(os.path.basename(path))[0]
        # every pipeline class this capture was approved for, not just the row's own class --
        # otherwise e.g. a robot arm in an iw_hub frame would be trained as background
        human = [b for p in approved_by_capture.get(stem, [path])
                 for b in copy_paste_augment._read_yolo_labels(_label_path_for(p))]
        _write_pseudo_labeled(batch_dir, path, stem, pseudo_labels.add_without_overlap(human, teacher[path]), split)

    # --- copy-paste each pipeline class onto LOCO replay backgrounds
    counts["copy_paste"] = sum(copy_paste_augment.augment(batch_dir, name, COPY_PASTE_PER_SOURCE, seed=seed)
                               for name in classes_in_train)

    # --- environment frames: the robot's surroundings, LOCO classes from production
    env = [p for p in db.environment_frames(conn, registry)
           if os.path.exists(p) and db.capture_day(p) not in held_days]
    random.shuffle(env)
    env = env[:env_frames_cap]
    teacher = pseudo_labels.production_boxes(env, loco_class_indices)
    for path in env:
        _write_pseudo_labeled(batch_dir, path, "env_" + os.path.splitext(os.path.basename(path))[0], teacher[path])
    counts["environment"] = len(env)

    # --- the rest of LOCO train, pseudo-labeled instead of human-labeled
    counts["loco_train_pseudo"] = _add_loco_train_pseudo(batch_dir, registry, loco_used) if use_loco_train_pseudo else 0

    assert_no_test_leak(batch_dir)
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
        "loco_ratio": loco_ratio,
        "min_images_per_loco_class": min_images_per_loco_class,
        "env_frames_cap": env_frames_cap,
        "pseudo_label_conf": pseudo_labels.PSEUDO_LABEL_CONF,
        "pseudo_label_source": {c["name"]: c.get("pseudo_label_source") for c in db.loco_classes(registry)},
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
    parser.add_argument("--loco-ratio", type=float, default=LOCO_RATIO,
                         help="LOCO images sampled per pipeline-class image (default 10.0 = 10:1)")
    parser.add_argument("--min-images-per-loco-class", type=int, default=MIN_IMAGES_PER_LOCO_CLASS,
                         help="Guaranteed minimum train-split images per LOCO class, to keep rare "
                              "classes (forklift, pallet_truck) from being crowded out by random "
                              "sampling (default 40; val split gets this scaled by VAL_FRACTION)")
    parser.add_argument("--env-frames-cap", type=int, default=ENV_FRAMES_CAP)
    parser.add_argument("--no-loco-train-pseudo", action="store_true",
                         help="skip the pseudo-labeled rest of LOCO train")
    args = parser.parse_args()
    build_batch(args.batch_name, replay_cap=args.replay_cap, loco_ratio=args.loco_ratio,
                min_images_per_loco_class=args.min_images_per_loco_class, env_frames_cap=args.env_frames_cap,
                use_loco_train_pseudo=not args.no_loco_train_pseudo)


if __name__ == "__main__":
    main()
