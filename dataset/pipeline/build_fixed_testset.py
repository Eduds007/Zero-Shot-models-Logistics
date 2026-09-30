"""
Builds ONE permanent, fixed test set mixing (a) a stratified sample of the
official LOCO val split (dataset/loco_val_paths.txt -- the fixed 2277-image
split, never touched by training), (b) a fixed slice of the robot's own
approved iw_hub captures (real lab video frames), carved out and
PERMANENTLY EXCLUDED from the training-eligible replay pool so this test
set never leaks into a future training round, and (c) optionally a fixed
sample of the Roboflow held_out_val/iw_hub set (--add-roboflow) -- that
pool is already permanent and training never touches it, so sampling from
it needs no carve-out/DB update, just a fixed, seeded pick.

This closes a gap flagged earlier in the project: run_finetune.py's own
per-epoch validation numbers came from build_training_batch.py's internal
train/val split, which was reshuffled from a SQL query with no ORDER BY --
not guaranteed to be the same images across rounds (fixed alongside this,
see db.approved_pool()). Every number this test set produces from now on
is against the exact same images, forever.

Run ONCE per class addition -- refuses to re-run any step whose output
directory already exists, since re-running would resample different
images (breaking comparability with anything already evaluated against
the current fixed set).

Usage:
    python build_fixed_testset.py [--n-per-loco-class 40] [--n-iw-hub 30]
    python build_fixed_testset.py --add-roboflow 50
"""

import argparse
import json
import os
import random
import shutil
import time

import db

FIXED_TESTSET_DIR = os.path.join(db.PIPELINE_ROOT, "fixed_testset")
LOCO_PATHS_FILE = os.path.join(db.PIPELINE_ROOT, "fixed_testset_loco_paths.txt")
LOCO_VAL_PATHS = os.path.join(db.DATASET_ROOT, "loco_val_paths.txt")
LOCO_ANNOTATIONS = os.path.join(db.DATASET_ROOT, "annotations", "loco-all-v1.json")
MANIFEST_PATH = os.path.join(FIXED_TESTSET_DIR, "manifest.json")
SEED = 0


def _select_stratified_loco_paths(n_per_class, seed=SEED):
    """Rarest-class-first stratified sample over the OFFICIAL val split only
    -- same rarity-first idea as build_training_batch.py's
    _stratified_loco_sample, but reading COCO annotations directly (no YOLO
    label files exist for the official LOCO split) and with no random
    top-up phase, since a test set doesn't need a target total size, just
    good per-class coverage."""
    random.seed(seed)
    with open(LOCO_VAL_PATHS) as f:
        val_paths = {line.strip().lstrip("/") for line in f if line.strip()}
    with open(LOCO_ANNOTATIONS) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"] if img["path"].lstrip("/") in val_paths}
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    images_by_class = {name: set() for name in cat_names.values()}
    for ann in coco["annotations"]:
        img = images_by_id.get(ann["image_id"])
        if img is not None:
            images_by_class[cat_names[ann["category_id"]]].add(img["path"])

    selected = set()
    for name in sorted(images_by_class, key=lambda n: len(images_by_class[n])):
        pool = [p for p in images_by_class[name] if p not in selected]
        random.shuffle(pool)
        selected.update(pool[:n_per_class])

    counts = {name: len(images_by_class[name] & selected) for name in images_by_class}
    return sorted(selected), counts


def _carve_out_iw_hub_slice(n_iw_hub, seed=SEED):
    """Copies the resolved real files (approved_pending_training/iw_hub's
    images are themselves symlinks to the original synced capture) into the
    fixed test set, removes the symlink+label from the training-eligible
    directory, and flips those rows' review_status in the DB so
    approved_pool() -- which drives replay sampling -- never sees them
    again."""
    random.seed(seed + 1)
    cls = db.class_by_name("iw_hub")
    src_dir = os.path.join(db.PROJECT_ROOT, cls["replay_dir"])
    src_images, src_labels = os.path.join(src_dir, "images"), os.path.join(src_dir, "labels")
    dst_images = os.path.join(FIXED_TESTSET_DIR, "iw_hub", "images")
    dst_labels = os.path.join(FIXED_TESTSET_DIR, "iw_hub", "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    all_images = sorted(f for f in os.listdir(src_images) if f.endswith(".jpg"))
    if len(all_images) < n_iw_hub:
        raise SystemExit(f"only {len(all_images)} approved iw_hub images available, need {n_iw_hub}")
    random.shuffle(all_images)
    chosen = all_images[:n_iw_hub]

    conn = db.connect()
    for fname in chosen:
        stem = os.path.splitext(fname)[0]
        real_img = os.path.realpath(os.path.join(src_images, fname))
        shutil.copy2(real_img, os.path.join(dst_images, fname))
        os.remove(os.path.join(src_images, fname))
        src_lbl = os.path.join(src_labels, stem + ".txt")
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, os.path.join(dst_labels, stem + ".txt"))
        conn.execute(
            "UPDATE frames SET review_status = 'held_out_test' "
            "WHERE class_name = 'iw_hub' AND review_status = 'approved' AND local_path LIKE ?",
            (f"%/{fname}",),
        )
    conn.commit()
    return chosen


def _add_roboflow_slice(n_roboflow, seed=SEED):
    """Fixed, seeded sample of held_out_val/iw_hub (Roboflow) -- already
    permanent/never used in training, so this only ever copies, no carve-out
    or DB update needed. Refuses to re-run if the slice already exists, so
    the composition stays fixed once created."""
    random.seed(seed + 2)
    cls = db.class_by_name("iw_hub")
    src_dir = os.path.join(db.PROJECT_ROOT, cls["held_out_val"])
    src_images, src_labels = os.path.join(src_dir, "images"), os.path.join(src_dir, "labels")
    dst_images = os.path.join(FIXED_TESTSET_DIR, "iw_hub_roboflow", "images")
    dst_labels = os.path.join(FIXED_TESTSET_DIR, "iw_hub_roboflow", "labels")
    if os.path.exists(dst_images):
        raise SystemExit(
            f"{dst_images} already exists -- refusing to resample it. Delete it by hand first "
            f"if you really mean to rebuild this slice."
        )
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    all_images = sorted(f for f in os.listdir(src_images) if f.endswith(".jpg"))
    if len(all_images) < n_roboflow:
        raise SystemExit(f"only {len(all_images)} held_out_val/iw_hub images available, need {n_roboflow}")
    random.shuffle(all_images)
    chosen = all_images[:n_roboflow]

    for fname in chosen:
        stem = os.path.splitext(fname)[0]
        shutil.copy2(os.path.join(src_images, fname), os.path.join(dst_images, fname))
        src_lbl = os.path.join(src_labels, stem + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(dst_labels, stem + ".txt"))
    return chosen


def add_roboflow(n_roboflow, seed=SEED):
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"{FIXED_TESTSET_DIR} doesn't exist yet -- run the base build first")
    chosen = _add_roboflow_slice(n_roboflow, seed)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    manifest["iw_hub_roboflow"] = {
        "n_images": len(chosen), "source": "held_out_val/iw_hub (Roboflow, cross-domain)",
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"iw_hub_roboflow: {len(chosen)} Roboflow images -> {FIXED_TESTSET_DIR}/iw_hub_roboflow/")
    print(f"manifest updated: {MANIFEST_PATH}")


def build(n_per_loco_class, n_iw_hub, seed=SEED):
    if os.path.exists(FIXED_TESTSET_DIR):
        raise SystemExit(
            f"{FIXED_TESTSET_DIR} already exists -- refusing to rebuild it (would carve out a "
            f"different iw_hub slice and break comparability with prior eval_fixed_testset.py runs). "
            f"Delete it by hand first if you really mean to rebuild the fixed test set from scratch."
        )
    os.makedirs(FIXED_TESTSET_DIR, exist_ok=True)

    loco_paths, loco_counts = _select_stratified_loco_paths(n_per_loco_class, seed)
    with open(LOCO_PATHS_FILE, "w") as f:
        f.write("\n".join(loco_paths) + "\n")

    iw_hub_files = _carve_out_iw_hub_slice(n_iw_hub, seed)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "loco": {"n_per_class_target": n_per_loco_class, "total_images": len(loco_paths), "counts": loco_counts},
        "iw_hub": {"n_images": len(iw_hub_files), "source": "approved_pending_training/iw_hub (real lab video)"},
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"LOCO: {len(loco_paths)} images across 5 classes -> {LOCO_PATHS_FILE}")
    print(f"  per-class counts: {loco_counts}")
    print(f"iw_hub: {len(iw_hub_files)} real lab-video images -> {FIXED_TESTSET_DIR}/iw_hub/")
    print(f"manifest: {MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-loco-class", type=int, default=40)
    parser.add_argument("--n-iw-hub", type=int, default=30)
    parser.add_argument("--add-roboflow", type=int, default=None,
                         help="Add N images from the Roboflow held_out_val/iw_hub set to an already-built "
                              "fixed test set, instead of building the base set")
    args = parser.parse_args()
    if args.add_roboflow is not None:
        add_roboflow(args.add_roboflow)
    else:
        build(args.n_per_loco_class, args.n_iw_hub)


if __name__ == "__main__":
    main()
