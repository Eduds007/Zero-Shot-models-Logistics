"""
Adds the SAM3-labeled unannotated LOCO images (sam3_label_loco_unannotated.py)
to the corrected robot-frame self-training batch and retrains with 3 seeds,
so the only difference from that run's round 3 is the extra SAM3 images.

The base batch's val split (used by ultralytics to pick best.pt / early-stop)
holds LOCO val images -- the same images the gate scores on -- so it's
replaced by annotated LOCO *train*-split images that aren't in this batch's
train split, and assert_no_test_leak() refuses to train if any test image
(LOCO val, its warehouses 1 and 4, or the fixed iw_hub test images) is in
the batch. Those replacement images were in the pristine baseline's own
training set, though, so its val score peaks at epoch ~2 and best.pt /
early stopping would pick a barely-trained model: train the fixed 30-epoch
schedule (patience=0) and evaluate last.pt instead.

Usage:
    python train_with_sam3_loco.py --n 5000 --variant _wh235           # SAM3-only labels
    python train_with_sam3_loco.py --n 5000 --variant _wh235_hybrid    # after hybrid_relabel_sam3_loco.py
    python train_with_sam3_loco.py --n 5000 --variant _wh235_apple     # after apple_threshold_relabel.py
"""

import argparse
import json
import os
import random
import shutil

import db
from build_training_batch import LOCO_SRC, _ts_key, assert_no_test_leak
from promote_if_better import LOCO_VAL_PATHS, evaluate_model
from run_finetune import finetune
from self_training import OUT_ROOT, eval_iw_hub

BASE = os.path.join(OUT_ROOT, "batches", "self_training_robot_prodteacher_r3")
SEEDS = [0, 1, 2]
BATCH_SIZE = 8


def replace_loco_val(out):
    """Swap every LOCO-val image in the batch's val split for an annotated
    LOCO train-split image not used in the batch's train split."""
    with open(LOCO_VAL_PATHS) as f:
        test_keys = {_ts_key(l.strip()) for l in f if l.strip()}
    img_val, lbl_val = os.path.join(out, "images", "val"), os.path.join(out, "labels", "val")
    removed = [f for f in os.listdir(img_val) if _ts_key(f) in test_keys]
    for f in removed:
        os.remove(os.path.join(img_val, f))
        lbl = os.path.join(lbl_val, os.path.splitext(f)[0] + ".txt")
        if os.path.lexists(lbl):
            os.remove(lbl)
    in_train = {_ts_key(f) for f in os.listdir(os.path.join(out, "images", "train"))}
    pool = sorted(f for f in os.listdir(os.path.join(LOCO_SRC, "images", "train"))
                  if f.endswith(".jpg") and _ts_key(f) not in in_train)
    for f in random.Random(0).sample(pool, len(removed)):
        os.symlink(os.path.join(LOCO_SRC, "images", "train", f), os.path.join(img_val, f))
        stem = os.path.splitext(f)[0]
        os.symlink(os.path.join(LOCO_SRC, "labels", "train", stem + ".txt"), os.path.join(lbl_val, stem + ".txt"))
    print(f"val split: replaced {len(removed)} LOCO-val images with held-out LOCO train-split images", flush=True)


def build(n, variant=""):
    src = os.path.join(db.DATASET_ROOT, "loco_unannotated_full", f"sam3_labeled_{n}{variant}")
    out = os.path.join(OUT_ROOT, "batches", f"robot_plus_sam3loco_{n}{variant}")
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(BASE, out, symlinks=True)
    for root, _dirs, files in os.walk(out):
        for f in files:
            if f.endswith(".cache"):
                os.remove(os.path.join(root, f))
    added = 0
    for f in sorted(os.listdir(os.path.join(src, "labels"))):
        stem = os.path.splitext(f)[0]
        os.symlink(os.path.realpath(os.path.join(src, "images", stem + ".jpg")),
                   os.path.join(out, "images", "train", stem + ".jpg"))
        shutil.copy2(os.path.join(src, "labels", f), os.path.join(out, "labels", "train", f))
        added += 1
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path) as fh:
        lines = fh.read().splitlines()
    with open(yaml_path, "w") as fh:
        fh.write("\n".join(f"path: {out}" if l.startswith("path:") else l for l in lines) + "\n")
    print(f"batch {out}: +{added} SAM3-labeled LOCO images", flush=True)
    replace_loco_val(out)
    assert_no_test_leak(out)
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--variant", default="", help='label-set suffix, e.g. "_hybrid"')
    args = parser.parse_args()
    registry = db.load_class_registry()
    loco_names = [c["name"] for c in db.loco_classes(registry)]
    data_yaml = build(args.n, args.variant)
    results_path = os.path.join(OUT_ROOT, f"results_sam3loco_{args.n}{args.variant}.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else []
    done = {r["seed"] for r in results}
    for seed in SEEDS:
        if seed in done:
            continue
        best = finetune(data_yaml, f"robot_plus_sam3loco_{args.n}{args.variant}_s{seed}",
                        batch=BATCH_SIZE, seed=seed, patience=0)
        weights = os.path.join(os.path.dirname(best), "last.pt")
        results.append({"seed": seed, "weights": weights,
                        "loco": evaluate_model(weights, registry, loco_names, db.DATASET_ROOT),
                        "iw_hub": eval_iw_hub(weights)})
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"SEED {seed} RESULT: {json.dumps({k: results[-1][k] for k in ('loco', 'iw_hub')})}", flush=True)
    print("SAM3 LOCO TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
