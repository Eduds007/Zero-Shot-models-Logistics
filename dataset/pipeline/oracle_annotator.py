"""
Oracle test: is the forgetting of pallet_truck and stillage caused by the
pseudo-label annotator (low recall at 0.5 -- sam3_label_quality.py) or by
the retraining itself?

Takes the human-annotated LOCO train-split images that the self-training
base batch (self_training_robot_prodteacher_r3) doesn't use, and adds them
to that batch with three label sets -- same images, only the labels change:

  hybrid        SAM3 for small_load_carrier / stillage / pallet_truck,
                production for forklift / pallet (the pipeline's annotator)
  oracle_st_pt  as hybrid, but stillage and pallet_truck from the human labels
  oracle_all    every class from the human labels (plain extra replay)

If oracle_st_pt passes the gate where hybrid fails, the annotator is the
cause; if oracle_all fails too, the retraining is. Caveat: the production
model was trained on these images, so its pseudo-labels here are better
than on unseen images (same for every arm). The robot frames in the base
batch keep their production labels in every arm.

Training as train_with_sam3_loco.py: 3 seeds, batch 8, 30 fixed epochs,
last.pt; then promote_if_better.py --dry-run on each arm's 3 seeds.

Usage:
    python oracle_annotator.py
"""

import json
import os
import random
import shutil
import subprocess
import sys

import db
from build_training_batch import LOCO_SRC, _ts_key, assert_no_test_leak
from copy_paste_augment import _read_yolo_labels, _write_yolo_labels
from promote_if_better import evaluate_model
from pseudo_labels import production_boxes
from run_finetune import finetune
from self_training import OUT_ROOT, eval_iw_hub
from train_with_sam3_loco import BASE, BATCH_SIZE, SEEDS, replace_loco_val

ROOT = os.path.join(db.DATASET_ROOT, "loco_oracle")
RESULTS = os.path.join(OUT_ROOT, "results_oracle_annotator.json")
RESERVE = 400  # unused annotated train images left for replace_loco_val's val split
SAM3_CLASSES = {"small_load_carrier", "stillage", "pallet_truck"}
ORACLE_CLASSES = {"stillage", "pallet_truck"}
ARMS = ["hybrid", "oracle_st_pt", "oracle_all"]


def pool():
    used = set()
    for split in ("train", "val"):
        used |= {_ts_key(f) for f in os.listdir(os.path.join(BASE, "images", split))}
    files = sorted(f for f in os.listdir(os.path.join(LOCO_SRC, "images", "train"))
                   if f.endswith(".jpg") and _ts_key(f) not in used)
    random.Random(0).shuffle(files)
    return sorted(files[RESERVE:])


def _sam3_label_path(f):
    # s3.label names its outputs lunann_<parent dir>_<file>
    return os.path.join(ROOT, "sam3", "labels", f"lunann_train_{os.path.splitext(f)[0]}.txt")


def run_sam3(files):
    import sam3_label_loco_unannotated as s3

    s3.label([os.path.join(LOCO_SRC, "images", "train", f) for f in files],
             os.path.join(ROOT, "sam3"), db.load_class_registry())


def sam3_labels(files):
    """{file: [(cls, cx, cy, w, h)]} from SAM3 at 0.5, cached in ROOT/sam3.
    SAM3 runs in a subprocess: s3.label() enters a global bfloat16 autocast
    it never exits, and YOLO training in the same process then diverges
    (every AP50 = 0)."""
    if not all(os.path.exists(_sam3_label_path(f)) for f in files):
        subprocess.run([sys.executable, __file__, "--label-only"], check=True)
    return {f: _read_yolo_labels(_sam3_label_path(f)) for f in files}


def label_sets(files, registry):
    idx = {c["name"]: c["class_idx"] for c in db.loco_classes(registry)}
    sam3_c, oracle_c = {idx[n] for n in SAM3_CLASSES}, {idx[n] for n in ORACLE_CLASSES}
    prod_c = set(idx.values()) - sam3_c
    sam3 = sam3_labels(files)
    prod = production_boxes([os.path.join(LOCO_SRC, "images", "train", f) for f in files], prod_c)
    sets = {a: {} for a in ARMS}
    for f in files:
        gt = _read_yolo_labels(os.path.join(LOCO_SRC, "labels", "train", os.path.splitext(f)[0] + ".txt"))
        p = prod[os.path.join(LOCO_SRC, "images", "train", f)]
        s = [b for b in sam3[f] if b[0] in sam3_c]
        sets["hybrid"][f] = s + p
        sets["oracle_st_pt"][f] = [b for b in s if b[0] not in oracle_c] + [b for b in gt if b[0] in oracle_c] + p
        sets["oracle_all"][f] = gt
    names = {v: k for k, v in idx.items()}
    stats = {a: {names[c]: sum(b[0] == c for bs in sets[a].values() for b in bs) for c in names} for a in ARMS}
    print(f"{len(files)} images; boxes per arm: {json.dumps(stats)}", flush=True)
    return sets, stats


def build(arm, labels):
    out = os.path.join(OUT_ROOT, "batches", f"oracle_annotator_{arm}")
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(BASE, out, symlinks=True)
    for root, _dirs, fs in os.walk(out):
        for f in fs:
            if f.endswith(".cache"):
                os.remove(os.path.join(root, f))
    for f, boxes in labels.items():
        os.symlink(os.path.join(LOCO_SRC, "images", "train", f), os.path.join(out, "images", "train", f))
        _write_yolo_labels(os.path.join(out, "labels", "train", os.path.splitext(f)[0] + ".txt"), boxes)
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path) as fh:
        lines = fh.read().splitlines()
    with open(yaml_path, "w") as fh:
        fh.write("\n".join(f"path: {out}" if l.startswith("path:") else l for l in lines) + "\n")
    replace_loco_val(out)
    assert_no_test_leak(out)
    return yaml_path


def main():
    registry = db.load_class_registry()
    loco_names = [c["name"] for c in db.loco_classes(registry)]
    files = pool()
    sets, stats = label_sets(files, registry)
    report = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"runs": []}
    report.update({"n_images": len(files), "boxes": stats})
    for arm in ARMS:
        yaml_path = build(arm, sets[arm])
        for seed in SEEDS:
            if any(r["arm"] == arm and r["seed"] == seed for r in report["runs"]):
                continue
            best = finetune(yaml_path, f"oracle_annotator_{arm}_s{seed}", batch=BATCH_SIZE, seed=seed, patience=0)
            weights = os.path.join(os.path.dirname(best), "last.pt")
            report["runs"].append({"arm": arm, "seed": seed, "weights": weights,
                                   "loco": evaluate_model(weights, registry, loco_names, db.DATASET_ROOT),
                                   "iw_hub": eval_iw_hub(weights)})
            with open(RESULTS, "w") as fh:
                json.dump(report, fh, indent=2)
            print(f"{arm} SEED {seed} RESULT: {json.dumps(report['runs'][-1]['loco'])}", flush=True)
    for arm in ARMS:
        weights = [r["weights"] for r in report["runs"] if r["arm"] == arm]
        print(f"===== GATE (dry run): {arm} =====", flush=True)
        subprocess.run([sys.executable, "promote_if_better.py", "--run-name", f"oracle_annotator_{arm}",
                        "--weights", *weights, "--dry-run"], check=False)
    print("ORACLE ANNOTATOR DONE", flush=True)


if __name__ == "__main__":
    if "--label-only" in sys.argv:
        run_sam3(pool())
    else:
        main()
