"""
How far does SAM3 get with zero human labels? Two YOLOv8m detectors trained
from scratch (COCO-pretrained weights, as the production model loco_full_v3)
on the same LOCO images, for all 5 LOCO classes -- only the labels differ:

  human  LOCO's own annotations
  sam3   SAM3 at 0.5 with eval_loco.py's prompts (the pipeline's annotator,
         sam3_label_loco_unannotated.label), no human in the loop

Same images everywhere:
  train  the official LOCO train split (warehouses 2, 3, 5) minus val
  val    VAL_FRACTION of it (seed 0), each arm with its own labels -- the
         zero-human arm never sees a human label, not even for picking
         best.pt
  test   the official LOCO val split (warehouses 1, 4, 2277 images), human
         labels, scored as promote_if_better.py scores the gate

Recipe as loco_full_v3: yolov8m.pt, imgsz 960, batch 16, 150 epochs,
patience 40, best.pt; 3 seeds per arm, interleaved. Also reports SAM3's box
precision / recall against the human labels on the training images, and a
95% paired-bootstrap CI of sam3 - human per class and seed.

Usage:
    python loco_sam3_vs_human.py
"""

import json
import os
import random
import subprocess
import sys

import db
from build_training_batch import LOCO_SRC
from copy_paste_augment import _read_yolo_labels
from promote_if_better import compare_with_production
from run_finetune import YOLO_RUNS_DIR
from self_training import OUT_ROOT

ROOT = os.path.join(db.DATASET_ROOT, "loco_sam3_vs_human")
SAM3_DIR = os.path.join(db.DATASET_ROOT, "loco_oracle", "sam3")  # shared cache, same annotator and threshold
RESULTS = os.path.join(OUT_ROOT, "results_loco_sam3_vs_human.json")
VAL_FRACTION = 0.1
SEEDS = [0, 1, 2]
ARMS = ["human", "sam3"]
TRAIN_ARGS = {"epochs": 150, "patience": 40, "batch": 16, "imgsz": 960}


def train_files():
    return sorted(f for f in os.listdir(os.path.join(LOCO_SRC, "images", "train")) if f.endswith(".jpg"))


def sam3_label_path(f):
    # s3.label names its outputs lunann_<parent dir>_<file>
    return os.path.join(SAM3_DIR, "labels", f"lunann_train_{os.path.splitext(f)[0]}.txt")


def ensure_sam3(files):
    """SAM3 runs in its own process: it enters a global bfloat16 autocast
    that breaks YOLO training in the same process."""
    if not all(os.path.exists(sam3_label_path(f)) for f in files):
        subprocess.run([sys.executable, __file__, "--label-only"], check=True)


def build(arm, split):
    root = os.path.join(ROOT, arm)
    for s, files in split.items():
        os.makedirs(os.path.join(root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", s), exist_ok=True)
        for f in files:
            img = os.path.join(root, "images", s, f)
            if not os.path.lexists(img):
                os.symlink(os.path.join(LOCO_SRC, "images", "train", f), img)
            src = (os.path.join(LOCO_SRC, "labels", "train", os.path.splitext(f)[0] + ".txt") if arm == "human"
                   else sam3_label_path(f))
            lbl = os.path.join(root, "labels", s, os.path.splitext(f)[0] + ".txt")
            if not os.path.lexists(lbl):
                os.symlink(src, lbl)
    names = {c["class_idx"]: c["name"] for c in db.loco_classes()}
    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {root}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n")
    return yaml_path


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def label_quality(files):
    """SAM3 box precision / recall per class against the human labels
    (greedy one-to-one matching at IoU 0.5; IoU is scale-invariant, so
    normalized YOLO coordinates do)."""
    names = {c["class_idx"]: c["name"] for c in db.loco_classes()}
    stats = {n: {"tp": 0, "fp": 0, "n_gt": 0} for n in names.values()}
    for f in files:
        gt = _read_yolo_labels(os.path.join(LOCO_SRC, "labels", "train", os.path.splitext(f)[0] + ".txt"))
        pred = _read_yolo_labels(sam3_label_path(f))
        for c, n in names.items():
            free = [b for b in gt if b[0] == c]
            stats[n]["n_gt"] += len(free)
            for b in (b for b in pred if b[0] == c):
                best = max(free, key=lambda g: _iou(b, g), default=None)
                if best is not None and _iou(b, best) >= 0.5:
                    stats[n]["tp"] += 1
                    free.remove(best)
                else:
                    stats[n]["fp"] += 1
    for s in stats.values():
        s["precision"] = round(s["tp"] / (s["tp"] + s["fp"]), 4) if s["tp"] + s["fp"] else None
        s["recall"] = round(s["tp"] / s["n_gt"], 4) if s["n_gt"] else None
    return stats


def main():
    from ultralytics import YOLO

    registry = db.load_class_registry()
    loco = [c for c in registry if c["source"] == "loco"]
    loco_names = [c["name"] for c in loco]
    files = train_files()
    ensure_sam3(files)
    shuffled = files[:]
    random.Random(0).shuffle(shuffled)
    n_val = round(len(files) * VAL_FRACTION)
    split = {"train": sorted(shuffled[n_val:]), "val": sorted(shuffled[:n_val])}
    yamls = {arm: build(arm, split) for arm in ARMS}

    report = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"runs": [], "comparisons": []}
    report["split"] = {s: len(v) for s, v in split.items()}
    report["sam3_label_quality_train"] = label_quality(split["train"])
    print(f"split {report['split']}; SAM3 label quality: {json.dumps(report['sam3_label_quality_train'])}", flush=True)
    for seed in SEEDS:
        for arm in ARMS:
            if any(r["arm"] == arm and r["seed"] == seed for r in report["runs"]):
                continue
            name = f"loco_{arm}_scratch_s{seed}"
            YOLO("yolov8m.pt").train(data=yamls[arm], seed=seed, project=YOLO_RUNS_DIR, name=name, exist_ok=True,
                                     **TRAIN_ARGS)
            weights = os.path.join(YOLO_RUNS_DIR, name, "weights", "best.pt")
            report["runs"].append({"arm": arm, "seed": seed, "weights": weights})
            with open(RESULTS, "w") as fh:
                json.dump(report, fh, indent=2)
        if any(c["seed"] == seed for c in report["comparisons"]):
            continue
        w = {r["arm"]: r["weights"] for r in report["runs"] if r["seed"] == seed}
        sam3, human, ci = compare_with_production(w["sam3"], w["human"], loco, loco_names, set(loco_names),
                                                  db.DATASET_ROOT)
        report["comparisons"].append({"seed": seed, "human": human, "sam3": sam3,
                                      "ci_sam3_minus_human": {n: list(v) for n, v in ci.items()}})
        with open(RESULTS, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"SEED {seed} RESULT: human={json.dumps(human)} sam3={json.dumps(sam3)} ci={json.dumps(ci)}",
              flush=True)
    print("SAM3 VS HUMAN DONE", flush=True)


if __name__ == "__main__":
    if "--label-only" in sys.argv:
        import sam3_label_loco_unannotated as s3

        s3.label([os.path.join(LOCO_SRC, "images", "train", f) for f in train_files()], SAM3_DIR,
                 db.load_class_registry())
    else:
        main()
