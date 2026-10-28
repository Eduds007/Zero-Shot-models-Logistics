"""
Second pipeline class, offline: does the cycle add the KUKA arm
("robotic_arm") on top of a model that already knows iw_hub, without
forgetting the 5 LOCO classes or iw_hub?

Only frames already captured are used -- no new recording:
  train  the g_loco_train_pseudo batch of baselines.py (iw_hub frames, LOCO
         replay, iw_hub copy-paste, environment frames, LOCO train
         pseudo-labels), minus every frame of the 2026-10-05 capture, plus
         SAM3 "robotic arm" boxes at 0.5 on every robot frame left in it
         (the arm is in 66 of the 83 iw_hub frames -- without these labels
         it would be trained as background)
  test   fixed_testset/robotic_arm: 84 frames of the 2026-10-05 capture,
         SAM3 pre-labels corrected by hand (109 arms); the split is by day so
         near-duplicate video frames don't cross it

Base and reference: baselines.py's g_loco_train_pseudo seed 2 (best LOCO
sum of the 3 seeds), standing in for a promoted iw_hub model -- none passed
the real gate. It is not production: nothing here touches the database,
class_registry.json, production or the cron. Caveats: g itself saw the
10-05 frames as environment frames (the arm as background), which can only
make the test harder; with freeze=21 the new nc=7 head's last class conv is
re-initialized, as in every pipeline round.

Training as baselines.py: freeze=21, imgsz 960, batch 8, 30 fixed epochs,
last.pt, 3 seeds. Gate as promote_if_better.py, but against g seed 2: no
LOCO class or iw_hub may regress (95% paired-bootstrap CI entirely below
-0.01), and the arm, new, needs AP50 >= MIN_AP50_FOR_NEW_CLASS.

Usage:
    python kuka_arm_experiment.py
"""

import copy
import json
import os
import subprocess
import sys

import cv2

import db
from baselines import BATCH_SIZE, SEEDS, VARIANTS, build
from build_training_batch import assert_no_test_leak
from copy_paste_augment import _read_yolo_labels, _write_yolo_labels
from promote_if_better import MIN_AP50_FOR_NEW_CLASS, REGRESSION_TOLERANCE, compare_with_production, decide_promotion
from pseudo_labels import add_without_overlap
from run_finetune import YOLO_RUNS_DIR, finetune
from self_training import OUT_ROOT

ARM = {"name": "robotic_arm", "class_idx": 6, "source": "pipeline", "prompt": "robotic arm",
       "test_sets": ["dataset/pipeline/fixed_testset/robotic_arm"]}
TEST_DAY = "20261005"
BASE = os.path.join(YOLO_RUNS_DIR, "baseline_g_loco_train_pseudo_s2", "weights", "last.pt")
BATCH = "kuka_arm_g"
SAM3_DIR = os.path.join(db.PIPELINE_ROOT, "kuka_arm_sam3")
RESULTS = os.path.join(OUT_ROOT, "results_kuka_arm.json")


def registry():
    return copy.deepcopy(db.load_class_registry()) + [ARM]


def robot_images(batch_dir):
    """(split, stem, image path) of every robot frame in the batch."""
    out = []
    for split in ("train", "val"):
        for f in sorted(os.listdir(os.path.join(batch_dir, "images", split))):
            if f.startswith(("capture_", "env_capture_")):
                out.append((split, os.path.splitext(f)[0], os.path.realpath(os.path.join(batch_dir, "images", split, f))))
    return out


def label_arm(paths):
    """SAM3 "robotic arm" at 0.5 -> SAM3_DIR/<file stem>.txt (class 6). Runs
    in its own process (--label-only): SAM3 enters a global bfloat16
    autocast that breaks YOLO training in the same process."""
    from sam3_label_quality import CONF, sam3_boxes

    boxes = sam3_boxes(paths, {"arm": ARM["prompt"]})
    os.makedirs(SAM3_DIR, exist_ok=True)
    for p in paths:
        h, w = cv2.imread(p).shape[:2]
        _write_yolo_labels(os.path.join(SAM3_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt"),
                           [(ARM["class_idx"], (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
                            for x0, y0, x1, y1, s in boxes[p]["arm"] if s >= CONF])


def build_batch():
    yaml_path, counts = build(BATCH, VARIANTS["g_loco_train_pseudo"])
    batch_dir = os.path.dirname(yaml_path)
    dropped = 0
    for split, stem, path in robot_images(batch_dir):
        if f"capture_{TEST_DAY}_" in path:
            os.remove(os.path.join(batch_dir, "images", split, stem + ".jpg"))
            os.remove(os.path.join(batch_dir, "labels", split, stem + ".txt"))
            dropped += 1
    frames = robot_images(batch_dir)
    paths = sorted({p for _s, _st, p in frames})
    if not all(os.path.exists(os.path.join(SAM3_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")) for p in paths):
        list_file = os.path.join(batch_dir, "sam3_paths.txt")
        with open(list_file, "w") as f:
            f.write("\n".join(paths) + "\n")
        subprocess.run([sys.executable, __file__, "--label-only", list_file], check=True)
    n_arm = n_with = 0
    for split, stem, path in frames:
        lbl = os.path.join(batch_dir, "labels", split, stem + ".txt")
        arm = _read_yolo_labels(os.path.join(SAM3_DIR, os.path.splitext(os.path.basename(path))[0] + ".txt"))
        _write_yolo_labels(lbl, add_without_overlap(_read_yolo_labels(lbl), arm))
        n_arm += len(arm)
        n_with += bool(arm)
    reg = registry()
    with open(yaml_path, "w") as f:
        f.write(f"path: {batch_dir}\ntrain: images/train\nval: images/val\n")
        f.write(f"names: {json.dumps({c['class_idx']: c['name'] for c in reg})}\n")
    assert_no_test_leak(batch_dir)
    counts |= {"dropped_test_day_frames": dropped, "robot_frames": len(frames), "frames_with_arm": n_with,
               "arm_boxes": n_arm}
    print(f"batch {BATCH}: {counts}", flush=True)
    return yaml_path, counts


def gate(weights, reg):
    targets = [c["name"] for c in reg]
    known = set(targets) - {ARM["name"]}
    cand, ref, ci = compare_with_production(weights, BASE, reg, targets, known, db.DATASET_ROOT)
    passed, failures = decide_promotion(targets, cand, ref, {ARM["name"], "iw_hub"}, ci,
                                        REGRESSION_TOLERANCE, MIN_AP50_FOR_NEW_CLASS)
    return {"scores": cand, "reference": ref, "diff_ci": {n: list(v) for n, v in ci.items()},
            "passed": passed, "failures": failures}


def main():
    reg = registry()
    report = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"runs": []}
    if len(report["runs"]) < len(SEEDS):
        yaml_path, report["batch"] = build_batch()
    for seed in SEEDS:
        if any(r["seed"] == seed for r in report["runs"]):
            continue
        best = finetune(yaml_path, f"{BATCH}_s{seed}", baseline=BASE, batch=BATCH_SIZE, seed=seed, patience=0)
        weights = os.path.join(os.path.dirname(best), "last.pt")
        report["runs"].append({"seed": seed, "weights": weights, **gate(weights, reg)})
        with open(RESULTS, "w") as f:
            json.dump(report, f, indent=2)
        r = report["runs"][-1]
        print(f"SEED {seed} RESULT: passed={r['passed']} scores={json.dumps(r['scores'])} "
              f"ci={json.dumps(r['diff_ci'])}", flush=True)
        for fail in r["failures"]:
            print(f" - {fail}", flush=True)
    print(f"reference (g seed 2): {json.dumps(report['runs'][0]['reference'])}", flush=True)
    print(f"KUKA ARM DONE: {sum(r['passed'] for r in report['runs'])}/{len(report['runs'])} seeds pass", flush=True)


if __name__ == "__main__":
    if "--label-only" in sys.argv:
        with open(sys.argv[sys.argv.index("--label-only") + 1]) as fh:
            label_arm([line.strip() for line in fh if line.strip()])
    else:
        main()
