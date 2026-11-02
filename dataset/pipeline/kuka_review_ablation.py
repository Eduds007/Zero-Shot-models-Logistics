"""
How much does the human review add for a new class? The robotic_arm round
of the pipeline trained twice, everything equal but the arm frames:

  human  the arm frames a human approved in the review queue (SAM3 boxes)
  sam3   every arm frame SAM3's quality gate passed (mean confidence > 0.5),
         same SAM3 boxes, nobody looking -- zero human in the loop

Both arms are built by build_training_batch.build_batch() (recipe g, the
held-out capture days excluded) on a copy of pipeline.db, so nothing here
touches the real database, the registry, production or the cron. iw_hub,
LOCO replay, copy-paste, environment frames and LOCO train pseudo-labels
come out of the same code with the same seed. Training as
auto_train_trigger.py: batch 8, 30 fixed epochs, last.pt, 3 seeds.

Scored on every registry class (LOCO val, iw_hub test sets, the hand-
corrected fixed_testset/robotic_arm), plus a 95% paired-bootstrap CI of
sam3 - human per class and seed.

Run it after the review queue is done (decisions applied).

Usage:
    python kuka_review_ablation.py
"""

import json
import os
import shutil
import subprocess
import sys

import cv2

import db
from copy_paste_augment import _write_yolo_labels
from promote_if_better import compare_with_production, evaluate_model
from run_finetune import finetune
from self_training import OUT_ROOT

ARM = "robotic_arm"
SEEDS = [0, 1, 2]
BATCH_SIZE = 8
WORK = os.path.join(db.PIPELINE_ROOT, "kuka_review_ablation")
SAM3_DIR = os.path.join(WORK, "sam3_approved", ARM)  # images/ + labels/, like approved_pending_training/<class>
RESULTS = os.path.join(OUT_ROOT, "results_kuka_review_ablation.json")


def sam3_pass_frames(conn):
    """(row id, original capture path) of every arm row SAM3 passed, outside
    the held-out capture days."""
    held = db.held_out_capture_days()
    rows = conn.execute("SELECT id, captured_at FROM frames WHERE class_name = ? AND sam3_status = 'pass' ORDER BY id",
                        (ARM,)).fetchall()
    incoming = os.path.join(db.DATASET_ROOT, "incoming")
    out = []
    for row_id, captured_at in rows:
        robot = row_id[: -len(f"_{captured_at}_{ARM}")]
        path = os.path.join(incoming, robot, captured_at + ".jpg")
        if db.capture_day(path) not in held:
            out.append((row_id, path))
    return out


def label_frames(paths):
    """SAM3 arm boxes at the gate's 0.5 -> SAM3_DIR/labels (class_idx of the
    arm). Own process: SAM3 leaves a global bfloat16 autocast behind."""
    from sam3_label_quality import CONF, sam3_boxes

    idx = db.class_by_name(ARM)["class_idx"]
    prompt = db.class_by_name(ARM)["prompt"]
    boxes = sam3_boxes(paths, {"arm": prompt})
    os.makedirs(os.path.join(SAM3_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(SAM3_DIR, "labels"), exist_ok=True)
    for p in paths:
        h, w = cv2.imread(p).shape[:2]
        stem = os.path.splitext(os.path.basename(p))[0]
        img = os.path.join(SAM3_DIR, "images", stem + ".jpg")
        if not os.path.lexists(img):
            os.symlink(p, img)
        _write_yolo_labels(os.path.join(SAM3_DIR, "labels", stem + ".txt"),
                           [(idx, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
                            for x0, y0, x1, y1, s in boxes[p]["arm"] if s >= CONF])


def build(arm):
    """Batch for one arm on its own copy of pipeline.db."""
    db_copy = os.path.join(WORK, f"pipeline_{arm}.db")
    shutil.copy(db.DB_PATH, db_copy)
    real_connect = db.connect
    db.connect = lambda db_path=None: real_connect(db_copy)
    try:
        conn = db.connect()
        if arm == "sam3":
            frames = sam3_pass_frames(conn)
            conn.executemany("UPDATE frames SET review_status = 'approved', local_path = ? WHERE id = ?",
                             [(os.path.join(SAM3_DIR, "images", os.path.basename(p)), i) for i, p in frames])
            conn.commit()
        n_arm = conn.execute("SELECT COUNT(*) FROM frames WHERE class_name = ? AND review_status = 'approved'",
                             (ARM,)).fetchone()[0]
        import build_training_batch

        name = f"ablation_review_{arm}"
        shutil.rmtree(os.path.join(build_training_batch.BATCH_ROOT, name), ignore_errors=True)
        yaml_path = build_training_batch.build_batch(name)
    finally:
        db.connect = real_connect
    return yaml_path, n_arm


def main():
    os.makedirs(WORK, exist_ok=True)
    registry = db.load_class_registry()
    names = [c["name"] for c in registry]
    frames = sam3_pass_frames(db.connect())
    if not all(os.path.exists(os.path.join(SAM3_DIR, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"))
               for _i, p in frames):
        subprocess.run([sys.executable, __file__, "--label-only"], check=True)

    report = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"arms": {}, "runs": [], "comparisons": []}
    for arm in ("human", "sam3"):
        if all(any(r["arm"] == arm and r["seed"] == s for r in report["runs"]) for s in SEEDS):
            continue
        yaml_path, n_arm = build(arm)
        report["arms"][arm] = {"arm_frames": n_arm,
                               "manifest": json.load(open(os.path.join(os.path.dirname(yaml_path), "manifest.json")))["counts"]}
        for seed in SEEDS:
            if any(r["arm"] == arm and r["seed"] == seed for r in report["runs"]):
                continue
            best = finetune(yaml_path, f"ablation_review_{arm}_s{seed}", batch=BATCH_SIZE, seed=seed, patience=0)
            weights = os.path.join(os.path.dirname(best), "last.pt")
            report["runs"].append({"arm": arm, "seed": seed, "weights": weights,
                                   "scores": evaluate_model(weights, registry, names, db.DATASET_ROOT)})
            with open(RESULTS, "w") as f:
                json.dump(report, f, indent=2)
            print(f"{arm} SEED {seed}: {json.dumps(report['runs'][-1]['scores'])}", flush=True)

    for seed in SEEDS:
        if any(c["seed"] == seed for c in report["comparisons"]):
            continue
        w = {r["arm"]: r["weights"] for r in report["runs"] if r["seed"] == seed}
        sam3, human, ci = compare_with_production(w["sam3"], w["human"], registry, names, set(names), db.DATASET_ROOT)
        report["comparisons"].append({"seed": seed, "human": human, "sam3": sam3,
                                      "ci_sam3_minus_human": {n: list(v) for n, v in ci.items()}})
        with open(RESULTS, "w") as f:
            json.dump(report, f, indent=2)
        print(f"SEED {seed} RESULT: human={json.dumps(human)} sam3={json.dumps(sam3)} ci={json.dumps(ci)}", flush=True)
    print(f"arm frames: {json.dumps({a: v['arm_frames'] for a, v in report['arms'].items()})}", flush=True)
    print("REVIEW ABLATION DONE", flush=True)


if __name__ == "__main__":
    if "--label-only" in sys.argv:
        label_frames([p for _i, p in sam3_pass_frames(db.connect())])
    else:
        main()
