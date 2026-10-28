"""
Item 3 of the thesis plan: every anti-forgetting strategy under one final
protocol, so the differences come from the strategy alone.

Shared by every single-model variant:
  - the same approved iw_hub frames (db.approved_pool, seed 0, 80/20 split);
  - start from the pristine LOCO model (run_finetune.BASELINE), freeze=21,
    imgsz 960, batch 8, 30 fixed epochs, last.pt (no early stopping);
  - 3 training seeds per variant on one fixed batch;
  - val split never touches LOCO val (LOCO train images instead), and
    assert_no_test_leak() before training;
  - scored by promote_if_better.py --dry-run (95% paired-bootstrap CI vs
    production on LOCO val; iw_hub on its fixed test sets).

Variants (each adds one ingredient to an earlier one):
  a_naive              iw_hub frames only (iw_hub labels only)
  b_replay             a + LOCO replay (human labels, 10 LOCO images per frame)
  c_replay_distill     b + ultralytics feature distillation from production
  d_copy_paste         b + copy-paste of iw_hub onto LOCO replay images
  e_self_training      d + production pseudo-labels for the LOCO classes on the
                       iw_hub frames + environment frames (production labels)
  f_hybrid_bank        e + the LOCO bank (5000 unannotated LOCO images, hybrid
                       SAM3/production labels) -- the pipeline's current recipe
  g_loco_train_pseudo  e + the LOCO train images not used above, with hybrid
                       pseudo-labels instead of their human labels (the arm
                       that passed the gate once in oracle_annotator.py; only
                       images whose SAM3 labels are cached in loco_oracle/sam3)
  h_joint              iw_hub frames (+ production labels for LOCO classes) +
                       every LOCO train image with its human labels
  two_model            production untouched + a YOLOv8s iw_hub specialist
                       (train_specialist.py, 3 seeds), two_model_gate.py

Nothing here touches the database or production.

Usage:
    python baselines.py                    # everything, resumable
    python baselines.py --only b_replay    # one variant
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import copy_paste_augment
import db
import pseudo_labels
from build_training_batch import (COPY_PASTE_PER_SOURCE, ENV_FRAMES_CAP, LOCO_RATIO, LOCO_SRC,
                                  MIN_IMAGES_PER_LOCO_CLASS, VAL_FRACTION, _label_path_for, _stratified_loco_sample,
                                  _symlink_pair, _write_pseudo_labeled, assert_no_test_leak)
from promote_if_better import PRODUCTION_WEIGHTS, evaluate_model
from run_finetune import finetune
from self_training import OUT_ROOT, eval_iw_hub

BATCH_ROOT = os.path.join(db.PIPELINE_ROOT, "baseline_batches")
RESULTS = os.path.join(OUT_ROOT, "results_baselines.json")
ORACLE_SAM3 = os.path.join(db.DATASET_ROOT, "loco_oracle", "sam3", "labels")
SEEDS = [0, 1, 2]
BATCH_SIZE = 8
SEED = 0

VARIANTS = {
    "a_naive": set(),
    "b_replay": {"replay"},
    "c_replay_distill": {"replay", "distill"},
    "d_copy_paste": {"replay", "copy_paste"},
    "e_self_training": {"replay", "copy_paste", "pseudo_frames", "env"},
    "f_hybrid_bank": {"replay", "copy_paste", "pseudo_frames", "env", "loco_bank"},
    "g_loco_train_pseudo": {"replay", "copy_paste", "pseudo_frames", "env", "loco_train_pseudo"},
    "h_joint": {"pseudo_frames", "loco_all"},
}


def _loco_pair(batch_dir, fname, split, label=True):
    stem = os.path.splitext(fname)[0]
    _symlink_pair(os.path.join(LOCO_SRC, "images", "train", fname),
                  os.path.join(batch_dir, "images", split, fname), os.path.join(batch_dir, "labels", split, stem + ".txt"),
                  src_lbl=os.path.join(LOCO_SRC, "labels", "train", stem + ".txt") if label else None)


def build(name, parts):
    random.seed(SEED)
    registry = db.load_class_registry()
    loco_idx = {c["class_idx"] for c in db.loco_classes(registry)}
    batch_dir = os.path.join(BATCH_ROOT, name)
    if os.path.exists(batch_dir):
        shutil.rmtree(batch_dir)
    counts = {}

    # iw_hub frames: same sample and split in every variant
    conn = db.connect()
    split = {"train": [], "val": []}
    for cls in db.pipeline_classes(registry):
        pool = [p for _id, p in db.approved_pool(conn, cls["name"]) if os.path.exists(p)]
        random.Random(SEED).shuffle(pool)
        n_val = max(1, round(len(pool) * VAL_FRACTION))
        split["val"] += pool[:n_val]
        split["train"] += pool[n_val:]
    frames = [(s, p) for s in split for p in split[s]]
    teacher = pseudo_labels.production_boxes([p for _s, p in frames], loco_idx) if "pseudo_frames" in parts else {}
    for s, p in frames:
        stem = os.path.splitext(os.path.basename(p))[0]
        human = copy_paste_augment._read_yolo_labels(_label_path_for(p))
        _write_pseudo_labeled(batch_dir, p, stem, pseudo_labels.add_without_overlap(human, teacher.get(p, [])), s)
    counts["iw_hub_frames"] = {s: len(split[s]) for s in split}

    # LOCO val split (always; LOCO train images): sized like the replay's val side
    loco_all = sorted(f for f in os.listdir(os.path.join(LOCO_SRC, "images", "train")) if f.endswith(".jpg"))
    used = set()
    n_val = round(len(split["val"]) * LOCO_RATIO)
    val_loco = _stratified_loco_sample(os.path.join(LOCO_SRC, "images", "train"),
                                       os.path.join(LOCO_SRC, "labels", "train"), loco_all, n_val,
                                       max(1, round(MIN_IMAGES_PER_LOCO_CLASS * VAL_FRACTION)), loco_idx)
    for f in val_loco:
        _loco_pair(batch_dir, f, "val")
    used.update(val_loco)

    if "replay" in parts:
        replay = _stratified_loco_sample(os.path.join(LOCO_SRC, "images", "train"),
                                         os.path.join(LOCO_SRC, "labels", "train"),
                                         [f for f in loco_all if f not in used],
                                         round(len(split["train"]) * LOCO_RATIO), MIN_IMAGES_PER_LOCO_CLASS, loco_idx)
        for f in replay:
            _loco_pair(batch_dir, f, "train")
        used.update(replay)
        counts["loco_replay"] = len(replay)

    if "loco_all" in parts:
        rest = [f for f in loco_all if f not in used]
        for f in rest:
            _loco_pair(batch_dir, f, "train")
        used.update(rest)
        counts["loco_all_human"] = len(rest)

    if "copy_paste" in parts:
        counts["copy_paste"] = sum(copy_paste_augment.augment(batch_dir, c["name"], COPY_PASTE_PER_SOURCE, seed=SEED)
                                   for c in db.pipeline_classes(registry))

    if "env" in parts:
        env = [p for p in db.environment_frames(conn, registry) if os.path.exists(p)]
        random.Random(SEED).shuffle(env)
        env = env[:ENV_FRAMES_CAP]
        teacher = pseudo_labels.production_boxes(env, loco_idx)
        for p in env:
            _write_pseudo_labeled(batch_dir, p, "env_" + os.path.splitext(os.path.basename(p))[0], teacher[p])
        counts["environment"] = len(env)

    if "loco_bank" in parts:
        bank = pseudo_labels.loco_bank(registry)
        for p, stem, boxes in bank:
            _write_pseudo_labeled(batch_dir, p, stem, boxes)
        counts["loco_bank"] = len(bank)

    if "loco_train_pseudo" in parts:
        sources = {c["class_idx"]: c.get("pseudo_label_source", "production") for c in db.loco_classes(registry)}
        from_sam3 = {c for c, s in sources.items() if s == "sam3"}
        from_prod = {c for c, s in sources.items() if s == "production"}
        rest = [f for f in loco_all if f not in used and
                os.path.exists(os.path.join(ORACLE_SAM3, f"lunann_train_{os.path.splitext(f)[0]}.txt"))]
        paths = [os.path.join(LOCO_SRC, "images", "train", f) for f in rest]
        prod = pseudo_labels.production_boxes(paths, from_prod)
        for f, p in zip(rest, paths):
            stem = os.path.splitext(f)[0]
            sam3 = [b for b in copy_paste_augment._read_yolo_labels(
                os.path.join(ORACLE_SAM3, f"lunann_train_{stem}.txt")) if b[0] in from_sam3]
            _write_pseudo_labeled(batch_dir, p, stem, sam3 + prod[p])
        used.update(rest)
        counts["loco_train_pseudo"] = len(rest)

    assert_no_test_leak(batch_dir)
    yaml_path = os.path.join(batch_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {batch_dir}\ntrain: images/train\nval: images/val\n")
        f.write(f"names: {json.dumps({c['class_idx']: c['name'] for c in registry})}\n")
    with open(os.path.join(batch_dir, "manifest.json"), "w") as f:
        json.dump({"variant": name, "parts": sorted(parts), "counts": counts}, f, indent=2)
    print(f"batch {name}: {counts}", flush=True)
    return yaml_path, counts


def _load():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"batches": {}, "runs": [], "two_model": []}


def _save(report):
    with open(RESULTS, "w") as f:
        json.dump(report, f, indent=2)


def run_variant(name, report):
    registry = db.load_class_registry()
    loco_names = [c["name"] for c in db.loco_classes(registry)]
    parts = VARIANTS[name]
    if all(any(r["variant"] == name and r["seed"] == s for r in report["runs"]) for s in SEEDS):
        return
    yaml_path, counts = build(name, parts)
    report["batches"][name] = counts
    for seed in SEEDS:
        if any(r["variant"] == name and r["seed"] == seed for r in report["runs"]):
            continue
        best = finetune(yaml_path, f"baseline_{name}_s{seed}", batch=BATCH_SIZE, seed=seed, patience=0,
                        teacher_weights=PRODUCTION_WEIGHTS if "distill" in parts else None)
        weights = os.path.join(os.path.dirname(best), "last.pt")
        report["runs"].append({"variant": name, "seed": seed, "weights": weights,
                               "loco": evaluate_model(weights, registry, loco_names, db.DATASET_ROOT),
                               "iw_hub": eval_iw_hub(weights)})
        _save(report)
        print(f"{name} SEED {seed} RESULT: {json.dumps(report['runs'][-1]['loco'])}", flush=True)


def run_two_model(report):
    import train_specialist
    from two_model_gate import run_gate

    for seed in SEEDS:
        if any(r["seed"] == seed for r in report["two_model"]):
            continue
        batch = f"baseline_specialist_s{seed}"
        shutil.rmtree(os.path.join(train_specialist.BATCH_ROOT, batch), ignore_errors=True)
        data_yaml = train_specialist.build_batch(batch, seed=seed)
        from ultralytics import YOLO

        YOLO(train_specialist.SPECIALIST_BASE).train(
            data=data_yaml, epochs=100, patience=20, batch=BATCH_SIZE, imgsz=960, seed=seed,
            project=train_specialist.YOLO_RUNS_DIR, name=batch, exist_ok=True)
        weights = os.path.join(train_specialist.YOLO_RUNS_DIR, batch, "weights", "best.pt")
        passed, gate = run_gate(weights, promote=False)
        report["two_model"].append({"seed": seed, "weights": weights, "passed": passed, "gate": gate})
        _save(report)
        print(f"two_model SEED {seed}: passed={passed}", flush=True)


def gate(name, report):
    weights = [r["weights"] for r in report["runs"] if r["variant"] == name]
    print(f"===== GATE (dry run): {name} =====", flush=True)
    subprocess.run([sys.executable, "promote_if_better.py", "--run-name", f"baseline_{name}",
                    "--weights", *weights, "--dry-run"], check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[*VARIANTS, "two_model"], default=None)
    args = parser.parse_args()
    report = _load()
    names = [args.only] if args.only else [*VARIANTS, "two_model"]
    for name in names:
        if name == "two_model":
            run_two_model(report)
        else:
            run_variant(name, report)
    for name in names:
        if name != "two_model":
            gate(name, report)
    print("BASELINES DONE", flush=True)


if __name__ == "__main__":
    main()
