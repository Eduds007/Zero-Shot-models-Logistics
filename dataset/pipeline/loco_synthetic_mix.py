"""
Does adding synthetic images to training help on real images only? The
human arm of loco_sam3_vs_human.py (LOCO train, human labels, YOLOv8m from
COCO weights) retrained with Isaac Sim renders added to its train split --
same real val split, same real test set (the official LOCO val split, 2277
images), same recipe and seeds, so the human runs there are the baseline.

Synthetic images: test_dataset/ of github.com/vidaldani/3D_object_detection
(Isaac Sim Replicator, 640x640, tight 2D boxes with a per-frame class map).
Classes mapped to LOCO: klt -> small_load_carrier, forklift, pallet, stillage,
pallet_truck; "transporter" (not a LOCO class) and boxes whose semantic id
isn't in the frame's class map are dropped.

Usage:
    python loco_synthetic_mix.py --prepare   # download + convert only
    python loco_synthetic_mix.py
"""

import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

import db
from loco_sam3_vs_human import RESULTS as BASELINE_RESULTS
from loco_sam3_vs_human import ROOT as SPLIT_ROOT
from loco_sam3_vs_human import SEEDS, TRAIN_ARGS
from promote_if_better import compare_with_production
from run_finetune import YOLO_RUNS_DIR
from self_training import OUT_ROOT

REPO = "https://raw.githubusercontent.com/vidaldani/3D_object_detection/main/test_dataset"
TREE = "https://api.github.com/repos/vidaldani/3D_object_detection/git/trees/main?recursive=1"
SYNTH = os.path.join(db.DATASET_ROOT, "synthetic_isaac")
ROOT = os.path.join(db.DATASET_ROOT, "loco_synthetic_mix")
RESULTS = os.path.join(OUT_ROOT, "results_loco_synthetic_mix.json")
CLASS_MAP = {"klt": "small_load_carrier", "forklift": "forklift", "pallet": "pallet", "stillage": "stillage",
             "pallet_truck": "pallet_truck"}
WORKERS = 4  # half the default: a run was killed without a traceback, likely by the host's OOM killer


def prepare():
    """Downloads the renders and writes SYNTH/{images,labels} (YOLO, RGB jpg)."""
    from PIL import Image

    idx = {c["name"]: c["class_idx"] for c in db.loco_classes()}
    with urllib.request.urlopen(TREE) as r:
        tree = json.load(r)["tree"]
    frames = sorted(os.path.basename(x["path"]).split("_")[0] for x in tree
                    if x["path"].startswith("test_dataset/rgb/") and x["path"].endswith("_rgb.png"))
    raw = os.path.join(SYNTH, "raw")
    for d in ("raw", "images", "labels"):
        os.makedirs(os.path.join(SYNTH, d), exist_ok=True)
    counts, dropped = {n: 0 for n in idx}, 0
    for k in frames:
        files = {"rgb": f"rgb/{k}_rgb.png", "boxes": f"bb_tight/{k}_bounding_box_2d_tight.npy",
                 "labels": f"bb_tight/{k}_bounding_box_2d_tight_labels.json"}
        local = {}
        for key, rel in files.items():
            local[key] = os.path.join(raw, os.path.basename(rel))
            if not os.path.exists(local[key]):
                urllib.request.urlretrieve(f"{REPO}/{rel}", local[key])
        img = Image.open(local["rgb"]).convert("RGB")
        w, h = img.size
        img.save(os.path.join(SYNTH, "images", f"synth_{k}.jpg"), quality=95)
        names = {int(i): v["class"] for i, v in json.load(open(local["labels"])).items()}
        lines = []
        boxes = np.load(local["boxes"])
        if boxes.dtype.names:  # most frames: a structured array; some: a plain (n, 6) array, same columns
            boxes = np.stack([boxes[c].astype(float) for c in ("semanticId", "x_min", "y_min", "x_max", "y_max")], 1)
        for sem, x0, y0, x1, y1 in boxes.reshape(-1, boxes.shape[-1] if boxes.size else 6)[:, :5].tolist():
            name = CLASS_MAP.get(names.get(int(sem)))  # ids missing from the map: unlabeled prims
            if name is None:
                dropped += 1
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            lines.append(f"{idx[name]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                         f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            counts[name] += 1
        with open(os.path.join(SYNTH, "labels", f"synth_{k}.txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"{len(frames)} synthetic images; boxes {counts}; dropped (not LOCO) {dropped}", flush=True)
    return len(frames), counts


def build():
    """The human arm's split (symlinks) + the synthetic images in train."""
    human = os.path.join(SPLIT_ROOT, "human")
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            dst = os.path.join(ROOT, kind, split)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(os.path.join(human, kind, split)):
                if not os.path.lexists(os.path.join(dst, f)):
                    os.symlink(os.path.realpath(os.path.join(human, kind, split, f)), os.path.join(dst, f))
        for f in os.listdir(os.path.join(SYNTH, kind)):
            if not os.path.lexists(os.path.join(ROOT, kind, "train", f)):
                os.symlink(os.path.join(SYNTH, kind, f), os.path.join(ROOT, kind, "train", f))
    names = {c["class_idx"]: c["name"] for c in db.loco_classes()}
    yaml_path = os.path.join(ROOT, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {ROOT}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n")
    return yaml_path


def main():
    registry = db.load_class_registry()
    loco = [c for c in registry if c["source"] == "loco"]
    loco_names = [c["name"] for c in loco]
    report = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"runs": []}
    report["synthetic"], report["synthetic_boxes"] = prepare()
    yaml_path = build()
    report["n_train"] = len(os.listdir(os.path.join(ROOT, "images", "train")))
    baseline = {r["seed"]: r["weights"] for r in json.load(open(BASELINE_RESULTS))["runs"] if r["arm"] == "human"}
    for seed in SEEDS:
        if any(r["seed"] == seed for r in report["runs"]):
            continue
        name = f"loco_human_synth_s{seed}"
        last = os.path.join(YOLO_RUNS_DIR, name, "weights", "last.pt")
        if os.path.exists(last):  # killed mid-run: carry on from its last epoch
            model, args = last, {"resume": True}
        else:
            model, args = "yolov8m.pt", {"data": yaml_path, "seed": seed, "project": YOLO_RUNS_DIR, "name": name,
                                         "exist_ok": True, "workers": WORKERS, **TRAIN_ARGS}
        subprocess.run([sys.executable, "-c",
                        "import sys, json; from ultralytics import YOLO; "
                        "YOLO(sys.argv[1]).train(**json.loads(sys.argv[2]))",
                        model, json.dumps(args)], check=True)
        weights = os.path.join(YOLO_RUNS_DIR, name, "weights", "best.pt")
        mixed, human, ci = compare_with_production(weights, baseline[seed], loco, loco_names, set(loco_names),
                                                   db.DATASET_ROOT)
        report["runs"].append({"seed": seed, "weights": weights, "human": human, "human_synth": mixed,
                               "ci_synth_minus_human": {n: list(v) for n, v in ci.items()}})
        with open(RESULTS, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"SEED {seed} RESULT: human={json.dumps(human)} human_synth={json.dumps(mixed)} ci={json.dumps(ci)}",
              flush=True)
    print("SYNTHETIC MIX DONE", flush=True)


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    else:
        main()
