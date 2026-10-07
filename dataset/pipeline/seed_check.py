"""
Is self-training round 3 really better than round 0, or is the gap training
noise? Retrains both batches with a FIXED batch size (the original runs used
AutoBatch, which picked 4 for round 0 and 12-16 for rounds 1-3 on a shared
GPU) and 3 seeds each, then evaluates exactly like self_training.py.

Usage:
    python seed_check.py
"""

import json
import os

import db
from promote_if_better import evaluate_model
from run_finetune import finetune
from self_training import OUT_ROOT, eval_iw_hub

BATCHES = {
    "round0": os.path.join(db.PIPELINE_ROOT, "training_batches", "pipeline_r0_pseudo", "data.yaml"),
    "round3": os.path.join(OUT_ROOT, "batches", "self_training_r3", "data.yaml"),
}
SEEDS = [0, 1, 2]
BATCH_SIZE = 8
RESULTS = os.path.join(OUT_ROOT, "seed_check.json")


def main():
    registry = db.load_class_registry()
    loco_names = [c["name"] for c in db.loco_classes(registry)]
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
    done = {(r["batch"], r["seed"]) for r in results}
    for seed in SEEDS:
        for name, data_yaml in BATCHES.items():
            if (name, seed) in done:
                continue
            run = f"seedcheck_{name}_s{seed}"
            weights = finetune(data_yaml, run, batch=BATCH_SIZE, seed=seed)
            results.append({"batch": name, "seed": seed, "weights": weights,
                            "loco": evaluate_model(weights, registry, loco_names, db.DATASET_ROOT),
                            "iw_hub": eval_iw_hub(weights)})
            with open(RESULTS, "w") as f:
                json.dump(results, f, indent=2)
            print(f"SEEDCHECK {name} seed={seed}: {json.dumps(results[-1]['loco'])} {json.dumps(results[-1]['iw_hub'])}",
                  flush=True)
    print("SEED CHECK DONE", flush=True)


if __name__ == "__main__":
    main()
