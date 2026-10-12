"""
Meant to run on a cron on the workstation (e.g. hourly): counts approved
frames not yet incorporated into any training round, and if there are
enough (APPROVAL_THRESHOLD), runs one full round end to end --
build_training_batch -> run_finetune (SEEDS, fixed epochs) ->
promote_if_better -- unattended. Below the threshold it's a fast no-op.

Each seed trains the full schedule with early stopping off and is gated on
its last.pt: the batch's val split holds images the pristine baseline was
trained on (see build_training_batch.py), so it can't pick a checkpoint.
promote_if_better promotes the best seed that passes the gate.

A lock file (.training_lock, holding this process's PID) stops two rounds
from overlapping if a round outlives the cron interval; a lock left behind
by a crashed process (PID no longer running) is treated as stale and
cleared automatically.

Usage (typically from cron):
    python auto_train_trigger.py [--threshold 50] [--dry-run]
"""

import argparse
import os
import time

import build_training_batch
import db
import promote_if_better
import run_finetune

APPROVAL_THRESHOLD = 50
SEEDS = (0, 1, 2)
BATCH_SIZE = 8  # fixed: AutoBatch picks 4-16 on the shared GPU, which changes results between rounds
LOCK_PATH = os.path.join(db.PIPELINE_ROOT, ".training_lock")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(path=LOCK_PATH):
    if os.path.exists(path):
        with open(path) as f:
            contents = f.read().strip()
        if contents.isdigit() and _pid_alive(int(contents)):
            return False
        print(f"stale lock found (pid {contents!r} not running) -- clearing it")
        os.remove(path)
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock(path=LOCK_PATH):
    if os.path.exists(path):
        os.remove(path)


def run_once(threshold=APPROVAL_THRESHOLD, dry_run=False):
    conn = db.connect()
    registry = db.load_class_registry()
    count = db.count_unincorporated_approved_total(conn, registry)
    print(f"{count} new approved frame(s) not yet incorporated into a training round (threshold: {threshold})")
    if count < threshold:
        return False

    if not acquire_lock():
        print("another round appears to be in progress (live lock held) -- skipping this cycle")
        return False

    try:
        run_name = f"pipeline_finetune_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"\n=== starting round {run_name} ===")

        data_yaml = build_training_batch.build_batch(run_name)
        weights = []
        for seed in SEEDS:
            print(f"\n--- fine-tuning, seed {seed} ---")
            best = run_finetune.finetune(data_yaml, f"{run_name}_s{seed}", batch=BATCH_SIZE, seed=seed, patience=0)
            weights.append(os.path.join(os.path.dirname(best), "last.pt"))
        print(f"\n--- promotion gate ---")
        promoted = promote_if_better.promote(run_name, weights, dry_run=dry_run)

        print(f"\n=== round {run_name} finished (promoted={promoted}) ===")
        return True
    finally:
        release_lock()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=APPROVAL_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true", help="run the gate but never touch production")
    args = parser.parse_args()
    run_once(args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
