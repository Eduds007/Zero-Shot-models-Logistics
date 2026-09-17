"""
Meant to run on a cron on the workstation (e.g. hourly): counts approved
frames not yet incorporated into any training round, and if there are
enough (APPROVAL_THRESHOLD), runs one full round end to end --
build_training_batch -> run_finetune -> promote_if_better -- unattended.
Below the threshold it's a fast no-op.

A lock file (.training_lock, holding this process's PID) stops two rounds
from overlapping if a round outlives the cron interval; a lock left behind
by a crashed process (PID no longer running) is treated as stale and
cleared automatically.

Usage (typically from cron):
    python auto_train_trigger.py [--threshold 50]
"""

import argparse
import os
import time

import build_training_batch
import db
import promote_if_better
import run_finetune

APPROVAL_THRESHOLD = 50
LOCK_PATH = os.path.join(db.PIPELINE_ROOT, ".training_lock")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock():
    if os.path.exists(LOCK_PATH):
        with open(LOCK_PATH) as f:
            contents = f.read().strip()
        if contents.isdigit() and _pid_alive(int(contents)):
            return False
        print(f"stale lock found (pid {contents!r} not running) -- clearing it")
        os.remove(LOCK_PATH)
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock():
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)


def run_once(threshold=APPROVAL_THRESHOLD):
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
        print(f"\n--- fine-tuning ---")
        weights = run_finetune.finetune(data_yaml, run_name)
        print(f"\n--- promotion gate ---")
        promoted = promote_if_better.promote(run_name, weights)

        print(f"\n=== round {run_name} finished (promoted={promoted}) ===")
        return True
    finally:
        release_lock()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=APPROVAL_THRESHOLD)
    args = parser.parse_args()
    run_once(args.threshold)


if __name__ == "__main__":
    main()
