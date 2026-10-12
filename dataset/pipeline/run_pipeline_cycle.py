"""
One full workstation-side pipeline cycle, meant for an hourly cron:

  1. ingest_from_robot.py  -- register frames sync_node.py rsynced into
     dataset/incoming/<robot_id>/;
  2. sam3_quality_gate.py   -- SAM3-check every pending frame (auto-approve
     the clear ones, stage the ambiguous ones for review on the robot,
     reject the rest); a no-op without pending frames;
  3. auto_train_trigger     -- one training round + promotion gate once
     enough new approved frames have piled up; a no-op below the threshold.

Review decisions arrive separately: review_sync_node.py on the robot
uploads them and runs apply_synced_decisions.py itself. The robot pulls a
newly promoted model on its own (model_sync_node.py).

A lock (.cycle_lock) keeps overlapping cron runs from stacking up while a
training round is still going. Steps 1-2 run as subprocesses so a crash in
one shows up in the log and the cycle still moves on.

Cron (crontab -e on the workstation):
    0 * * * * cd $HOME/Desktop/Zero-Shot-models-Logistics/dataset/pipeline && \\
        PYTORCH_ALLOC_CONF=expandable_segments:True \\
        ../../.venv/bin/python run_pipeline_cycle.py >> logs/pipeline_cycle.log 2>&1

Usage:
    python run_pipeline_cycle.py [--threshold 50] [--dry-run]
"""

import argparse
import os
import subprocess
import sys
import time

import auto_train_trigger
import db

CYCLE_LOCK = os.path.join(db.PIPELINE_ROOT, ".cycle_lock")


def _step(script):
    print(f"\n--- {script} ---", flush=True)
    result = subprocess.run([sys.executable, os.path.join(db.PIPELINE_ROOT, script)], cwd=db.PIPELINE_ROOT)
    if result.returncode != 0:
        print(f"{script} failed (exit {result.returncode}) -- continuing", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=auto_train_trigger.APPROVAL_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true", help="run the promotion gate without touching production")
    args = parser.parse_args()

    print(f"\n===== pipeline cycle {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
    if not auto_train_trigger.acquire_lock(CYCLE_LOCK):
        print("previous cycle still running -- skipping", flush=True)
        return
    try:
        _step("ingest_from_robot.py")
        _step("sam3_quality_gate.py")
        print("\n--- auto_train_trigger ---", flush=True)
        auto_train_trigger.run_once(args.threshold, args.dry_run)
    finally:
        auto_train_trigger.release_lock(CYCLE_LOCK)


if __name__ == "__main__":
    main()
