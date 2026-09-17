"""
Runs one fine-tune round with the exact recipe already validated in
finetune_robot_frozen_mixed.py (freeze=21: backbone+neck frozen, only the
last C2f block + Detect head train; 30 epochs; imgsz 960) -- always
starting from the pristine LOCO-only baseline checkpoint, never a previous
pipeline round's weights, so drift never compounds across rounds (see
build_training_batch.py's docstring for the full rationale).

Exposed as a function (not just a __main__ block, unlike
finetune_robot_frozen_mixed.py) so auto_train_trigger.py can call it
directly.

Usage:
    python run_finetune.py --data path/to/data.yaml --run-name loco_finetune_pipeline_v3
"""

import argparse
import os

import db

BASELINE = os.path.join(db.PROJECT_ROOT, "runs", "detect", "yolo_runs", "loco_full_v3", "weights", "best.pt")
YOLO_RUNS_DIR = os.path.join(db.PROJECT_ROOT, "runs", "detect", "yolo_runs")
FREEZE_LAYERS = 21


def finetune(data_yaml, run_name, baseline=BASELINE, epochs=30, imgsz=960):
    from ultralytics import YOLO

    if not os.path.exists(baseline):
        raise FileNotFoundError(
            f"LOCO-only baseline checkpoint not found at {baseline} -- every pipeline fine-tune "
            f"round starts from this pristine checkpoint (never a previous round's weights); "
            f"it needs to be provisioned once (it's gitignored, same as every other run/ artifact)."
        )

    model = YOLO(baseline)
    model.train(
        data=data_yaml,
        epochs=epochs,
        patience=10,
        batch=-1,
        imgsz=imgsz,
        cls=1.0,
        freeze=FREEZE_LAYERS,
        project=YOLO_RUNS_DIR,
        name=run_name,
        exist_ok=True,
    )
    return os.path.join(YOLO_RUNS_DIR, run_name, "weights", "best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    weights = finetune(args.data, args.run_name, args.baseline, args.epochs, args.imgsz)
    print(f"finetune complete: {weights}")


if __name__ == "__main__":
    main()
