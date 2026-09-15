"""
Control experiment for finetune_robot_frozen.py: same baseline checkpoint,
same freeze (backbone+neck frozen, only last C2f + Detect head trainable),
same 102/26 video-frame split, same hyperparameters -- but WITHOUT the added
iw_hub class (plain 5-class dataset/sam3_v6_pseudo_split/loco.yaml).

Purpose: isolate whether adding the new class caused the LOCO-val forgetting
seen in finetune_robot_frozen.py, or whether it was already caused by
fine-tuning the head on this small/skewed 102-image video-frame set alone
(as already suspected from yolov8_v4 in dataset/results/loco_eval_results.csv,
which forgot catastrophically on a full fine-tune of the same frames, no new
class involved).

Usage:
    python finetune_frozen_control.py
"""

import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

BASELINE = os.path.join(PROJECT_ROOT, "runs/detect/yolo_runs/loco_full_v3/weights/best.pt")
DATA_YAML = os.path.join(ROOT, "sam3_v6_pseudo_split", "loco.yaml")
FREEZE_LAYERS = 21

if __name__ == "__main__":
    model = YOLO(BASELINE)
    model.train(
        data=DATA_YAML,
        epochs=30,
        patience=10,
        batch=-1,
        imgsz=960,
        cls=1.0,
        freeze=FREEZE_LAYERS,
        project=os.path.join(PROJECT_ROOT, "runs/detect/yolo_runs"),
        name="loco_finetune_pseudo_frozen_control_v1",
        exist_ok=True,
    )
