"""
Second attempt at the robot-class fine-tune (see finetune_robot_frozen.py),
this time diluting the video-frame training sample 50/50 with real LOCO
images (dataset/sam3_v6_pseudo_mixed_with_robot, built by
build_mixed_robot_dataset.py: 102 video + 102 LOCO train, 26 video + 26 LOCO
val), instead of training the frozen head on video frames alone.

Rationale: yolov8_v10_robot_frozen vs. yolov8_v11_control_frozen (see
loco_eval_results.csv) showed the LOCO-val forgetting was NOT caused by the
new class -- it was caused by fine-tuning the shared Detect head on a small
(102 img), class-distribution-skewed video-frame sample, which pulls the
head's learned decision boundary away from LOCO's actual distribution. Same
freeze (backbone+neck frozen, only last C2f + Detect head trainable) as
before; only the training data composition changes.

Usage:
    python finetune_robot_frozen_mixed.py
"""

import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

BASELINE = os.path.join(PROJECT_ROOT, "runs/detect/yolo_runs/loco_full_v3/weights/best.pt")
DATA_YAML = os.path.join(ROOT, "sam3_v6_finetune_robot_mixed.yaml")
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
        name="loco_finetune_pseudo_robot_frozen_mixed_v1",
        exist_ok=True,
    )
