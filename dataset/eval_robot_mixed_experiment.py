"""
Logs the mixed-data robot-class fine-tune (finetune_robot_frozen_mixed.py:
same freeze=21 recipe as yolov8_v10_robot_frozen, but trained on 102 video +
102 LOCO images, 50/50, instead of video frames alone) to
results/loco_eval_results.csv:

  1. yolov8_v13_robot_mixed_locoval -- forgetting check on official LOCO val
     (2277 imgs), compared against yolov8_v10_robot_frozen (video-only,
     AP_50 OVERALL 0.105) and the baseline (yolov8_v9_baseline, 0.257).
  2. yolov8_v14_robot_mixed_iwhub -- generalization check on the iw_hub
     Roboflow dataset (509 imgs, fully held out), compared against
     yolov8_v12_robot_iwhub (video-only, AP_50 0.179).

Usage:
    python eval_robot_mixed_experiment.py
"""

import eval_loco
from eval_loco import build_gt_subset, evaluate
from eval_robot_finetune_experiments import (
    LOCO_ANNOTATIONS, LOCO_VAL_PATHS, LOCO_CLASS_ORDER, ROOT,
    build_iwhub_gt, predict_on_loco, predict_iwhub,
)

MIXED_WEIGHTS = ROOT + "/../runs/detect/yolo_runs/loco_finetune_pseudo_robot_frozen_mixed_v1/weights/best.pt"


def main():
    with open(LOCO_VAL_PATHS) as f:
        loco_val_image_paths = [line.strip() for line in f if line.strip()]
    loco_gt = build_gt_subset(LOCO_ANNOTATIONS, loco_val_image_paths)

    eval_loco.CATEGORY_PROMPTS = {name: "(fine-tuned class, no prompt)" for name in LOCO_CLASS_ORDER}
    eval_loco.MODEL_NAME = ("ultralytics YOLOv8m (loco_finetune_pseudo_robot_frozen_mixed_v1/best.pt, "
                             "backbone+neck frozen, fine-tuned on 102 SAM3 video-frame pseudo-labels "
                             "(incl. iw_hub) + 102 real LOCO images 50/50, conf=0.5)")
    print("\n=== yolov8_v13_robot_mixed_locoval ===")
    results, elapsed = predict_on_loco(loco_gt, MIXED_WEIGHTS, ROOT)
    evaluate(
        loco_gt, results,
        message=(
            "Second attempt at the robot-class fine-tune, this time diluting the 102 video-frame "
            "train images 50/50 with 102 randomly-sampled real LOCO train images (and 26+26 for val) "
            "-- dataset/sam3_v6_pseudo_mixed_with_robot, built by build_mixed_robot_dataset.py -- "
            "instead of training the frozen head (freeze=21, same as yolov8_v10_robot_frozen) on video "
            "frames alone. Rationale: yolov8_v10_robot_frozen vs. yolov8_v11_control_frozen showed the "
            "video-frame set's class distribution (not the new class) was the cause of LOCO-val "
            "forgetting. Evaluated on the official held-out LOCO val split (2277 imgs). Compare AP_50 "
            "OVERALL to yolov8_v10_robot_frozen (0.105, video-only) and the baseline yolov8_v9_baseline "
            "(0.257)."
        ),
        elapsed_sec=elapsed, run_name="yolov8_v13_robot_mixed_locoval", csv_path="results/loco_eval_results.csv",
    )

    print("\n=== yolov8_v14_robot_mixed_iwhub ===")
    iwhub_gt = build_iwhub_gt()
    eval_loco.CATEGORY_PROMPTS = {"iw_hub": "(fine-tuned class, no prompt)"}
    eval_loco.MODEL_NAME = ("ultralytics YOLOv8m (loco_finetune_pseudo_robot_frozen_mixed_v1/best.pt, "
                             "backbone+neck frozen, fine-tuned on 102 video + 102 LOCO images 50/50 "
                             "for the iw_hub class, conf=0.001)")
    results, elapsed = predict_iwhub(iwhub_gt, MIXED_WEIGHTS, conf=0.001)
    evaluate(
        iwhub_gt, results,
        message=(
            "iw_hub generalization check for the mixed-data fine-tune (see yolov8_v13_robot_mixed_locoval "
            "for the LOCO-val side). Evaluated on the full iw.hub-detection-pdyql Roboflow export (509 "
            "images, CC BY 4.0, none used in training). conf=0.001 (full score range, for a valid AP "
            "curve) -- see yolov8_v12_robot_iwhub for the same conf=0.5 zero-detections caveat, checked "
            "here too. Compare AP_50 to yolov8_v12_robot_iwhub (0.179, video-only fine-tune)."
        ),
        elapsed_sec=elapsed, run_name="yolov8_v14_robot_mixed_iwhub", csv_path="results/loco_eval_results.csv",
    )


if __name__ == "__main__":
    main()
