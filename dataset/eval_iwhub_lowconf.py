"""
Retries the yolov8_v12_robot_iwhub logging at conf=0.001 (the pipeline's
documented convention for COCOeval -- see eval_loco_yolo.py: "low threshold,
COCOeval needs the full score range"). The first attempt at conf=0.5 (this
CSV's other recent rows' "real-world threshold" convention) produced ZERO
iw_hub detections across all 509 images -- itself a meaningful finding
(logged in the message), but useless for computing AP, which needs the full
score range to build its precision-recall curve.

Usage:
    python eval_iwhub_lowconf.py
"""

import eval_loco
from eval_loco import evaluate
from eval_robot_finetune_experiments import ROBOT_WEIGHTS, build_iwhub_gt, predict_iwhub

if __name__ == "__main__":
    iwhub_gt = build_iwhub_gt()
    print(f"iw_hub: {len(iwhub_gt['images'])} images, {len(iwhub_gt['annotations'])} GT boxes")

    eval_loco.CATEGORY_PROMPTS = {"iw_hub": "(fine-tuned class, no prompt)"}
    eval_loco.MODEL_NAME = ("ultralytics YOLOv8m (loco_finetune_pseudo_robot_frozen_v1/best.pt, "
                             "backbone+neck frozen, fine-tuned on 11 SAM3-pseudo-labeled video frames "
                             "for the iw_hub class, conf=0.001)")
    results, elapsed = predict_iwhub(iwhub_gt, ROBOT_WEIGHTS, conf=0.001)
    evaluate(
        iwhub_gt, results,
        message=(
            "Does the new iw_hub class generalize beyond the 11 SAM3-pseudo-labeled video frames it "
            "was trained on? Evaluated on the full iw.hub-detection-pdyql Roboflow export (509 images, "
            "CC BY 4.0, github.com/eduardo-milanez -- train+valid+test all combined, none of it used "
            "anywhere in training/fine-tuning, all 509 held-out). conf=0.001 here (not this CSV's usual "
            "0.5 'real-world' threshold) because at conf=0.5 this run produced ZERO iw_hub detections "
            "across all 509 images -- the new class, learned from only 11 positive examples, is not "
            "confident enough to clear that bar on unseen images; that zero-recall-at-0.5 result is "
            "itself the headline finding, this row's AP/AR (full score range) is the supporting detail. "
            "Only the model's class-5 (iw_hub) output head is evaluated; this dataset has no LOCO-class "
            "annotations."
        ),
        elapsed_sec=elapsed, run_name="yolov8_v12_robot_iwhub", csv_path="results/loco_eval_results.csv",
    )
