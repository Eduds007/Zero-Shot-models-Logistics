"""
Logs the LoRA-style ("last layers only") robot-class fine-tune experiment to
results/loco_eval_results.csv, on the same COCOeval pipeline as every other
row (eval_loco.evaluate()), so numbers are directly comparable:

  1. yolov8_v10_robot_frozen   -- the fine-tuned model (LOCO baseline +
     backbone/neck frozen + iw_hub class added from 11 SAM3-labeled video
     frames), evaluated on the official LOCO val split. Measures forgetting
     of the 5 original classes.
  2. yolov8_v11_control_frozen -- control: identical recipe (same freeze,
     same 102/26 video-frame split, same hyperparameters) but WITHOUT the
     added iw_hub class. Isolates whether the new class caused the
     forgetting in (1), or whether it was already there from fine-tuning
     on this small/skewed video-frame set alone.
  3. yolov8_v12_robot_iwhub    -- the fine-tuned model from (1), evaluated
     on the separate iw_hub Roboflow dataset (509 images, CC BY 4.0,
     eduardo-milanez/iw.hub-detection-pdyql -- fully held out, none of it
     was used anywhere in training). Measures whether the new class was
     actually learned.

Usage:
    python eval_robot_finetune_experiments.py
"""

import json
import os
import time
from pathlib import Path

import eval_loco
from eval_loco import build_gt_subset, evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

LOCO_ANNOTATIONS = os.path.join(ROOT, "annotations", "loco-all-v1.json")
LOCO_VAL_PATHS = os.path.join(ROOT, "loco_val_paths.txt")
LOCO_CLASS_ORDER = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]

IWHUB_ROOT = Path(
    "/tmp/claude-4314312/-home-go39sin-Desktop-Zero-Shot-models-Logistics/"
    "9acee791-294a-46b5-8ee0-2c9fdccd61f0/scratchpad/iw_hub"
)

ROBOT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "runs/detect/yolo_runs/loco_finetune_pseudo_robot_frozen_v1/weights/best.pt")
CONTROL_WEIGHTS = os.path.join(
    PROJECT_ROOT, "runs/detect/yolo_runs/loco_finetune_pseudo_frozen_control_v1/weights/best.pt")

CONF = 0.5  # matches the "real-world confidence threshold" convention of yolov8_v9/_baseline/selfcheck_v2
IMGSZ = 960


def predict_on_loco(gt, weights, dataset_root, conf=CONF, imgsz=IMGSZ, batch=32):
    """Same as eval_loco_yolo.run_predictions, but silently drops any class
    index outside the 5 LOCO classes (e.g. the iw_hub model's class 5) --
    LOCO GT has no such category, so those detections are just not evaluable
    here."""
    from ultralytics import YOLO

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    class_idx_to_cat_id = {i: name_to_cat_id[name] for i, name in enumerate(LOCO_CLASS_ORDER)}

    model = YOLO(weights)
    image_paths = [os.path.join(dataset_root, img["path"].lstrip("/")) for img in gt["images"]]
    id_by_path = {p: img["id"] for p, img in zip(image_paths, gt["images"])}

    results = []
    start_time = time.perf_counter()
    n_done = 0
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        preds = model.predict(chunk, conf=conf, imgsz=imgsz, device=0, verbose=False)
        for path, pred in zip(chunk, preds):
            image_id = id_by_path[path]
            boxes = pred.boxes
            for xyxy, score, cls in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
            ):
                cls = int(cls)
                if cls not in class_idx_to_cat_id:
                    continue  # e.g. iw_hub (class 5) -- not a LOCO category, skip
                x0, y0, x1, y1 = xyxy
                results.append({
                    "image_id": image_id, "category_id": class_idx_to_cat_id[cls],
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(score),
                })
            n_done += 1
        print(f"[{n_done}/{len(image_paths)}] batch done")
    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def build_iwhub_gt():
    """Merges the 3 Roboflow export splits (train/valid/test, 344/85/80 =
    509 images) into one COCO-format GT with a single 'iw_hub' category.
    None of these images were used anywhere in training/fine-tuning."""
    images, annotations = [], []
    next_img_id = 0
    next_ann_id = 0
    for split in ("train", "valid", "test"):
        coco = json.load(open(IWHUB_ROOT / split / "_annotations.coco.json"))
        old_to_new = {}
        for im in coco["images"]:
            new_id = next_img_id
            next_img_id += 1
            old_to_new[im["id"]] = new_id
            images.append({"id": new_id, "path": f"/{split}/{im['file_name']}",
                            "width": im["width"], "height": im["height"]})
        for a in coco["annotations"]:
            if a["category_id"] != 1:  # 0 = Roboflow's placeholder root category, skip
                continue
            annotations.append({
                "id": next_ann_id, "image_id": old_to_new[a["image_id"]],
                "category_id": 0, "bbox": a["bbox"], "area": a.get("area", 0), "iscrowd": 0,
            })
            next_ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "iw_hub"}]}


def predict_iwhub(gt, weights, conf=CONF, imgsz=IMGSZ, batch=32):
    from ultralytics import YOLO

    model = YOLO(weights)
    image_paths = [str(IWHUB_ROOT) + img["path"] for img in gt["images"]]
    id_by_path = {p: img["id"] for p, img in zip(image_paths, gt["images"])}

    results = []
    start_time = time.perf_counter()
    n_done = 0
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        preds = model.predict(chunk, conf=conf, imgsz=imgsz, device=0, verbose=False)
        for path, pred in zip(chunk, preds):
            image_id = id_by_path[path]
            boxes = pred.boxes
            for xyxy, score, cls in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
            ):
                if int(cls) != 5:  # only the iw_hub head output is evaluable against this GT
                    continue
                x0, y0, x1, y1 = xyxy
                results.append({
                    "image_id": image_id, "category_id": 0,
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(score),
                })
            n_done += 1
        print(f"[{n_done}/{len(image_paths)}] batch done")
    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def main():
    with open(LOCO_VAL_PATHS) as f:
        loco_val_image_paths = [line.strip() for line in f if line.strip()]
    loco_gt = build_gt_subset(LOCO_ANNOTATIONS, loco_val_image_paths)
    print(f"LOCO val: {len(loco_gt['images'])} images, {len(loco_gt['annotations'])} GT boxes")

    eval_loco.CATEGORY_PROMPTS = {name: "(fine-tuned class, no prompt)" for name in LOCO_CLASS_ORDER}

    # --- 1. robot-frozen model on LOCO val (forgetting check) ---
    print("\n=== yolov8_v10_robot_frozen: robot-class fine-tune (frozen backbone/neck) on LOCO val ===")
    eval_loco.MODEL_NAME = ("ultralytics YOLOv8m (loco_finetune_pseudo_robot_frozen_v1/best.pt, "
                             "backbone+neck frozen, fine-tuned on 102 SAM3 video-frame pseudo-labels "
                             "+ iw_hub class, conf=0.5)")
    results, elapsed = predict_on_loco(loco_gt, ROBOT_WEIGHTS, ROOT)
    evaluate(
        loco_gt, results,
        message=(
            "LoRA-style fine-tune (freeze=21: only the last C2f block + Detect head trainable, "
            "backbone+neck frozen) starting from the LOCO-only baseline (loco_full_v3/best.pt), "
            "trained 30 epochs on dataset/sam3_v6_pseudo_split_with_robot (102 train / 26 val SAM3 "
            "video-frame pseudo-labels), which extends the existing 5-class video pseudo-labels with "
            "a 6th 'iw_hub' class detected via the 'yellow robot platform' SAM3 prompt (conf>0.5, "
            "11/128 frames had a robot box, all landed in the train split). Evaluated here on the "
            "official held-out LOCO val split (2277 imgs, never trained on) to check for forgetting "
            "of the 5 original classes vs. the baseline (yolov8_v9_baseline: AP_50 OVERALL 0.257). "
            "See yolov8_v11_control_frozen for the isolating control (same recipe, no new class)."
        ),
        elapsed_sec=elapsed, run_name="yolov8_v10_robot_frozen", csv_path="results/loco_eval_results.csv",
    )

    # --- 2. control (no robot class) on LOCO val ---
    print("\n=== yolov8_v11_control_frozen: control fine-tune (frozen, NO robot class) on LOCO val ===")
    eval_loco.MODEL_NAME = ("ultralytics YOLOv8m (loco_finetune_pseudo_frozen_control_v1/best.pt, "
                             "backbone+neck frozen, fine-tuned on 102 SAM3 video-frame pseudo-labels, "
                             "5 classes only, conf=0.5)")
    results, elapsed = predict_on_loco(loco_gt, CONTROL_WEIGHTS, ROOT)
    evaluate(
        loco_gt, results,
        message=(
            "Control for yolov8_v10_robot_frozen: identical recipe (same freeze=21, same "
            "dataset/sam3_v6_pseudo_split 102/26 video-frame split, same hyperparameters, same "
            "loco_full_v3 starting checkpoint) but WITHOUT adding the iw_hub class -- plain 5-class "
            "fine-tune. Purpose: isolate whether adding a new class caused the LOCO-val forgetting "
            "seen in yolov8_v10_robot_frozen, or whether it was already caused by fine-tuning the "
            "shared Detect head on this small (102 img), class-imbalanced video-frame set alone (as "
            "previously seen with a full, unfrozen fine-tune in yolov8_v4). Result: both this control "
            "(AP_50 OVERALL 0.117) and yolov8_v10_robot_frozen (AP_50 OVERALL 0.104) degrade almost "
            "identically from the baseline (0.257) -- the new class is NOT the cause; the small/skewed "
            "video-frame set is."
        ),
        elapsed_sec=elapsed, run_name="yolov8_v11_control_frozen", csv_path="results/loco_eval_results.csv",
    )

    # --- 3. robot-frozen model on iw_hub Roboflow set ---
    print("\n=== yolov8_v12_robot_iwhub: robot-class fine-tune on iw_hub Roboflow dataset ===")
    iwhub_gt = build_iwhub_gt()
    print(f"iw_hub: {len(iwhub_gt['images'])} images, {len(iwhub_gt['annotations'])} GT boxes")
    eval_loco.CATEGORY_PROMPTS = {"iw_hub": "(fine-tuned class, no prompt)"}
    eval_loco.MODEL_NAME = ("ultralytics YOLOv8m (loco_finetune_pseudo_robot_frozen_v1/best.pt, "
                             "backbone+neck frozen, fine-tuned on 11 SAM3-pseudo-labeled video frames "
                             "for the iw_hub class, conf=0.5)")
    results, elapsed = predict_iwhub(iwhub_gt, ROBOT_WEIGHTS)
    evaluate(
        iwhub_gt, results,
        message=(
            "Does the new iw_hub class generalize beyond the 11 SAM3-pseudo-labeled video frames it "
            "was trained on? Evaluated on the full iw.hub-detection-pdyql Roboflow export (509 images, "
            "CC BY 4.0, github.com/eduardo-milanez -- train+valid+test all combined, since none of it "
            "was used anywhere in training/fine-tuning, all 509 are held-out). Only the model's class-5 "
            "(iw_hub) output head is evaluated here; this dataset has no LOCO-class annotations."
        ),
        elapsed_sec=elapsed, run_name="yolov8_v12_robot_iwhub", csv_path="results/loco_eval_results.csv",
    )


if __name__ == "__main__":
    main()
