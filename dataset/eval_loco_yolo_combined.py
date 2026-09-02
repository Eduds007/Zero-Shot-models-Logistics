"""
Evaluates a YOLOv8 model on a COMBINED validation set: the official LOCO val
split (loco_val_paths.txt, human-annotated) PLUS the 128-image SAM3
pseudo-label bank (dataset/sam3_v6_pseudo, 2fps video frames).

CAVEAT, stamped into the CSV message: unlike eval_loco_yolo.py's val-split
rows, this is a mixed benchmark -- part human ground truth (LOCO), part
SAM3's own pseudo-labels. If the model being evaluated was trained on any of
the 128 SAM3 images (e.g. loco_full_merged_v1, which trained on 102 of them),
this row is partially train-set-contaminated on top of that -- treat it as a
combined sanity-check, not a clean apples-to-apples benchmark against the
zero-shot / LOCO-only rows.

Usage:
    python eval_loco_yolo_combined.py --weights <best.pt> --conf 0.5 --imgsz 960 --message "..."
"""

import argparse
import os
import time

from eval_loco import build_gt_subset, evaluate
from eval_loco_yolo import CLASS_ORDER
from eval_on_pseudo_bank import PSEUDO_DIR, build_gt as build_pseudo_gt

ROOT = os.path.dirname(os.path.abspath(__file__))
ID_OFFSET = 1_000_000  # keeps SAM3-bank image/ann ids from colliding with LOCO's


def build_combined_gt(annotations_path, loco_val_paths, dataset_root, pseudo_dir, pseudo_subdir=""):
    loco_gt = build_gt_subset(annotations_path, loco_val_paths)
    for img in loco_gt["images"]:
        img["path"] = os.path.join(dataset_root, img["path"].lstrip("/"))  # make absolute

    pseudo_gt = build_pseudo_gt(pseudo_dir, pseudo_subdir)
    for img in pseudo_gt["images"]:
        img["id"] += ID_OFFSET
    for a in pseudo_gt["annotations"]:
        a["id"] += ID_OFFSET
        a["image_id"] += ID_OFFSET

    return {
        "images": loco_gt["images"] + pseudo_gt["images"],
        "annotations": loco_gt["annotations"] + pseudo_gt["annotations"],
        "categories": loco_gt["categories"],
    }, len(loco_gt["images"]), len(pseudo_gt["images"])


def run_predictions(gt, weights, conf, imgsz, batch=32):
    from ultralytics import YOLO

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    class_idx_to_cat_id = {i: name_to_cat_id[name] for i, name in enumerate(CLASS_ORDER)}

    model = YOLO(weights)
    image_paths = [img["path"] for img in gt["images"]]
    id_by_path = {img["path"]: img["id"] for img in gt["images"]}

    results = []
    start_time = time.perf_counter()
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i : i + batch]
        preds = model.predict(chunk, conf=conf, imgsz=imgsz, device=0, verbose=False)
        for path, pred in zip(chunk, preds):
            image_id = id_by_path[path]
            boxes = pred.boxes
            for xyxy, score, cls in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
            ):
                x0, y0, x1, y1 = xyxy
                results.append({
                    "image_id": image_id,
                    "category_id": class_idx_to_cat_id[int(cls)],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(score),
                })
        print(f"[{min(i + batch, len(image_paths))}/{len(image_paths)}] batch done")
    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json")
    parser.add_argument("--loco-val-list", default="loco_val_paths.txt")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--pseudo-dir", default=PSEUDO_DIR)
    parser.add_argument("--pseudo-subdir", default="", help="e.g. 'val' to use only the 26-image held-out split instead of all 128.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--results-csv", default="results/loco_eval_results.csv")
    parser.add_argument("--run", default=None)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    print(f"=== {args.message} ===\n")

    with open(args.loco_val_list) as f:
        loco_val_paths = [line.strip() for line in f if line.strip()]

    gt, n_loco, n_pseudo = build_combined_gt(
        args.annotations, loco_val_paths, args.dataset_root, args.pseudo_dir, args.pseudo_subdir
    )
    print(f"Combined GT: {n_loco} LOCO val images + {n_pseudo} SAM3 pseudo-bank images "
          f"= {len(gt['images'])} images, {len(gt['annotations'])} GT boxes")

    results, elapsed_sec = run_predictions(gt, args.weights, args.conf, args.imgsz, args.batch)
    print(f"Inference: {elapsed_sec:.1f}s total, {elapsed_sec / len(gt['images']):.4f}s/image")

    import eval_loco
    eval_loco.MODEL_NAME = f"ultralytics YOLOv8 ({os.path.basename(args.weights)}, fine-tuned, conf={args.conf})"
    eval_loco.CATEGORY_PROMPTS = {name: "(fine-tuned class, no prompt)" for name in CLASS_ORDER}

    message = f"[MIXED VAL SET: LOCO human GT + SAM3 pseudo-label GT] {args.message}"
    evaluate(gt, results, message, elapsed_sec, run_name=args.run,
             run_prefix="yolov8_combinedval", csv_path=args.results_csv)


if __name__ == "__main__":
    main()
