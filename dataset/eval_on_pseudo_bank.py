"""
Evaluates a YOLOv8 model against the SAM3 pseudo-labeled bank
(dataset/sam3_v6_pseudo) itself, through the same COCOeval/CSV pipeline as
eval_loco_yolo.py, so the row lands in the same schema (AP50, AP75, AP95,
AR100) as every other run.

IMPORTANT CAVEAT, always stamped into the CSV message: this is a
self-referential check, not a benchmark. The "ground truth" here is SAM3's
own pseudo-labels, not human annotation -- a model trained on this same bank
will look artificially good here. Use only to sanity-check convergence, never
to compare against the LOCO-val rows (yolov8_v1/v2/v3/v4, sam3_*, owlv2_*, ...).

Usage:
    python eval_on_pseudo_bank.py --weights <best.pt> --message "..."
"""

import argparse
import glob
import os
import time

from PIL import Image

from eval_loco import evaluate

PSEUDO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_v6_pseudo")

# same class order as convert_loco_to_yolo.py / generate_sam3_pseudo_dataset.py,
# mapped to the real LOCO category ids so the CSV's category names line up
# with every other run.
CLASS_ORDER = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]
CATEGORIES = [
    {"id": 3, "name": "small_load_carrier"},
    {"id": 5, "name": "forklift"},
    {"id": 7, "name": "pallet"},
    {"id": 10, "name": "stillage"},
    {"id": 11, "name": "pallet_truck"},
]
NAME_TO_CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}


def build_gt(pseudo_dir, subdir=""):
    image_paths = sorted(glob.glob(os.path.join(pseudo_dir, "images", subdir, "*.jpg")))
    images, annotations = [], []
    ann_id = 0

    for img_id, img_path in enumerate(image_paths):
        w, h = Image.open(img_path).size
        stem = os.path.splitext(os.path.basename(img_path))[0]
        images.append({"id": img_id, "path": img_path, "width": w, "height": h,
                        "file_name": os.path.basename(img_path)})

        label_path = os.path.join(pseudo_dir, "labels", subdir, stem + ".txt")
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_idx, cx, cy, bw, bh = line.split()
                cls_idx = int(cls_idx)
                cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": NAME_TO_CAT_ID[CLASS_ORDER[cls_idx]],
                    "bbox": [x, y, bw * w, bh * h], "area": (bw * w) * (bh * h),
                    "iscrowd": False,
                })
                ann_id += 1

    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def run_predictions(gt, weights, conf=0.001, imgsz=960, batch=32):
    from ultralytics import YOLO

    class_idx_to_cat_id = {i: NAME_TO_CAT_ID[name] for i, name in enumerate(CLASS_ORDER)}
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
    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-dir", default=PSEUDO_DIR)
    parser.add_argument("--subdir", default="", help="Subfolder under images/ and labels/ to restrict to, e.g. 'val' for a train/val split layout.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--results-csv", default="results/loco_eval_results_yolo.csv")
    parser.add_argument("--run", default=None)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    print(f"=== {args.message} ===\n")

    gt = build_gt(args.pseudo_dir, args.subdir)
    print(f"Self-check GT: {len(gt['images'])} images, {len(gt['annotations'])} boxes "
          f"(SAM3 pseudo-labels, NOT human ground truth)")

    results, elapsed_sec = run_predictions(gt, args.weights, args.conf, args.imgsz)
    print(f"Inference: {elapsed_sec:.1f}s total, {elapsed_sec / len(gt['images']):.4f}s/image")

    import eval_loco
    eval_loco.MODEL_NAME = f"ultralytics YOLOv8 ({os.path.basename(args.weights)}, self-check vs. its own SAM3 pseudo-label training bank, conf={args.conf})"
    eval_loco.CATEGORY_PROMPTS = {name: "(fine-tuned class, no prompt)" for name in CLASS_ORDER}

    message = f"[SELF-REFERENTIAL, NOT A BENCHMARK] {args.message}"
    evaluate(gt, results, message, elapsed_sec, run_name=args.run,
             run_prefix="yolov8_selfcheck", csv_path=args.results_csv)


if __name__ == "__main__":
    main()
