"""
Evaluate a YOLOv8 model fine-tuned on LOCO, on the same GT/CSV pipeline as
eval_loco.py (SAM3), eval_loco_owlv2.py and eval_loco_grounding_dino.py, so
results are directly comparable to the zero-shot rows.

Unlike the zero-shot models, this one was *trained* on LOCO (see
convert_loco_to_yolo.py + `yolo detect train ... data=yolo/loco.yaml`), using
the official per-subset train/val split. Evaluation here therefore defaults
to the held-out val split (loco_val_paths.txt) -- images the model never
trained on -- not an arbitrary hand-picked sample, so the numbers reflect
generalization rather than memorization.

Usage:
    python eval_loco_yolo.py --weights yolo_runs/loco_full_v1/weights/best.pt \
        --message "yolov8s fine-tuned on full LOCO train split (2820 imgs, 100 epochs)"
"""

import argparse
import os
import time

from eval_loco import build_gt_subset, evaluate

# fixed class order used by convert_loco_to_yolo.py -- YOLO class index -> LOCO category name
CLASS_ORDER = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]


def run_predictions(gt, dataset_root, weights, conf=0.001, imgsz=640, batch=32):
    from ultralytics import YOLO

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    class_idx_to_cat_id = {i: name_to_cat_id[name] for i, name in enumerate(CLASS_ORDER)}

    model = YOLO(weights)

    image_paths = [os.path.join(dataset_root, img["path"].lstrip("/")) for img in gt["images"]]
    id_by_path = {
        os.path.join(dataset_root, img["path"].lstrip("/")): img["id"] for img in gt["images"]
    }

    results = []
    start_time = time.perf_counter()
    n_done = 0
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i : i + batch]
        preds = model.predict(chunk, conf=conf, imgsz=imgsz, device=0, verbose=False)
        n_batch_dets = 0
        for path, pred in zip(chunk, preds):
            image_id = id_by_path[path]
            boxes = pred.boxes
            for xyxy, score, cls in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
            ):
                x0, y0, x1, y1 = xyxy
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": class_idx_to_cat_id[int(cls)],
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(score),
                    }
                )
                n_batch_dets += 1
            n_done += 1
        print(f"[{n_done}/{len(image_paths)}] batch done: {n_batch_dets} detections")

    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default="loco_val_paths.txt",
                         help="Default: the official LOCO val split (held out from training), not a hand-picked sample.")
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--weights", required=True, help="Path to trained YOLOv8 weights (best.pt).")
    parser.add_argument("--conf", type=float, default=0.001, help="Low threshold -- COCOeval needs the full score range.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--results-csv", default="results/loco_eval_results.csv")
    parser.add_argument("--run", default=None, help="Run name, e.g. 'yolov8_v1'. Auto-incremented as 'yolov8_vN' if omitted.")
    parser.add_argument(
        "--message", required=True,
        help="Required. What does this run represent? "
             "e.g. \"yolov8s fine-tuned on full LOCO train split (2820 imgs / 86936 boxes), 100 epochs, evaluated on held-out val split\"",
    )
    args = parser.parse_args()

    print(f"=== {args.message} ===\n")

    with open(args.sample_list) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    gt = build_gt_subset(args.annotations, image_paths)
    print(f"Evaluating on {len(gt['images'])} images, {len(gt['annotations'])} GT boxes")

    results, elapsed_sec = run_predictions(gt, args.dataset_root, args.weights, args.conf, args.imgsz, args.batch)
    print(f"Inference: {elapsed_sec:.1f}s total, {elapsed_sec / len(gt['images']):.4f}s/image")

    # reuse eval_loco's evaluate()/CSV writer, but stamp the correct model name
    # and blank out the zero-shot text-prompt column -- this model is fine-tuned,
    # it never saw a text prompt for any class.
    import eval_loco
    eval_loco.MODEL_NAME = f"ultralytics YOLOv8 ({os.path.basename(args.weights)}, fine-tuned on LOCO train split, conf={args.conf})"
    eval_loco.CATEGORY_PROMPTS = {name: "(fine-tuned class, no prompt)" for name in CLASS_ORDER}

    evaluate(
        gt, results, args.message, elapsed_sec,
        run_name=args.run, run_prefix="yolov8", csv_path=args.results_csv,
    )


if __name__ == "__main__":
    main()
