"""
Evaluate SAM3 (image predictor, text-prompted) on a subset of the LOCO dataset.

For each image, runs one text prompt per LOCO category and collects the predicted
boxes/scores as COCO-format detections, then scores them against the LOCO ground
truth (bbox IoU) with pycocotools.

Usage:
    python eval_loco.py --sample-list /tmp/loco_sample_paths.txt --annotations annotations/loco-all-v1.json
"""

import argparse
import copy
import csv
import datetime
import json
import os
import time

from PIL import Image

CATEGORY_PROMPTS = {
    "small_load_carrier": "plastic tote box",
    "forklift": "forklift",
    "pallet": "pallet",
    "stillage": "wire mesh cage",
    "pallet_truck": "pallet truck",
}

MODEL_NAME = "facebook/sam3 (sam3.pt, build_sam3_image_model)"

CSV_FIELDS = [
    "run", "date", "model", "sample_images", "category", "prompt",
    "gt_boxes", "predictions", "AP_50_95", "AP_50", "AP_75", "AP_95", "AR_100",
    "sec_per_image", "images_per_sec", "message",
]


def _next_run_name(csv_path, prefix):
    """Run names are always '<model family>_v<n>' (e.g. 'sam3_v2', 'gdino_v1') so
    each model's runs auto-number independently of the other's."""
    if not os.path.exists(csv_path):
        return f"{prefix}_v1"
    with open(csv_path, newline="") as f:
        runs = {row["run"] for row in csv.DictReader(f)}
    n = 1
    while f"{prefix}_v{n}" in runs:
        n += 1
    return f"{prefix}_v{n}"


def build_gt_subset(annotations_path, image_paths):
    with open(annotations_path) as f:
        coco = json.load(f)

    wanted = {p.lstrip("/") for p in image_paths}
    images = [img for img in coco["images"] if img["path"].lstrip("/") in wanted]
    image_ids = {img["id"] for img in images}
    annotations = [a for a in coco["annotations"] if a["image_id"] in image_ids]

    gt = {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }
    return gt


def run_predictions(gt, dataset_root, confidence_threshold=0.5):
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # the model mixes bf16 and fp32 weights internally; without autocast, matmuls
    # between them raise "mat1 and mat2 must have the same dtype" (see the
    # sam3_image_predictor_example notebook, which sets up the same context)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    results = []
    start_time = time.perf_counter()
    for i, img_meta in enumerate(gt["images"]):
        image_path = os.path.join(dataset_root, img_meta["path"].lstrip("/"))
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)

        for cat_id, cat_name in cat_id_to_name.items():
            prompt = CATEGORY_PROMPTS.get(cat_name, cat_name.replace("_", " "))
            state = processor.set_text_prompt(prompt=prompt, state=state)

            boxes = state["boxes"].cpu().tolist()
            scores = state["scores"].cpu().tolist()
            for box, score in zip(boxes, scores):
                x0, y0, x1, y1 = box
                results.append(
                    {
                        "image_id": img_meta["id"],
                        "category_id": cat_id,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": score,
                    }
                )

        print(f"[{i + 1}/{len(gt['images'])}] {img_meta['path']}: "
              f"{sum(1 for r in results if r['image_id'] == img_meta['id'])} detections")

    torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def _run_coco_eval(coco_gt, coco_dt, cat_ids=None):
    """Runs COCOeval (optionally restricted to cat_ids) and returns the
    (AP_50_95, AP_50, AP_75, AR_100) tuple from its summary stats."""
    from pycocotools.cocoeval import COCOeval

    ce = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if cat_ids is not None:
        ce.params.catIds = cat_ids
    ce.evaluate()
    ce.accumulate()
    ce.summarize()
    # ce.stats order: [AP, AP50, AP75, AP_s, AP_m, AP_l, AR@1, AR@10, AR@100, AR_s, AR_m, AR_l]
    return ce.stats[0], ce.stats[1], ce.stats[2], ce.stats[8]


def _ap_at_iou(coco_gt, coco_dt, iou, cat_ids=None):
    """AP at a single, strict IoU threshold (e.g. 0.95) -- separate from the
    AP_50_95 average, since a single-image-comparison of the two extremes
    (AP_50 vs AP_95) shows how loose/tight the predicted boxes are."""
    import numpy as np
    from pycocotools.cocoeval import COCOeval

    ce = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if cat_ids is not None:
        ce.params.catIds = cat_ids
    ce.params.iouThrs = np.array([iou])
    ce.evaluate()
    ce.accumulate()
    ce.summarize()
    return ce.stats[0] if ce.stats is not None else float("nan")


def evaluate(gt, results, message, elapsed_sec, run_name=None, run_prefix="sam3", csv_path="results/loco_eval_results.csv"):
    if not message or not message.strip():
        raise ValueError(
            "message is required: describe what this run's prompts/config represent "
            "(e.g. 'baseline literal category-name prompts') so results.csv stays readable later."
        )

    from pycocotools.coco import COCO

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    gt_counts = {cid: 0 for cid in cat_id_to_name}
    for a in gt["annotations"]:
        gt_counts[a["category_id"]] += 1
    det_counts = {cid: 0 for cid in cat_id_to_name}
    for r in results:
        det_counts[r["category_id"]] += 1

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    if len(results) == 0:
        print("No detections produced -- skipping COCOeval.")
        return

    coco_dt = coco_gt.loadRes(copy.deepcopy(results))

    print("=== OVERALL ===")
    overall = _run_coco_eval(coco_gt, coco_dt)
    overall_ap95 = _ap_at_iou(coco_gt, coco_dt, 0.95)

    per_category = {}
    per_category_ap95 = {}
    for cat_id, cat_name in cat_id_to_name.items():
        print(f"-- {cat_name} --")
        per_category[cat_id] = _run_coco_eval(coco_gt, coco_dt, cat_ids=[cat_id])
        per_category_ap95[cat_id] = _ap_at_iou(coco_gt, coco_dt, 0.95, cat_ids=[cat_id])

    run_name = run_name or _next_run_name(csv_path, run_prefix)
    date = datetime.date.today().isoformat()
    n_images = len(gt["images"])
    sec_per_image = elapsed_sec / n_images
    images_per_sec = n_images / elapsed_sec
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # guard against a missing trailing newline in the existing file (e.g. from a
    # hand edit, or a previous run whose write got cut short) -- otherwise the
    # first new row gets glued onto the end of the last existing line.
    if not write_header and os.path.getsize(csv_path) > 0:
        with open(csv_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            ends_with_newline = f.read(1) == b"\n"
        if not ends_with_newline:
            with open(csv_path, "a", newline="") as f:
                f.write("\n")

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for cat_id, cat_name in cat_id_to_name.items():
            ap, ap50, ap75, ar100 = per_category[cat_id]
            writer.writerow({
                "run": run_name, "date": date, "message": message, "model": MODEL_NAME,
                "sample_images": n_images, "category": cat_name,
                "prompt": CATEGORY_PROMPTS.get(cat_name, cat_name.replace("_", " ")),
                "gt_boxes": gt_counts[cat_id], "predictions": det_counts[cat_id],
                "AP_50_95": f"{ap:.3f}", "AP_50": f"{ap50:.3f}",
                "AP_75": f"{ap75:.3f}", "AP_95": f"{per_category_ap95[cat_id]:.3f}",
                "AR_100": f"{ar100:.3f}",
                "sec_per_image": f"{sec_per_image:.4f}", "images_per_sec": f"{images_per_sec:.2f}",
            })
        ap, ap50, ap75, ar100 = overall
        writer.writerow({
            "run": run_name, "date": date, "message": message, "model": MODEL_NAME,
            "sample_images": n_images, "category": "OVERALL",
            "prompt": "(all prompts above)",
            "gt_boxes": len(gt["annotations"]), "predictions": len(results),
            "AP_50_95": f"{ap:.3f}", "AP_50": f"{ap50:.3f}",
            "AP_75": f"{ap75:.3f}", "AP_95": f"{overall_ap95:.3f}",
            "AR_100": f"{ar100:.3f}",
            "sec_per_image": f"{sec_per_image:.4f}", "images_per_sec": f"{images_per_sec:.2f}",
        })

    print(f"\nAppended run '{run_name}' to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default="/tmp/loco_sample_paths.txt")
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--out", default="loco_eval_results.json")
    parser.add_argument("--results-csv", default="results/loco_eval_results.csv")
    parser.add_argument("--run", default=None, help="Run name, e.g. 'sam3_v3'. Auto-incremented as 'sam3_vN' if omitted.")
    parser.add_argument(
        "--message", required=True,
        help="Required. What does this run's prompts/config represent? "
             "e.g. \"baseline: literal category-name prompts\" or "
             "\"reworded stillage->'wire mesh cage' to fix zero recall from v1\"",
    )
    args = parser.parse_args()

    print(f"=== {args.message} ===\n")

    with open(args.sample_list) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    gt = build_gt_subset(args.annotations, image_paths)
    print(f"Evaluating on {len(gt['images'])} images, {len(gt['annotations'])} GT boxes")

    results, elapsed_sec = run_predictions(gt, args.dataset_root, args.confidence_threshold)
    print(f"Inference: {elapsed_sec:.1f}s total, {elapsed_sec / len(gt['images']):.3f}s/image")

    with open(args.out, "w") as f:
        json.dump({"gt": gt, "detections": results}, f)
    print(f"Saved raw predictions + GT subset to {args.out}")

    evaluate(gt, results, args.message, elapsed_sec, run_name=args.run, csv_path=args.results_csv)


if __name__ == "__main__":
    main()
