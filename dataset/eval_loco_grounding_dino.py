"""
Evaluate Grounding DINO (HF transformers) on the same LOCO sample/GT/CSV
pipeline as eval_loco.py, so results are directly comparable to the SAM3 run.

Unlike SAM3's per-category prompting, Grounding DINO grounds all category
phrases in a single forward pass per image (one text query with all
candidate phrases, period-separated).

Usage:
    python eval_loco_grounding_dino.py --sample-list /tmp/loco_sample_paths.txt
"""

import argparse
import os
import time

from PIL import Image

from eval_loco import CATEGORY_PROMPTS, build_gt_subset, evaluate

MODEL_CLASSES = {
    "grounding_dino": "GroundingDinoForObjectDetection",
    "mm_grounding_dino": "MMGroundingDinoForObjectDetection",
}


def _match_label(returned_text, candidate_labels):
    """Grounding DINO sometimes merges adjacent phrases (e.g. returns
    'fork wire mesh cage' instead of 'wire mesh cage'). Recover the intended
    category by picking the longest candidate label that appears as a
    substring of the returned text."""
    returned_text = returned_text.lower()
    matches = [label for label in candidate_labels if label.lower() in returned_text]
    if not matches:
        return None
    return max(matches, key=len)


def run_predictions(gt, dataset_root, model_id, arch, box_threshold=0.3, text_threshold=0.25):
    import torch
    import transformers
    from transformers import AutoProcessor

    model_cls = getattr(transformers, MODEL_CLASSES[arch])

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    label_to_name = {CATEGORY_PROMPTS[name]: name for name in cat_id_to_name.values()}
    candidate_labels = list(label_to_name.keys())
    text_query = ". ".join(candidate_labels) + "."

    processor = AutoProcessor.from_pretrained(model_id)
    model = model_cls.from_pretrained(model_id).to("cuda").eval()

    results = []
    unmatched = 0
    start_time = time.perf_counter()
    for i, img_meta in enumerate(gt["images"]):
        image_path = os.path.join(dataset_root, img_meta["path"].lstrip("/"))
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, text=text_query, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        det = processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
            text_labels=[candidate_labels],
        )[0]

        n_this_image = 0
        for box, score, text_label in zip(det["boxes"], det["scores"], det["text_labels"]):
            label = label_to_name.get(text_label) or label_to_name.get(
                _match_label(text_label, candidate_labels) or ""
            )
            if label is None:
                unmatched += 1
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            results.append({
                "image_id": img_meta["id"],
                "category_id": name_to_cat_id[label],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(score),
            })
            n_this_image += 1

        print(f"[{i + 1}/{len(gt['images'])}] {img_meta['path']}: {n_this_image} detections")

    torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start_time

    if unmatched:
        print(f"WARNING: {unmatched} detections had text that didn't match any known category and were dropped.")

    return results, elapsed_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default="/tmp/loco_sample_paths.txt")
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--arch", choices=list(MODEL_CLASSES), default="grounding_dino",
                         help="grounding_dino = IDEA-Research checkpoints; mm_grounding_dino = openmmlab-community checkpoints")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--results-csv", default="results/loco_eval_results_zeroshot.csv")
    parser.add_argument("--run", default=None, help="Run name, e.g. 'gdino_v3'. Auto-incremented as 'gdino_vN' if omitted.")
    parser.add_argument(
        "--message", required=True,
        help="Required. What does this run represent? "
             "e.g. \"grounding-dino-tiny baseline, single multi-phrase query per image\" or "
             "\"mm_grounding_dino_large_all, lowered thresholds for its lower score calibration\"",
    )
    args = parser.parse_args()

    print(f"=== {args.message} ===\n")

    with open(args.sample_list) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    gt = build_gt_subset(args.annotations, image_paths)
    print(f"Evaluating on {len(gt['images'])} images, {len(gt['annotations'])} GT boxes")

    results, elapsed_sec = run_predictions(gt, args.dataset_root, args.model_id, args.arch,
                                            args.box_threshold, args.text_threshold)
    print(f"Inference: {elapsed_sec:.1f}s total, {elapsed_sec / len(gt['images']):.3f}s/image")

    # reuse eval_loco's evaluate()/CSV writer, but stamp the correct model name
    import eval_loco
    eval_loco.MODEL_NAME = (
        f"{args.model_id} (transformers, single-query grounding, "
        f"thr={args.box_threshold}/{args.text_threshold})"
    )
    evaluate(gt, results, args.message, elapsed_sec, run_name=args.run, run_prefix="gdino", csv_path=args.results_csv)


if __name__ == "__main__":
    main()
