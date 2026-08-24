"""
Evaluate OWLv2 (HF transformers) on the same LOCO sample/GT/CSV pipeline as
eval_loco.py and eval_loco_grounding_dino.py, so results are directly
comparable across all three.

Like Grounding DINO, OWLv2 grounds all category phrases in a single forward
pass per image (one call with the list of candidate label strings).

Usage:
    python eval_loco_owlv2.py --sample-list /tmp/loco_sample_paths.txt --message "..."
"""

import argparse
import os
import time

from PIL import Image

from eval_loco import CATEGORY_PROMPTS, build_gt_subset, evaluate


def run_predictions(gt, dataset_root, model_id, threshold=0.1):
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    label_to_name = {CATEGORY_PROMPTS[name]: name for name in cat_id_to_name.values()}
    candidate_labels = list(label_to_name.keys())

    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to("cuda").eval()

    results = []
    start_time = time.perf_counter()
    for i, img_meta in enumerate(gt["images"]):
        image_path = os.path.join(dataset_root, img_meta["path"].lstrip("/"))
        image = Image.open(image_path).convert("RGB")

        inputs = processor(text=[candidate_labels], images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        det = processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[image.size[::-1]],
            text_labels=[candidate_labels],
        )[0]

        for box, score, text_label in zip(det["boxes"], det["scores"], det["text_labels"]):
            label = label_to_name[text_label]
            x0, y0, x1, y1 = [float(v) for v in box]
            results.append({
                "image_id": img_meta["id"],
                "category_id": name_to_cat_id[label],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(score),
            })

        print(f"[{i + 1}/{len(gt['images'])}] {img_meta['path']}: {len(det['boxes'])} detections")

    torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default="/tmp/loco_sample_paths.txt")
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--model-id", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--results-csv", default="results/loco_eval_results.csv")
    parser.add_argument("--run", default=None, help="Run name, e.g. 'owlv2_v2'. Auto-incremented as 'owlv2_vN' if omitted.")
    parser.add_argument(
        "--message", required=True,
        help="Required. What does this run represent? "
             "e.g. \"owlv2-base-patch16-ensemble baseline, single multi-label query per image, threshold=0.1\"",
    )
    args = parser.parse_args()

    print(f"=== {args.message} ===\n")

    with open(args.sample_list) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    gt = build_gt_subset(args.annotations, image_paths)
    print(f"Evaluating on {len(gt['images'])} images, {len(gt['annotations'])} GT boxes")

    results, elapsed_sec = run_predictions(gt, args.dataset_root, args.model_id, args.threshold)
    print(f"Inference: {elapsed_sec:.1f}s total, {elapsed_sec / len(gt['images']):.3f}s/image")

    import eval_loco
    eval_loco.MODEL_NAME = f"{args.model_id} (transformers, single-query grounding, thr={args.threshold})"
    evaluate(gt, results, args.message, elapsed_sec, run_name=args.run, run_prefix="owlv2", csv_path=args.results_csv)


if __name__ == "__main__":
    main()
