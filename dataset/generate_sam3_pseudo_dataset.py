"""
Builds a pseudo-labeled YOLO-format dataset from the 2fps video frames
(assets/frames_2fps), using SAM3 with the exact sam3_v6 config (see
dataset/results/loco_eval_results.csv, run 'sam3_v6'):
  - confidence_threshold=0.5 for small_load_carrier, pallet, stillage, pallet_truck
  - confidence_threshold=0.05 for forklift, then capped to the top-2
    highest-score detections

Two filters decide which frames make it into the dataset (both agreed with
the user beforehand, see conversation -- nothing here is a default guess):
  1. Blur: frames with Laplacian variance < --blur-threshold (120) are
     dropped as motion-blurred, before running SAM3 on them at all.
  2. Confidence: of the remaining frames, only those whose detections
     (pooled across all 5 categories) average a confidence score > 0.5 are
     kept. A frame with zero detections fails this by definition.

Output is a flat (no train/val split) YOLO-format set:
    dataset/sam3_v6_pseudo/images/*.jpg  (symlinks to assets/frames_2fps)
    dataset/sam3_v6_pseudo/labels/*.txt  (standard 5-column YOLO labels,
                                           no confidence column)

Usage:
    python generate_sam3_pseudo_dataset.py
"""

import argparse
import glob
import os
import statistics

from eval_loco import CATEGORY_PROMPTS

ROOT = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(ROOT, "..", "assets", "frames_2fps")
OUT_DIR = os.path.join(ROOT, "sam3_v6_pseudo")

# Same class order as convert_loco_to_yolo.py, for consistency across the repo.
CLASS_ORDER = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]

DEFAULT_THRESHOLD = 0.5
FORKLIFT_THRESHOLD = 0.05
FORKLIFT_TOP_K = 2
CONFIDENCE_KEEP_BAR = 0.5  # image-level: mean score across all its detections must exceed this


def laplacian_variance(image_path):
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()


def filter_blurry(frame_paths, blur_threshold):
    kept, dropped = [], []
    for p in frame_paths:
        v = laplacian_variance(p)
        (kept if v >= blur_threshold else dropped).append((p, v))
    return kept, dropped


def run_sam3_on_frame(processor, image):
    """Returns list of (category_name, [x0,y0,x1,y1], score) using the sam3_v6 config."""
    state = processor.set_image(image)
    detections = []

    for cat_name in CLASS_ORDER:
        prompt = CATEGORY_PROMPTS[cat_name]
        if cat_name == "forklift":
            processor.set_confidence_threshold(FORKLIFT_THRESHOLD)
        else:
            processor.set_confidence_threshold(DEFAULT_THRESHOLD)

        state = processor.set_text_prompt(prompt=prompt, state=state)
        boxes = state["boxes"].cpu().tolist()
        scores = state["scores"].cpu().tolist()

        pairs = list(zip(boxes, scores))
        if cat_name == "forklift":
            pairs.sort(key=lambda bs: bs[1], reverse=True)
            pairs = pairs[:FORKLIFT_TOP_K]

        for box, score in pairs:
            detections.append((cat_name, box, score))

    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default=FRAMES_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--blur-threshold", type=float, default=120.0,
                         help="Laplacian variance below this is dropped as blurred.")
    parser.add_argument("--confidence-keep-bar", type=float, default=CONFIDENCE_KEEP_BAR,
                         help="Frame is kept only if mean detection confidence exceeds this.")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    print(f"Found {len(frame_paths)} frames in {args.frames_dir}")

    sharp, blurry = filter_blurry(frame_paths, args.blur_threshold)
    print(f"Blur filter (threshold={args.blur_threshold}): "
          f"{len(sharp)} kept, {len(blurry)} dropped as blurred")

    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=DEFAULT_THRESHOLD)

    images_dir = os.path.join(args.out_dir, "images")
    labels_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    name_to_class_idx = {name: i for i, name in enumerate(CLASS_ORDER)}
    kept_count = 0
    rejected_low_confidence = 0

    for i, (path, blur_var) in enumerate(sharp):
        image = Image.open(path).convert("RGB")
        w, h = image.size

        detections = run_sam3_on_frame(processor, image)
        scores = [s for _, _, s in detections]
        mean_conf = statistics.mean(scores) if scores else 0.0
        passes = mean_conf > args.confidence_keep_bar

        status = "KEPT" if passes else "rejected"
        print(f"[{i + 1}/{len(sharp)}] {os.path.basename(path)}: "
              f"{len(detections)} detections, mean_conf={mean_conf:.3f} -> {status}")

        if not passes:
            rejected_low_confidence += 1
            continue

        stem = os.path.splitext(os.path.basename(path))[0]
        dst_img = os.path.join(images_dir, stem + ".jpg")
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(path), dst_img)

        lines = []
        for cat_name, box, _score in detections:
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            cx, cy, bw, bh = (max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh))
            lines.append(f"{name_to_class_idx[cat_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        kept_count += 1

    print("\n=== summary ===")
    print(f"total frames:              {len(frame_paths)}")
    print(f"dropped (blurred):         {len(blurry)}")
    print(f"dropped (mean conf <= {args.confidence_keep_bar}): {rejected_low_confidence}")
    print(f"kept in dataset:           {kept_count}")
    print(f"wrote images/labels to {args.out_dir}")


if __name__ == "__main__":
    main()
