"""
Copy-paste augmentation for a pipeline class (default iw_hub): crops each
instance via its own YOLO bbox from a training batch's own images and
pastes it onto randomly chosen background images in that same batch that
don't already contain the class, generating extra synthetic training
images -- boosts the class's effective training signal (diversity of
position/scale/background) without needing more real captures. Simple
rectangular paste (no mask, no blending), matching the bbox-level
cut-paste approach for detection (Dwibedi et al., "Cut, Paste and Learn:
Surprisingly Easy Synthesis for Instance Detection", ICCV 2017) rather
than the mask-based instance-segmentation variant.

Must run AFTER build_training_batch.py has already written the batch's
images/train + labels/train -- it reads and appends to those directories
directly, in place.

Usage:
    python copy_paste_augment.py --batch-dir path/to/training_batches/v9 \
        --class-name iw_hub --pastes-per-source 3
"""

import argparse
import os
import random

import cv2

import db


def _read_yolo_labels(path):
    boxes = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    boxes.append((int(parts[0]), *(float(x) for x in parts[1:])))
    return boxes


def _write_yolo_labels(path, boxes):
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in boxes]
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def augment(batch_dir, class_name, pastes_per_source=3, scale_range=(0.6, 1.4), seed=0, max_sources=None):
    random.seed(seed)
    registry = db.load_class_registry()
    cls = db.class_by_name(class_name, registry)
    class_idx = cls["class_idx"]

    images_dir = os.path.join(batch_dir, "images", "train")
    labels_dir = os.path.join(batch_dir, "labels", "train")
    all_images = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))

    sources, backgrounds = [], []
    for fname in all_images:
        stem = os.path.splitext(fname)[0]
        boxes = _read_yolo_labels(os.path.join(labels_dir, stem + ".txt"))
        (sources if any(b[0] == class_idx for b in boxes) else backgrounds).append(fname)

    if not sources:
        raise SystemExit(f"no '{class_name}' images found in {images_dir} -- run build_training_batch.py first")
    if not backgrounds:
        raise SystemExit(f"no background images (without '{class_name}') found in {images_dir}")

    # Classes with many instances per image (e.g. pallet_truck) can otherwise
    # generate a wildly disproportionate volume of synthetic images relative
    # to sparser classes -- cap how many source images get processed so one
    # class's augmentation can't dominate the batch's composition.
    if max_sources is not None and len(sources) > max_sources:
        sources = random.sample(sources, max_sources)

    n_written = 0
    for src_fname in sources:
        stem = os.path.splitext(src_fname)[0]
        src_img = cv2.imread(os.path.join(images_dir, src_fname))
        h, w = src_img.shape[:2]
        src_boxes = [b for b in _read_yolo_labels(os.path.join(labels_dir, stem + ".txt")) if b[0] == class_idx]

        for _, cx, cy, bw, bh in src_boxes:
            x0, y0 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
            x1, y1 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = src_img[y0:y1, x0:x1]
            crop_h, crop_w = crop.shape[:2]

            for _ in range(pastes_per_source):
                bg_fname = random.choice(backgrounds)
                bg_stem = os.path.splitext(bg_fname)[0]
                bg_img = cv2.imread(os.path.join(images_dir, bg_fname))
                bg_h, bg_w = bg_img.shape[:2]

                scale = random.uniform(*scale_range)
                new_w = max(4, min(bg_w, int(crop_w * scale)))
                new_h = max(4, min(bg_h, int(crop_h * scale)))
                resized = cv2.resize(crop, (new_w, new_h))

                paste_x = random.randint(0, bg_w - new_w)
                paste_y = random.randint(0, bg_h - new_h)
                composite = bg_img.copy()
                composite[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

                out_boxes = _read_yolo_labels(os.path.join(labels_dir, bg_stem + ".txt")) + [(
                    class_idx,
                    (paste_x + new_w / 2) / bg_w,
                    (paste_y + new_h / 2) / bg_h,
                    new_w / bg_w,
                    new_h / bg_h,
                )]

                out_stem = f"synth_copypaste_{class_name}_{n_written:05d}"
                cv2.imwrite(os.path.join(images_dir, out_stem + ".jpg"), composite)
                _write_yolo_labels(os.path.join(labels_dir, out_stem + ".txt"), out_boxes)
                n_written += 1

    print(f"sources={len(sources)} backgrounds={len(backgrounds)} -> wrote {n_written} synthetic images to {images_dir}")
    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--class-name", default="iw_hub")
    parser.add_argument("--pastes-per-source", type=int, default=3)
    parser.add_argument("--max-sources", type=int, default=None,
                         help="Cap the number of source images processed (randomly sampled), for classes "
                              "with many instances per image that would otherwise generate a "
                              "disproportionate volume of synthetic images")
    args = parser.parse_args()
    augment(args.batch_dir, args.class_name, args.pastes_per_source, max_sources=args.max_sources)


if __name__ == "__main__":
    main()
