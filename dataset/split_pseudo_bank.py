"""
Splits dataset/sam3_v6_pseudo (128 images, flat, no split) into train/val for
fine-tuning, so the model can be evaluated on frames it never trained on.

Two strategies:
  - "temporal" (default): a contiguous cut (first 80% of frames -> train,
    last 20% -> val). Safer: the source frames come from one continuous 2fps
    walkthrough, so adjacent frames are near-duplicates -- a random split
    risks leaking near-identical frames across train/val and inflating val.
  - "random": shuffles before splitting (fixed seed for reproducibility).
    Requested explicitly despite the leakage risk above -- writes to a
    separate output dir so both splits can be compared side by side.

Usage:
    python split_pseudo_bank.py [--strategy temporal|random] [--seed 0]
"""

import argparse
import glob
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "sam3_v6_pseudo")
TRAIN_FRACTION = 0.8

CLASS_ORDER = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, "sam3_v6_pseudo_split" if args.strategy == "temporal"
                            else "sam3_v6_pseudo_split_random")

    image_paths = sorted(glob.glob(os.path.join(SRC_DIR, "images", "*.jpg")))
    if args.strategy == "random":
        random.Random(args.seed).shuffle(image_paths)
    n_train = round(len(image_paths) * TRAIN_FRACTION)
    splits = {"train": image_paths[:n_train], "val": image_paths[n_train:]}
    if args.strategy == "random":
        for s in splits:
            splits[s] = sorted(splits[s])  # cosmetic: stable file order on disk

    for split in splits:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    for split, paths in splits.items():
        for img_path in paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            dst_img = os.path.join(out_dir, "images", split, stem + ".jpg")
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(img_path), dst_img)

            src_label = os.path.join(SRC_DIR, "labels", stem + ".txt")
            dst_label = os.path.join(out_dir, "labels", split, stem + ".txt")
            if not os.path.exists(dst_label):
                os.symlink(os.path.abspath(src_label), dst_label)

    yaml_path = os.path.join(out_dir, "loco.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_ORDER)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASS_ORDER):
            f.write(f"  {i}: {name}\n")

    print(f"train: {len(splits['train'])} images "
          f"({os.path.basename(splits['train'][0])} .. {os.path.basename(splits['train'][-1])})")
    print(f"val:   {len(splits['val'])} images "
          f"({os.path.basename(splits['val'][0])} .. {os.path.basename(splits['val'][-1])})")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
