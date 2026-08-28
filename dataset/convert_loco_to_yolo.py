"""
Converts the full LOCO dataset (COCO-format annotations) into a YOLO-format
dataset directory, using LOCO's own official train/val split (loco-sub*-v1-*.json,
by subset) rather than an ad hoc random split.

Two split modes are available:
  - "official" (default): train={subset-2,3,5}, val={subset-1,4} -- 2 of 5
    warehouses held out entirely (~55%/45%), tests generalization to unseen
    environments.
  - "papersplit": train={subset-2,3,4,5}, val={subset-1} -- matches the split
    used by Savas & Hinckeldeyn 2022 ("Critical Evaluation of LOCO dataset
    with Machine Learning", who report 64.54% mAP50 with YOLOv4): only the
    single largest subset (subset-1, ~1075 images) held out, ~79%/21%. Used
    here only to reproduce their evaluation setup for a fair comparison, not
    as our default.

Usage:
    python convert_loco_to_yolo.py [--split official|papersplit]
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = os.path.join(ROOT, "annotations")
DATASET_ROOT = ROOT  # image paths in the JSON are like "/dataset/subset-1/...jpg"

ALL_ANNOTATIONS = os.path.join(ANNOTATIONS_DIR, "loco-all-v1.json")
SPLIT_MODES = {
    "official": {
        "train": ["loco-sub2-v1-train.json", "loco-sub3-v1-train.json", "loco-sub5-v1-train.json"],
        "val": ["loco-sub1-v1-val.json", "loco-sub4-v1-val.json"],
    },
    "papersplit": {
        "train": ["loco-sub2-v1-train.json", "loco-sub3-v1-train.json",
                   "loco-sub4-v1-val.json", "loco-sub5-v1-train.json"],
        "val": ["loco-sub1-v1-val.json"],
    },
}

# Fixed class order -- matches CATEGORY_PROMPTS in eval_loco.py so class index
# is stable across the zero-shot CSV rows and this YOLO run.
CLASS_ORDER = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]


def load_split_paths(split_files):
    paths_by_split = {}
    for split, files in split_files.items():
        paths = set()
        for fname in files:
            with open(os.path.join(ANNOTATIONS_DIR, fname)) as f:
                d = json.load(f)
            paths |= {im["path"] for im in d["images"]}
        paths_by_split[split] = paths
    return paths_by_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLIT_MODES), default="official")
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, "yolo" if args.split == "official" else f"yolo_{args.split}")
    path_prefix = "loco" if args.split == "official" else f"loco_{args.split}"

    with open(ALL_ANNOTATIONS) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_class_idx = {name: i for i, name in enumerate(CLASS_ORDER)}
    assert set(cat_id_to_name.values()) == set(CLASS_ORDER), (cat_id_to_name, CLASS_ORDER)

    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    paths_by_split = load_split_paths(SPLIT_MODES[args.split])

    for split in paths_by_split:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0}
    box_counts = {"train": 0, "val": 0}
    skipped = 0
    for img in coco["images"]:
        path = img["path"]
        split = "train" if path in paths_by_split["train"] else "val" if path in paths_by_split["val"] else None
        if split is None:
            skipped += 1
            continue

        src = os.path.join(DATASET_ROOT, path.lstrip("/"))
        stem = f"{img['id']}_{os.path.splitext(img['file_name'])[0]}"
        dst_img = os.path.join(out_dir, "images", split, stem + ".jpg")
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(src), dst_img)

        w, h = img["width"], img["height"]
        lines = []
        for a in anns_by_image.get(img["id"], []):
            cls = name_to_class_idx[cat_id_to_name[a["category_id"]]]
            x, y, bw, bh = a["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            # clip -- a few LOCO boxes run slightly outside the image bounds
            cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            box_counts[split] += 1

        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        counts[split] += 1

    for split in paths_by_split:
        paths = sorted(img["path"] for img in coco["images"] if img["path"] in paths_by_split[split])
        with open(os.path.join(ROOT, f"{path_prefix}_{split}_paths.txt"), "w") as f:
            f.write("\n".join(paths) + "\n")

    yaml_path = os.path.join(out_dir, "loco.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_ORDER)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASS_ORDER):
            f.write(f"  {i}: {name}\n")

    print(f"train: {counts['train']} images, {box_counts['train']} boxes")
    print(f"val:   {counts['val']} images, {box_counts['val']} boxes")
    print(f"skipped (no split assignment): {skipped}")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
