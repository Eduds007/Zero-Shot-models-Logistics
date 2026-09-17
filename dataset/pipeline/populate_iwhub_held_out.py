"""
One-time (re-runnable) converter: builds the permanent held-out validation
set promote_if_better.py needs for the iw_hub class, from the Roboflow
export (train+valid+test, 509 images total, CC BY 4.0,
eduardo-milanez/iw.hub-detection-pdyql) -- the same external, fully
held-out set eval_robot_finetune_experiments.py's build_iwhub_gt() already
used, converted here to a plain YOLO images/+labels/ folder instead of an
in-memory COCO dict, so promote_if_better.py's yolo_dir_to_coco_gt() can
read it the same way it reads any other class's held-out set.

None of these 509 images have ever been used in training/fine-tuning
anywhere in this project -- keep it that way; this folder is read-only
input to the promotion gate, never written to by build_training_batch.py.

Usage:
    python populate_iwhub_held_out.py --export-dir /tmp/iwhub_export
"""

import argparse
import json
import os

import db

CLASS_NAME = "iw_hub"
ROBOFLOW_REAL_CATEGORY_ID = 1  # category_id 0 is Roboflow's placeholder root "iw.hub"


def convert(export_dir, out_dir, class_idx):
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    total = 0
    for split in ("train", "valid", "test"):
        ann_path = os.path.join(export_dir, split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            coco = json.load(f)

        boxes_by_image = {}
        for a in coco["annotations"]:
            if a["category_id"] != ROBOFLOW_REAL_CATEGORY_ID:
                continue
            boxes_by_image.setdefault(a["image_id"], []).append(a["bbox"])

        for img in coco["images"]:
            src = os.path.join(export_dir, split, img["file_name"])
            if not os.path.exists(src):
                continue
            # split dirs can repeat filenames across splits -- prefix keeps them unique
            stem = f"{split}_{os.path.splitext(img['file_name'])[0]}"
            dst_img = os.path.join(images_dir, stem + ".jpg")
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(src), dst_img)

            w, h = img["width"], img["height"]
            lines = []
            for x, y, bw, bh in boxes_by_image.get(img["id"], []):
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nbw, nbh = bw / w, bh / h
                lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nbw:.6f} {nbh:.6f}")
            with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            total += 1

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True,
                         help="Extracted Roboflow COCO export directory (has train/valid/test/ subfolders)")
    args = parser.parse_args()

    registry = db.load_class_registry()
    cls = db.class_by_name(CLASS_NAME, registry)
    out_dir = os.path.join(db.PROJECT_ROOT, cls["held_out_val"])

    total = convert(args.export_dir, out_dir, cls["class_idx"])
    print(f"wrote {total} images+labels to {out_dir}")


if __name__ == "__main__":
    main()
