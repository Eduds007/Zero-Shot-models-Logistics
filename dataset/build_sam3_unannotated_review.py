"""
Runs SAM3 (sam3_v6 config) on a random sample of LOCO images that have no
official annotation (dataset/loco_unannotated_full/images/) and builds an
offline viewer to judge every predicted box by hand -- there is no GT here,
so the reviewer's verdicts are what give precision, plus a per-image count of
objects SAM3 missed for a recall estimate.

Output (default sam3_gt_review/unannotated/, served next to the GT vs SAM3
viewer): index.html + data.js + images/ + predictions.json.

Usage (from dataset/):
    ../.venv/bin/python build_sam3_unannotated_review.py [--n 100] [--seed 42]
"""

import argparse
import json
import os
import random
import shutil

from PIL import Image

from eval_loco import CATEGORY_PROMPTS, run_predictions

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-dir", default="loco_unannotated_full/images")
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json",
                        help="Only used for the category ids/names.")
    parser.add_argument("--out", default="sam3_gt_review/unannotated")
    args = parser.parse_args()

    paths = sorted(os.path.join(root, f) for root, _, files in os.walk(args.images_dir)
                   for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
    sample = random.Random(args.seed).sample(paths, args.n)
    print(f"Sampled {len(sample)} of {len(paths)} unannotated images (seed {args.seed})")

    with open(args.annotations) as f:
        categories = json.load(f)["categories"]
    images = []
    for i, p in enumerate(sample):
        w, h = Image.open(p).size
        images.append({"id": i + 1, "path": p, "width": w, "height": h})

    detections, elapsed = run_predictions({"images": images, "categories": categories}, ".")
    print(f"SAM3: {elapsed:.1f}s, {len(detections)} detections")

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    with open(os.path.join(args.out, "predictions.json"), "w") as f:
        json.dump({"seed": args.seed, "images": images, "categories": categories,
                   "detections": detections}, f)

    dets_by_image = {}
    for d in detections:
        dets_by_image.setdefault(d["image_id"], []).append(d)
    out_images = []
    for img in images:
        name = f"{img['id']:03d}.jpg"
        shutil.copyfile(img["path"], os.path.join(args.out, "images", name))
        dets = sorted(dets_by_image.get(img["id"], []), key=lambda d: -d["score"])
        out_images.append({**img, "file": f"images/{name}",
                           "det": [{"id": f"det{k}", "cat": d["category_id"], "bbox": d["bbox"],
                                    "score": d["score"]} for k, d in enumerate(dets)]})

    data = {
        "config": "sam3_v6",
        "seed": args.seed,
        "categories": [{"id": c["id"], "name": c["name"],
                        "prompt": CATEGORY_PROMPTS.get(c["name"], c["name"])} for c in categories],
        "images": out_images,
    }
    with open(os.path.join(args.out, "data.js"), "w") as f:
        f.write("window.REVIEW_DATA = ")
        json.dump(data, f)
        f.write(";\n")
    shutil.copyfile(os.path.join(HERE, "sam3_pred_review_viewer.html"), os.path.join(args.out, "index.html"))
    print(f"-> open {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
