"""
Builds a side-by-side GT vs SAM3 review viewer for a random sample of
annotated LOCO images.

1. Draws --n random images from the LOCO annotations (fixed seed). The draw is
   repeated with the next seed until all 5 classes appear in the sample's GT,
   so the sample stays uniformly random but no class is left without examples.
2. Runs SAM3 with the exact sam3_v6 config (prompts/thresholds/top-k from
   eval_loco.py).
3. Matches predictions to GT per image and per class (greedy by score,
   IoU >= --iou) into TP / FP / FN, with an automatic hint for each FP
   (wrong class, loose box, duplicate, no overlap).
4. Writes <out>/index.html + data.js + images/, an offline viewer where every
   FP and FN can be tagged by hand (e.g. "real object missing from GT"). The
   tags are exported as CSV and scored by score_sam3_gt_review.py.

Usage (from dataset/):
    ../.venv/bin/python build_sam3_gt_review.py [--n 100] [--seed 42] [--out sam3_gt_review]
"""

import argparse
import json
import os
import random
import shutil

from eval_loco import CATEGORY_PROMPTS, build_gt_subset, run_predictions

HERE = os.path.dirname(os.path.abspath(__file__))


def sample_images(coco, n, seed, max_tries=1000):
    all_cats = {c["id"] for c in coco["categories"]}
    cats_by_image = {}
    for a in coco["annotations"]:
        cats_by_image.setdefault(a["image_id"], set()).add(a["category_id"])

    for k in range(max_tries):
        rng = random.Random(seed + k)
        images = rng.sample(coco["images"], n)
        covered = set().union(*(cats_by_image.get(img["id"], set()) for img in images))
        if covered == all_cats:
            return images, seed + k
    raise RuntimeError(f"no sample of {n} images covering all classes in {max_tries} tries")


def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_image(gts, dets, iou_thr):
    """Greedy COCO-style matching per class. Returns (gt_items, det_items),
    each item carrying its status (TP/FP/FN), matched partner and best IoU."""
    gt_items = [{"id": f"gt{g['id']}", "cat": g["category_id"], "bbox": g["bbox"],
                 "status": "FN", "match": None} for g in gts]
    det_items = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        det_items.append({"cat": d["category_id"], "bbox": d["bbox"], "score": d["score"],
                          "status": "FP", "match": None, "hint": None})

    for i, d in enumerate(det_items):
        d["id"] = f"det{i}"
        best, best_iou = None, iou_thr
        for g in gt_items:
            if g["cat"] != d["cat"] or g["match"] is not None:
                continue
            v = iou(d["bbox"], g["bbox"])
            if v >= best_iou:
                best, best_iou = g, v
        if best is not None:
            d["status"], d["match"], d["iou"] = "TP", best["id"], round(best_iou, 3)
            best["status"], best["match"] = "TP", d["id"]

    # hints for FPs, so the reviewer starts from the most likely explanation
    for d in det_items:
        if d["status"] != "FP":
            continue
        same = max((iou(d["bbox"], g["bbox"]) for g in gt_items if g["cat"] == d["cat"]), default=0.0)
        other = max((iou(d["bbox"], g["bbox"]) for g in gt_items if g["cat"] != d["cat"]), default=0.0)
        dup = max((iou(d["bbox"], t["bbox"]) for t in det_items
                   if t["status"] == "TP" and t["cat"] == d["cat"]), default=0.0)
        d["iou"] = round(same, 3)
        if other >= iou_thr:
            d["hint"] = "wrong_class"
        elif dup >= iou_thr:
            d["hint"] = "duplicate"
        elif same >= 0.1:
            d["hint"] = "bad_box"
        else:
            d["hint"] = "no_gt_overlap"
    for g in gt_items:
        if g["status"] == "FN":
            g["iou"] = round(max((iou(g["bbox"], d["bbox"]) for d in det_items if d["cat"] == g["cat"]),
                                 default=0.0), 3)
    return gt_items, det_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--out", default="sam3_gt_review")
    parser.add_argument("--predictions", default=None,
                        help="Reuse a predictions.json from a previous run instead of running SAM3 again.")
    args = parser.parse_args()

    if args.predictions:
        with open(args.predictions) as f:
            saved = json.load(f)
        gt, detections, used_seed = saved["gt"], saved["detections"], saved["seed"]
    else:
        with open(args.annotations) as f:
            coco = json.load(f)
        images, used_seed = sample_images(coco, args.n, args.seed)
        gt = build_gt_subset(args.annotations, [img["path"] for img in images])
        print(f"Sampled {len(gt['images'])} images (seed {used_seed}), {len(gt['annotations'])} GT boxes")
        detections, elapsed = run_predictions(gt, args.dataset_root)
        print(f"SAM3: {elapsed:.1f}s, {len(detections)} detections")

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    with open(os.path.join(args.out, "predictions.json"), "w") as f:
        json.dump({"seed": used_seed, "gt": gt, "detections": detections}, f)

    gts_by_image, dets_by_image = {}, {}
    for a in gt["annotations"]:
        gts_by_image.setdefault(a["image_id"], []).append(a)
    for d in detections:
        dets_by_image.setdefault(d["image_id"], []).append(d)

    out_images = []
    for img in gt["images"]:
        src = os.path.join(args.dataset_root, img["path"].lstrip("/"))
        name = f"{img['id']}.jpg"
        shutil.copyfile(src, os.path.join(args.out, "images", name))
        gt_items, det_items = match_image(gts_by_image.get(img["id"], []),
                                          dets_by_image.get(img["id"], []), args.iou)
        out_images.append({"id": img["id"], "file": f"images/{name}", "path": img["path"],
                           "width": img["width"], "height": img["height"],
                           "gt": gt_items, "det": det_items})

    data = {
        "config": "sam3_v6",
        "seed": used_seed,
        "iou": args.iou,
        "categories": [{"id": c["id"], "name": c["name"],
                        "prompt": CATEGORY_PROMPTS.get(c["name"], c["name"])} for c in gt["categories"]],
        "images": out_images,
    }
    with open(os.path.join(args.out, "data.js"), "w") as f:
        f.write("window.REVIEW_DATA = ")
        json.dump(data, f)
        f.write(";\n")
    shutil.copyfile(os.path.join(HERE, "sam3_gt_review_viewer.html"), os.path.join(args.out, "index.html"))

    n = {s: 0 for s in ("TP", "FP", "FN")}
    for im in out_images:
        for item in im["det"] + [g for g in im["gt"] if g["status"] == "FN"]:
            n[item["status"]] += 1
    print(f"TP={n['TP']} FP={n['FP']} FN={n['FN']} -> open {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
