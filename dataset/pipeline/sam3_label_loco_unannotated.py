"""
SAM3 pseudo-labels for a sample of LOCO's ~32k unannotated images (5
warehouses), for self-training on real warehouse diversity in the spirit of
Mayershofer et al., "Improving visual object detection using synthetic
self-training" (Logistics Journal, 2021).

Downloads the unannotated subsets from the official LOCO webdisk if missing,
samples the same number of images from each chosen subset (seeded), and
runs SAM3 with eval_loco.py's per-class text prompts at a single 0.5
confidence threshold -- eval_loco's low forklift threshold + top-2 was tuned
for recall when measuring AP, which as training labels would be mostly false
boxes. Writes a YOLO images/ + labels/ folder (images are symlinks into the
extracted zips).

The official LOCO split is by warehouse (train = subsets 2, 3, 5; val =
subsets 1, 4), so --subsets 2 3 5 keeps the val warehouses unseen; sampling
all 5 subsets puts frames <1 s from val images into training.

Usage:
    python sam3_label_loco_unannotated.py --n 5000 --subsets 2 3 5
"""

import argparse
import json
import os
import random
import subprocess
import sys
import urllib.request

import db

sys.path.insert(0, db.DATASET_ROOT)
from eval_loco import CATEGORY_PROMPTS  # noqa: E402

ROOT = os.path.join(db.DATASET_ROOT, "loco_unannotated_full")
URL = ("https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=73e976ba"
       "&path=LOCO%5Cv1%5Cnot-annotated%5Csubset-{k}-jpg.zip")
CONF = 0.5
SEED = 0


def download_and_extract():
    for k in range(1, 6):
        out_dir = os.path.join(ROOT, "images", f"subset-{k}")
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            continue
        zip_path = os.path.join(ROOT, "zips", f"subset-{k}-jpg.zip")
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        if not os.path.exists(zip_path):
            print(f"downloading subset-{k}", flush=True)
            urllib.request.urlretrieve(URL.format(k=k), zip_path + ".part")
            os.replace(zip_path + ".part", zip_path)
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(["unzip", "-q", "-o", "-j", zip_path, "*.jpg", "-d", out_dir], check=True)
        print(f"subset-{k}: {len(os.listdir(out_dir))} images", flush=True)


def sample(n, annotated_names, subsets):
    rng = random.Random(SEED)
    per_subset = n // len(subsets)
    picked = []
    for k in subsets:
        d = os.path.join(ROOT, "images", f"subset-{k}")
        files = sorted(f for f in os.listdir(d) if f.endswith(".jpg") and f not in annotated_names)
        picked += [os.path.join(d, f) for f in rng.sample(files, min(per_subset, len(files)))]
    return picked


def label(paths, out_dir, registry):
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=CONF)

    loco = db.loco_classes(registry)
    img_dir, lbl_dir = os.path.join(out_dir, "images"), os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    counts = {c["name"]: 0 for c in loco}
    for i, p in enumerate(paths):
        name = f"lunann_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)}"
        lbl_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lbl_path):
            continue
        image = Image.open(p).convert("RGB")
        w, h = image.size
        state = processor.set_image(image)
        lines = []
        for c in loco:
            state = processor.set_text_prompt(prompt=CATEGORY_PROMPTS[c["name"]], state=state)
            for x0, y0, x1, y1 in state["boxes"].cpu().tolist():
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                bw, bh = (x1 - x0) / w, (y1 - y0) / h
                lines.append(f"{c['class_idx']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts[c["name"]] += 1
        dst = os.path.join(img_dir, name)
        if not os.path.exists(dst):
            os.symlink(p, dst)
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if (i + 1) % 250 == 0:
            print(f"[{i + 1}/{len(paths)}] boxes so far: {counts}", flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--subsets", type=int, nargs="+", default=[2, 3, 5],
                        help="LOCO subsets (warehouses) to sample; default = the official train warehouses")
    args = parser.parse_args()
    registry = db.load_class_registry()
    download_and_extract()
    with open(os.path.join(db.DATASET_ROOT, "annotations", "loco-all-v1.json")) as f:
        annotated = {img["file_name"] for img in json.load(f)["images"]}
    paths = sample(args.n, annotated, args.subsets)
    tag = "" if args.subsets == [1, 2, 3, 4, 5] else "_wh" + "".join(map(str, args.subsets))
    out_dir = os.path.join(ROOT, f"sam3_labeled_{args.n}{tag}")
    counts = label(paths, out_dir, registry)
    print(f"SAM3 LABELING DONE: {len(paths)} images -> {out_dir}; boxes (this run): {counts}", flush=True)


if __name__ == "__main__":
    main()
