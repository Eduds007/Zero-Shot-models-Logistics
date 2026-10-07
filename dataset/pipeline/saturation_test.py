"""
Saturation test for the SINGLE-model pipeline (no specialist): how many new
classes can the production recipe keep adding before the mean AP_50 of the
new classes stops dropping?

There is only one real new class (iw_hub), so new classes are simulated with
COCO categories (the standard source in incremental-detection benchmarks).
'person' is excluded: it appears unlabeled in LOCO images, which would teach
the model that people are background.

Each step k trains with the pipeline's own recipe (run_finetune.finetune:
restart from the pristine LOCO v8m baseline, freeze=21) on the first k COCO
classes plus a stratified LOCO replay, then evaluates:
  - AP_50 of every new class on a fixed COCO val2017 subset,
  - AP_50 of the 5 LOCO classes on the official LOCO val (conf=0.001 method).
Copy-paste augmentation is deliberately left out: it was an iw_hub-specific
booster for a tiny dataset and would multiply training time with 40-70
classes.

Per-class image selections are fixed by seed, so steps are nested (step 10
contains exactly step 5's data plus 5 more classes). Results are written
after every step to saturation/results.json and saturation/SUMMARY.md, and a
re-run skips finished steps -- so the whole thing survives being restarted.

Usage:
    python saturation_test.py            # steps 1,2,5,10,20,40 (+70 if still dropping)
"""

import json
import os
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import db
from build_training_batch import _stratified_loco_sample, _symlink_pair, LOCO_SRC
from promote_if_better import ap50_per_category, evaluate_model, predict_yolo
from run_finetune import finetune

COCO_ROOT = os.path.join(db.DATASET_ROOT, "coco")
OUT_ROOT = os.path.join(db.PIPELINE_ROOT, "saturation")
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{file}"

N_CLASSES_MAX = 70
# 80/class (the robot's real volume for iw_hub) left COCO classes starting at
# AP_50 ~0.01: a curve that starts on the floor can't show where it stops
# dropping. 300 train / up to 200 eval images per class measures the model's
# capacity limit instead, with a less noisy per-class test set.
TRAIN_PER_CLASS = 300
EVAL_PER_CLASS = 200
LOCO_RATIO = 10
STEPS = [1, 2, 5, 10, 20, 40]
EXTRA_STEP = 70
PLATEAU_DROP = 0.01  # mean new-class AP_50 dropping less than this between steps = stopped dropping
SEED = 0


# ----------------------------------------------------------------------------- COCO

def load_coco():
    ann_dir = os.path.join(COCO_ROOT, "annotations")
    if not os.path.exists(os.path.join(ann_dir, "instances_val2017.json")):
        os.makedirs(COCO_ROOT, exist_ok=True)
        zip_path = os.path.join(COCO_ROOT, "annotations.zip")
        print(f"downloading {ANN_URL}", flush=True)
        urllib.request.urlretrieve(ANN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for name in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
                z.extract(name, COCO_ROOT)
        os.remove(zip_path)
    coco = {}
    for split in ("train2017", "val2017"):
        with open(os.path.join(ann_dir, f"instances_{split}.json")) as f:
            coco[split] = json.load(f)
    return coco


def index(coco_split):
    imgs = {img["id"]: img for img in coco_split["images"]}
    anns_by_img, imgs_by_cat = {}, {}
    for a in coco_split["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
        imgs_by_cat.setdefault(a["category_id"], set()).add(a["image_id"])
    return imgs, anns_by_img, imgs_by_cat


def plan(coco):
    """Fixed, seeded class order and per-class image picks (nested across steps)."""
    rng = random.Random(SEED)
    cats = [c for c in coco["val2017"]["categories"] if c["name"] != "person"]
    rng.shuffle(cats)
    cats = cats[:N_CLASSES_MAX]
    _, _, train_by_cat = index(coco["train2017"])
    _, _, val_by_cat = index(coco["val2017"])
    picks = []
    for c in cats:
        train = sorted(train_by_cat[c["id"]])
        val = sorted(val_by_cat[c["id"]])
        picks.append({"cat": c, "train": rng.sample(train, min(TRAIN_PER_CLASS, len(train))),
                      "eval": rng.sample(val, min(EVAL_PER_CLASS, len(val)))})
    return picks


def download(img_meta, split):
    dst = os.path.join(COCO_ROOT, "images", split, img_meta["file_name"])
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(IMG_URL.format(split=split, file=img_meta["file_name"]), tmp)
            os.replace(tmp, dst)
            return dst
        except OSError:
            if attempt == 2:
                raise
    return dst


def download_all(coco, picks):
    jobs = []
    for split, key in (("train2017", "train"), ("val2017", "eval")):
        imgs, _, _ = index(coco[split])
        ids = {i for p in picks for i in p[key]}
        jobs += [(imgs[i], split) for i in ids]
    print(f"downloading {len(jobs)} COCO images (skips ones already on disk)", flush=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda j: download(*j), jobs))


# ----------------------------------------------------------------------- one step

def yolo_lines(anns, img_meta, cat_to_idx):
    w, h = img_meta["width"], img_meta["height"]
    lines = []
    for a in anns:
        if a["category_id"] not in cat_to_idx or a.get("iscrowd", 0):
            continue
        x, y, bw, bh = a["bbox"]
        lines.append(f"{cat_to_idx[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                     f"{bw / w:.6f} {bh / h:.6f}")
    return lines


def build_step(k, coco, picks, registry):
    step_dir = os.path.join(OUT_ROOT, f"step_{k:02d}")
    data_yaml = os.path.join(step_dir, "data.yaml")
    if os.path.exists(data_yaml):
        return data_yaml
    random.seed(SEED + k)
    loco = db.loco_classes(registry)
    active = picks[:k]
    cat_to_idx = {p["cat"]["id"]: len(loco) + i for i, p in enumerate(active)}
    train_imgs, train_anns, _ = index(coco["train2017"])

    # new-class images: 80/class for train; a 20% slice of them becomes the training-time val
    new_ids = sorted({i for p in active for i in p["train"]})
    random.shuffle(new_ids)
    n_val = max(1, len(new_ids) // 5)
    split_ids = {"val": new_ids[:n_val], "train": new_ids[n_val:]}
    for split, ids in split_ids.items():
        img_dir = os.path.join(step_dir, "images", split)
        lbl_dir = os.path.join(step_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in ids:
            meta = train_imgs[i]
            dst_img = os.path.join(img_dir, meta["file_name"])
            if not os.path.exists(dst_img):
                os.symlink(os.path.join(COCO_ROOT, "images", "train2017", meta["file_name"]), dst_img)
            with open(os.path.join(lbl_dir, os.path.splitext(meta["file_name"])[0] + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines(train_anns.get(i, []), meta, cat_to_idx)) + "\n")

        # LOCO replay, same stratified sampler the pipeline uses
        loco_img_dir = os.path.join(LOCO_SRC, "images", split)
        loco_lbl_dir = os.path.join(LOCO_SRC, "labels", split)
        loco_all = sorted(f for f in os.listdir(loco_img_dir) if f.endswith(".jpg"))
        target = min(len(loco_all), LOCO_RATIO * len(ids))
        for fname in _stratified_loco_sample(loco_img_dir, loco_lbl_dir, loco_all, target,
                                             40 if split == "train" else 8, {c["class_idx"] for c in loco}):
            stem = os.path.splitext(fname)[0]
            _symlink_pair(os.path.join(loco_img_dir, fname), os.path.join(img_dir, fname),
                          os.path.join(lbl_dir, stem + ".txt"), src_lbl=os.path.join(loco_lbl_dir, stem + ".txt"))

    names = {c["class_idx"]: c["name"] for c in loco}
    names.update({len(loco) + i: p["cat"]["name"] for i, p in enumerate(active)})
    with open(data_yaml, "w") as f:
        f.write(f"path: {step_dir}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n")
    return data_yaml


def eval_new_classes(weights, k, coco, picks, n_loco):
    val_imgs, val_anns, _ = index(coco["val2017"])
    active = picks[:k]
    cat_ids = {p["cat"]["id"] for p in active}
    img_ids = sorted({i for p in active for i in p["eval"]})
    gt = {
        "images": [dict(val_imgs[i], path=os.path.join(COCO_ROOT, "images", "val2017", val_imgs[i]["file_name"]))
                   for i in img_ids],
        "annotations": [a for i in img_ids for a in val_anns.get(i, []) if a["category_id"] in cat_ids],
        "categories": [p["cat"] for p in active],
    }
    idx_to_cat = {n_loco + i: p["cat"]["id"] for i, p in enumerate(active)}
    results, _ = predict_yolo(weights, gt, idx_to_cat, batch=8)
    ap = ap50_per_category(gt, results)
    return {p["cat"]["name"]: ap.get(p["cat"]["id"], 0.0) for p in active}


# --------------------------------------------------------------------------- report

def write_summary(results):
    with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = ["# Saturation test (single model, pipeline recipe, COCO classes as new classes)", "",
             "| new classes | mean AP50 new classes | min AP50 new class | LOCO mAP50 (5) | forklift | pallet_truck |",
             "|---|---|---|---|---|---|"]
    for r in results:
        new = list(r["new_classes"].values())
        lines.append(f"| {r['k']} | {sum(new) / len(new):.3f} | {min(new):.3f} | "
                     f"{sum(r['loco'].values()) / len(r['loco']):.3f} | {r['loco']['forklift']:.3f} | "
                     f"{r['loco']['pallet_truck']:.3f} |")
    plateau = plateau_step(results)
    lines += ["", f"Mean new-class AP50 stops dropping (step-to-step drop < {PLATEAU_DROP}) at: "
              f"{plateau if plateau else 'not reached in the steps run'}"]
    with open(os.path.join(OUT_ROOT, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def mean_new(r):
    v = list(r["new_classes"].values())
    return sum(v) / len(v)


def plateau_step(results):
    for prev, cur in zip(results, results[1:]):
        if mean_new(prev) - mean_new(cur) < PLATEAU_DROP:
            return f"{prev['k']} -> {cur['k']} classes"
    return None


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    registry = db.load_class_registry()
    loco_names = [c["name"] for c in db.loco_classes(registry)]
    coco = load_coco()
    picks = plan(coco)
    with open(os.path.join(OUT_ROOT, "class_order.json"), "w") as f:
        json.dump([p["cat"]["name"] for p in picks], f, indent=2)
    download_all(coco, picks)

    results_path = os.path.join(OUT_ROOT, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else []
    done = {r["k"] for r in results}

    steps = list(STEPS)
    while steps:
        k = steps.pop(0)
        if k not in done:
            print(f"\n===== step k={k} =====", flush=True)
            data_yaml = build_step(k, coco, picks, registry)
            weights = finetune(data_yaml, f"saturation_k{k:02d}")
            results.append({"k": k, "weights": weights,
                            "new_classes": eval_new_classes(weights, k, coco, picks, len(loco_names)),
                            "loco": evaluate_model(weights, registry, loco_names, db.DATASET_ROOT)})
            results.sort(key=lambda r: r["k"])
            write_summary(results)
            print(open(os.path.join(OUT_ROOT, "SUMMARY.md")).read(), flush=True)
        if not steps and k == STEPS[-1] and EXTRA_STEP not in done:
            by_k = {r["k"]: r for r in results}
            if mean_new(by_k[STEPS[-2]]) - mean_new(by_k[STEPS[-1]]) >= PLATEAU_DROP:
                steps.append(EXTRA_STEP)
    print("SATURATION TEST DONE", flush=True)


if __name__ == "__main__":
    main()
