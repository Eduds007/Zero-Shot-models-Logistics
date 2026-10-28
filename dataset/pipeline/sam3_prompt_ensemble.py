"""
SAM3 prompt ensembles: the "diversified attribute enhancement" (DAE) idea
of AugGen (Han et al., Neurocomputing 666, 2026) carried over to the
pipeline's annotator. AugGen has an LLM write variants of each unseen
class's attributes and fuses what the generator makes of each; here each
class gets several text prompts (variants written by Claude, an LLM, from
the original prompt) and SAM3's boxes for them are fused per image.

Fusion (no learned per-prompt weights: AugGen's adaptive g_nor would act as
a per-prompt score threshold, and the pipeline keeps one fixed 0.5):
  max   all prompts' boxes pooled, greedy NMS at IoU 0.5 (a box needs one
        prompt to find it -- recall-oriented)
  mean  same clusters, but a cluster's score is the mean over the selected
        prompts of each prompt's best score in it, 0 when a prompt missed it
        (a box needs several prompts to agree -- precision-oriented)

Prompt sets are chosen per class by greedy forward selection on box F1 at
confidence 0.5, on tuning images only:
  LOCO classes  500 annotated LOCO train-split images (warehouses 2, 3, 5)
  iw_hub        the Roboflow export's train split (344 images)
and only then scored, next to the current single prompt, on the test
images: the official LOCO val split (2277) and the Roboflow valid + test
splits (165), plus iw_hub false alarms on 500 LOCO val images. Caveat: the
Roboflow splits share near-duplicate frames, so its tuning and test images
aren't fully independent (few parameters are chosen: which prompts, which
fusion).

SAM3's raw boxes (score >= 0.05) are cached, so the selection can be re-run
without the GPU.

Usage:
    python sam3_prompt_ensemble.py
"""

import gzip
import json
import os
import random

import numpy as np

import db
from sam3_label_quality import (CONF, MATCH_IOU, N_NEGATIVE, ROBOFLOW, _gt_boxes, eval_loco, evaluate, gate_outcomes,
                                loco_val_gt, precision_recall, roboflow_gt, sam3_boxes)

CACHE = os.path.join(db.PIPELINE_ROOT, "sam3_prompt_ensemble_cache")
OUT = os.path.join(db.PIPELINE_ROOT, "sam3_prompt_ensemble.json")
LOCO_TRAIN_PATHS = os.path.join(db.DATASET_ROOT, "loco_train_paths.txt")
N_TUNE = 500
MIN_GAIN = 0.005  # F1 a prompt must add to join the set
MODES = ("max", "mean")

# the first prompt of each list is the one in use today
PROMPTS = {
    "small_load_carrier": ["plastic crate", "small load carrier", "plastic storage bin", "stackable plastic box",
                           "blue plastic container", "KLT box"],
    "forklift": ["forklift", "forklift truck", "industrial lift truck", "counterbalance forklift", "reach truck",
                 "warehouse forklift vehicle"],
    "pallet": ["pallet", "wooden pallet", "euro pallet", "shipping pallet", "plastic pallet", "wooden skid"],
    "stillage": ["wire mesh crate", "stillage", "metal mesh container", "wire mesh box pallet", "metal cage container",
                 "steel lattice box"],
    "pallet_truck": ["pallet jack", "hand pallet truck", "manual pallet truck", "pallet lifter",
                     "hand truck with forks", "electric pallet jack"],
    "iw_hub": ["yellow robot platform", "yellow mobile robot", "autonomous mobile robot", "low yellow robotic vehicle",
               "yellow transport robot", "flat yellow AGV"],
}


def _key(cls, prompt):
    return f"{cls}|{prompt}"


def cached_boxes(name, paths, prompts):
    """sam3_boxes(paths, prompts), cached in CACHE/<name>.json.gz."""
    path = os.path.join(CACHE, f"{name}.json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as f:
            boxes = json.load(f)
        if all(p in boxes and all(k in boxes[p] for k in prompts) for p in paths):
            return boxes
    boxes = sam3_boxes(paths, prompts)
    os.makedirs(CACHE, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({p: {k: [[round(v, 2) for v in b[:4]] + [round(b[4], 4)] for b in bs] for k, bs in d.items()}
                   for p, d in boxes.items()}, f)
    return boxes


def fuse(per_prompt, mode):
    """per_prompt: one box list per selected prompt -> fused box list."""
    boxes = np.array([b for bs in per_prompt for b in bs], dtype=float).reshape(-1, 5)
    if not len(boxes):
        return []
    src = np.array([i for i, bs in enumerate(per_prompt) for _ in bs])
    order = np.argsort(-boxes[:, 4])
    boxes, src = boxes[order], src[order]
    x0, y0, x1, y1 = boxes[:, :4].T
    iw = np.clip(np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0), 0, None)
    ih = np.clip(np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0), 0, None)
    area = (x1 - x0) * (y1 - y0)
    inter = iw * ih
    iou = inter / np.maximum(area[:, None] + area - inter, 1e-9)
    out, used = [], np.zeros(len(boxes), bool)
    for a in range(len(boxes)):
        if used[a]:
            continue
        members = ~used & (iou[a] >= MATCH_IOU)
        used |= members
        if mode == "max":
            score = boxes[a, 4]
        else:
            best = np.zeros(len(per_prompt))
            np.maximum.at(best, src[members], boxes[members, 4])
            score = best.mean()
        out.append((*boxes[a, :4].tolist(), float(score)))
    return out


def fused_boxes(boxes, cls, prompts, mode, gt=None):
    """{path: {cls: fused boxes}} for one class's prompt set (only gt's images if given)."""
    keep = {img["path"] for img in gt["images"]} if gt else boxes.keys()
    return {p: {cls: fuse([d.get(_key(cls, q), []) for q in prompts], mode) if len(prompts) > 1
                else d.get(_key(cls, prompts[0]), [])} for p, d in boxes.items() if p in keep}


def f1_at_conf(gt, boxes, cls, cat, prompts, mode):
    id_by_path = {img["path"]: img["id"] for img in gt["images"]}
    fused = fused_boxes(boxes, cls, prompts, mode, gt)
    return _f1(precision_recall(_gt_boxes(gt, cat),
                                {id_by_path[p]: [b for b in d[cls] if b[4] >= CONF] for p, d in fused.items()}))


def _f1(r):
    p, rc = r["precision"], r["recall"]
    return 0.0 if not p + rc or np.isnan(p) else 2 * p * rc / (p + rc)


def select(gt, boxes, cls, cat, mode):
    """Greedy forward selection of cls's prompts on F1 at CONF."""
    score = {q: f1_at_conf(gt, boxes, cls, cat, [q], mode) for q in PROMPTS[cls]}
    chosen = [max(score, key=score.get)]
    best = score[chosen[0]]
    while True:
        trials = {q: f1_at_conf(gt, boxes, cls, cat, chosen + [q], mode) for q in PROMPTS[cls] if q not in chosen}
        if not trials or max(trials.values()) < best + MIN_GAIN:
            return chosen, best, score
        q = max(trials, key=trials.get)
        chosen.append(q)
        best = trials[q]


def subset_gt(gt, keep):
    images = [img for img in gt["images"] if keep(img["path"])]
    ids = {img["id"] for img in images}
    return {"images": images, "annotations": [a for a in gt["annotations"] if a["image_id"] in ids],
            "categories": gt["categories"]}


def loco_tune_gt():
    with open(LOCO_TRAIN_PATHS) as f:
        paths = random.Random(0).sample([l.strip() for l in f if l.strip()], N_TUNE)
    gt = eval_loco.build_gt_subset(os.path.join(db.DATASET_ROOT, "annotations", "loco-all-v1.json"), paths)
    for img in gt["images"]:
        img["path"] = os.path.join(db.DATASET_ROOT, img["path"].lstrip("/"))
    return gt


def _summary(r):
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} | {"f1": round(_f1(r), 4)}


def run_class(cls, cat, tune_gt, tune_boxes, test_gt, test_boxes):
    sel = {m: select(tune_gt, tune_boxes, cls, cat, m) for m in MODES}
    mode = max(MODES, key=lambda m: sel[m][1])
    chosen = sel[mode][0]
    current = PROMPTS[cls][0]
    res = {
        "tuning": {"per_prompt_f1": {q: round(v, 4) for q, v in sel[mode][2].items()},
                   "selected": {m: {"prompts": sel[m][0], "f1": round(sel[m][1], 4)} for m in MODES},
                   "chosen_mode": mode},
        "test": {"current_prompt": _summary(evaluate(test_gt, fused_boxes(test_boxes, cls, [current], "max", test_gt),
                                                     cls, cat)),
                 "ensemble": _summary(evaluate(test_gt, fused_boxes(test_boxes, cls, chosen, mode, test_gt), cls, cat))},
    }
    t = res["test"]
    print(f"{cls}: {mode} {chosen} | test current P={t['current_prompt']['precision']:.3f} "
          f"R={t['current_prompt']['recall']:.3f} AP50={t['current_prompt']['ap50']:.3f} -> ensemble "
          f"P={t['ensemble']['precision']:.3f} R={t['ensemble']['recall']:.3f} AP50={t['ensemble']['ap50']:.3f}",
          flush=True)
    return res, chosen, mode


def main():
    loco_names = list(eval_loco.CATEGORY_PROMPTS)
    loco_prompts = {_key(c, q): q for c in loco_names for q in PROMPTS[c]}
    iw_prompts = {_key("iw_hub", q): q for q in PROMPTS["iw_hub"]}
    report = {"conf": CONF, "match_iou": MATCH_IOU, "min_gain": MIN_GAIN, "prompts": PROMPTS, "classes": {}}

    tune = loco_tune_gt()
    tune_boxes = cached_boxes("loco_tune", [img["path"] for img in tune["images"]], loco_prompts)
    val = loco_val_gt()
    val_paths = [img["path"] for img in val["images"]]
    val_boxes = cached_boxes("loco_val", val_paths, loco_prompts)
    cat_by_name = {c["name"]: c["id"] for c in val["categories"]}
    for cls in loco_names:
        report["classes"][cls], _, _ = run_class(cls, cat_by_name[cls], tune, tune_boxes, val, val_boxes)

    rf = roboflow_gt()
    rf_boxes = cached_boxes("roboflow", [img["path"] for img in rf["images"]], iw_prompts)
    in_train = lambda p: os.path.dirname(p) == os.path.join(ROBOFLOW, "train")  # noqa: E731
    rf_tune, rf_test = subset_gt(rf, in_train), subset_gt(rf, lambda p: not in_train(p))
    res, chosen, mode = run_class("iw_hub", 0, rf_tune, rf_boxes, rf_test, rf_boxes)
    neg = cached_boxes("loco_val_negatives", val_paths[:N_NEGATIVE], iw_prompts)
    for name, prompts, m in (("current_prompt", [PROMPTS["iw_hub"][0]], "max"), ("ensemble", chosen, mode)):
        rf_fused = fused_boxes(rf_boxes, "iw_hub", prompts, m, rf_test)
        neg_fused = fused_boxes(neg, "iw_hub", prompts, m)
        fa = [sum(b[4] >= CONF for b in d["iw_hub"]) for d in neg_fused.values()]
        res["test"][name] |= {"gate": gate_outcomes(rf_fused, "iw_hub"),
                              "false_alarm_images_on_loco": int(sum(x > 0 for x in fa)), "n_negatives": len(fa)}
    print(f"iw_hub gate / false alarms: {json.dumps({k: res['test'][k] for k in res['test']})}", flush=True)
    report["classes"]["iw_hub"] = res

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"PROMPT ENSEMBLE DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
