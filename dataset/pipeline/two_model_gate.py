"""
Promotion gate for the two-model system: production's LOCO model (never
retrained) + one pipeline-class specialist (train_specialist.py).

At inference the two models' detections are merged with specialist
priority: a LOCO-model box overlapping (IoU > MERGE_IOU) a specialist box
scoring >= OPERATING_CONF is dropped -- the typical conflict is the LOCO
model mislabeling the new object it has never seen (e.g. calling the yellow
iw_hub robot a pallet_truck). Because that rule can also delete a *correct*
LOCO box whenever the specialist fires falsely on it, the LOCO classes are
scored on the MERGED output, not on the LOCO model alone.

Decision rules:
  - LOCO classes: paired bootstrap over the official LOCO val images (same
    resampled images for "LOCO model alone" and "merged system"); a class
    regresses only if the 95% CI of (merged - alone) AP_50 lies entirely
    below zero. Replaces promote_if_better.py's fixed -0.01 tolerance, which
    was smaller than the run-to-run noise measured on this data.
  - Pipeline classes: first promotion needs AP_50 >= MIN_AP50_FOR_NEW_CLASS
    on the class's held_out_val; later promotions use the same paired
    bootstrap against the current production specialist.
  - False positives: at most MAX_FP_IMAGE_RATE of the LOCO val images (which
    contain no pipeline class) may get any pipeline-class detection scoring
    >= OPERATING_CONF from the specialist.

AP_50 is computed from COCOeval's own per-image matches (evaluate()), then
re-accumulated exactly like COCOeval.accumulate() for each bootstrap
resample -- resampling is done with per-image weights over one fixed sorted
detection list, so 1000 resamples cost seconds, not 1000 COCOeval runs.

Dry-run by default: prints the decision and only writes to
runs/detect/production/ with --promote (the robot side of the two-model
system isn't wired up yet).

Usage:
    python two_model_gate.py --specialist runs/detect/yolo_runs/specialist_v1/weights/best.pt [--promote]
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

import db

sys.path.insert(0, db.DATASET_ROOT)
import eval_loco  # noqa: E402
from promote_if_better import (  # noqa: E402
    AP_CONFIDENCE_THRESHOLD, LOCO_ANNOTATIONS, LOCO_VAL_PATHS, MIN_AP50_FOR_NEW_CLASS,
    PRODUCTION_DIR, PRODUCTION_REGISTRY, PRODUCTION_WEIGHTS, yolo_dir_to_coco_gt,
)

PRODUCTION_SPECIALIST = os.path.join(PRODUCTION_DIR, "specialist.pt")
MERGE_IOU = 0.7
OPERATING_CONF = 0.5  # the robot's own inference threshold
MAX_FP_IMAGE_RATE = 0.01
N_BOOTSTRAP = 1000
CI = 0.95
SEED = 0


# --------------------------------------------------------------------- inference

def predict_raw(weights, image_paths, conf=AP_CONFIDENCE_THRESHOLD, imgsz=960, batch=16):
    """{path: (xyxy[N,4], scores[N], cls[N])}"""
    from ultralytics import YOLO

    model = YOLO(weights)
    out = {}
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        for path, pred in zip(chunk, model.predict(chunk, conf=conf, imgsz=imgsz, device=0, verbose=False)):
            b = pred.boxes
            out[path] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
    return out


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def merge_detections(loco_det, spec_det, pipeline_idx, merge_iou=MERGE_IOU, spec_conf=OPERATING_CONF):
    """Specialist priority: drop LOCO-model boxes overlapping a confident
    specialist box; keep every specialist box of a pipeline class. The
    specialist's head also has the LOCO class slots (global indices) but is
    never trained on them, so anything it emits there is discarded."""
    (lb, ls, lc), (sb, ss, sc) = loco_det, spec_det
    own = np.isin(sc, list(pipeline_idx))
    sb, ss, sc = sb[own], ss[own], sc[own]
    strong = sb[ss >= spec_conf]
    keep = ~(_iou_matrix(lb, strong) > merge_iou).any(axis=1) if len(strong) else np.ones(len(lb), bool)
    return (np.concatenate([lb[keep], sb]), np.concatenate([ls[keep], ss]), np.concatenate([lc[keep], sc]))


def _suppressed(loco_det, merged_det, loco_idx):
    """(classes, scores) of LOCO-model boxes present in loco_det but missing
    from merged_det -- i.e. the ones the merge dropped."""
    lb, ls, lc = loco_det
    mb, _ms, mc = merged_det
    kept = mb[np.isin(mc, list(loco_idx))]
    gone = np.array([not np.any(np.all(np.isclose(kept, box), axis=1)) for box in lb], dtype=bool) \
        if len(lb) else np.zeros(0, bool)
    return lc[gone], ls[gone]


def to_coco_results(dets_by_path, gt, class_idx_to_cat_id):
    id_by_path = {img["path"]: img["id"] for img in gt["images"]}
    results = []
    for path, (xyxy, scores, classes) in dets_by_path.items():
        for (x0, y0, x1, y1), s, c in zip(xyxy, scores, classes):
            if int(c) in class_idx_to_cat_id:
                results.append({"image_id": id_by_path[path], "category_id": class_idx_to_cat_id[int(c)],
                                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "score": float(s)})
    return results


# ------------------------------------------------------- bootstrappable AP_50

class PerImageMatches:
    """COCOeval's per-image matching at IoU=0.5 / area=all / maxDets=100,
    flattened into one score-sorted detection list per category so AP_50 can
    be re-accumulated under any per-image weighting (bootstrap resample)."""

    def __init__(self, gt, results):
        import copy

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(gt)
        coco_gt.createIndex()
        self.img_ids = sorted(img["id"] for img in gt["images"])
        img_index = {img_id: i for i, img_id in enumerate(self.img_ids)}
        self.cat_ids = [c["id"] for c in gt["categories"]]
        self.recall_thrs = np.linspace(0.0, 1.0, 101)
        self.per_cat = {}
        if not results:
            for cat in self.cat_ids:
                npig = np.zeros(len(self.img_ids))
                for a in gt["annotations"]:
                    if a["category_id"] == cat and not a.get("iscrowd", 0):
                        npig[img_index[a["image_id"]]] += 1
                self.per_cat[cat] = (np.zeros(0), np.zeros(0, int), np.zeros(0, bool), np.zeros(0, bool), npig)
            return

        coco_dt = coco_gt.loadRes(copy.deepcopy(results))
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = self.img_ids
        ev.params.catIds = self.cat_ids
        ev.evaluate()
        all_area = ev.params.areaRng[0]
        max_det = ev.params.maxDets[-1]
        by_cat = {cat: {"scores": [], "img": [], "tp": [], "ign": [], "npig": np.zeros(len(self.img_ids))}
                  for cat in self.cat_ids}
        for e in ev.evalImgs:
            if e is None or e["aRng"] != all_area:
                continue
            d = by_cat[e["category_id"]]
            idx = img_index[e["image_id"]]
            d["npig"][idx] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
            n = min(len(e["dtScores"]), max_det)
            if n == 0:
                continue
            d["scores"].append(np.asarray(e["dtScores"][:n]))
            d["img"].append(np.full(n, idx))
            d["tp"].append(np.asarray(e["dtMatches"][0, :n]) > 0)
            d["ign"].append(np.asarray(e["dtIgnore"][0, :n]).astype(bool))
        for cat, d in by_cat.items():
            if d["scores"]:
                scores = np.concatenate(d["scores"])
                order = np.argsort(-scores, kind="mergesort")
                self.per_cat[cat] = (scores[order], np.concatenate(d["img"])[order],
                                     np.concatenate(d["tp"])[order], np.concatenate(d["ign"])[order], d["npig"])
            else:
                self.per_cat[cat] = (np.zeros(0), np.zeros(0, int), np.zeros(0, bool), np.zeros(0, bool), d["npig"])

    def ap50(self, cat, img_weights=None):
        _scores, img, tp, ign, npig_per_img = self.per_cat[cat]
        w_img = np.ones(len(self.img_ids)) if img_weights is None else img_weights
        npig = float(np.dot(w_img, npig_per_img))
        if npig == 0:
            return float("nan")
        if len(img) == 0:
            return 0.0
        w = w_img[img] * (~ign)
        tps = np.cumsum(w * tp)
        fps = np.cumsum(w * ~tp)
        rc = tps / npig
        pr = tps / (tps + fps + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, self.recall_thrs, side="left")
        q = np.where(inds < len(pr), pr[np.minimum(inds, len(pr) - 1)], 0.0)
        return float(q.mean())


def paired_bootstrap(a, b, cat_a, cat_b, n=N_BOOTSTRAP, seed=SEED):
    """a/b: PerImageMatches over the SAME image list. Returns point estimates
    and the CI of AP(b) - AP(a)."""
    rng = np.random.default_rng(seed)
    n_img = len(a.img_ids)
    diffs = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        diffs[i] = b.ap50(cat_b, w) - a.ap50(cat_a, w)
    lo, hi = np.nanpercentile(diffs, [(1 - CI) / 2 * 100, (1 + CI) / 2 * 100])
    return a.ap50(cat_a), b.ap50(cat_b), float(lo), float(hi)


# ------------------------------------------------------------------------- gate

def load_production():
    if not os.path.exists(PRODUCTION_REGISTRY):
        raise SystemExit("no production model yet -- nothing to pair the specialist with")
    with open(PRODUCTION_REGISTRY) as f:
        return json.load(f)


def run_gate(specialist, promote=False):
    registry = db.load_class_registry()
    production = load_production()
    loco_reg = db.loco_classes(registry)
    pipeline_reg = db.pipeline_classes(registry)
    failures = []

    # --- LOCO classes: LOCO model alone vs merged system, same images
    with open(LOCO_VAL_PATHS) as f:
        val_paths = [line.strip() for line in f if line.strip()]
    gt = eval_loco.build_gt_subset(LOCO_ANNOTATIONS, val_paths)
    for img in gt["images"]:
        img["path"] = os.path.join(db.DATASET_ROOT, img["path"].lstrip("/"))
    paths = [img["path"] for img in gt["images"]]
    cat_id = {c["name"]: c["id"] for c in gt["categories"]}
    loco_idx_to_cat = {c["class_idx"]: cat_id[c["name"]] for c in loco_reg}

    print(f"predicting {len(paths)} LOCO val images with the LOCO model and the specialist...")
    loco_det = predict_raw(PRODUCTION_WEIGHTS, paths)
    spec_det = predict_raw(specialist, paths)
    pipeline_idx = {c["class_idx"] for c in pipeline_reg}
    merged = {p: merge_detections(loco_det[p], spec_det[p], pipeline_idx) for p in paths}
    alone = PerImageMatches(gt, to_coco_results(loco_det, gt, loco_idx_to_cat))
    combined = PerImageMatches(gt, to_coco_results(merged, gt, loco_idx_to_cat))

    suppressed = [(os.path.basename(p), int(c), float(s))
                  for p in paths
                  for c, s in zip(*_suppressed(loco_det[p], merged[p], set(loco_idx_to_cat)))]
    visible = [x for x in suppressed if x[2] >= OPERATING_CONF]
    print(f"\nLOCO-model boxes removed by the merge: {len(suppressed)} at any score, "
          f"{len(visible)} at >= {OPERATING_CONF} (the ones the robot would actually have shown): {visible}")

    print(f"\n=== LOCO classes: merged system vs LOCO model alone (paired bootstrap, {CI:.0%} CI) ===")
    report = {"loco": {}, "suppressed_visible": visible, "suppressed_total": len(suppressed)}
    for c in loco_reg:
        a, b, lo, hi = paired_bootstrap(alone, combined, cat_id[c["name"]], cat_id[c["name"]])
        regressed = hi < 0
        report["loco"][c["name"]] = {"alone": a, "merged": b, "ci_diff": [lo, hi], "regressed": regressed}
        print(f"{c['name']:<20} alone={a:.4f} merged={b:.4f} diff CI=[{lo:+.4f}, {hi:+.4f}]"
              f"{'  REGRESSED' if regressed else ''}")
        if regressed:
            failures.append(f"{c['name']}: merged system significantly worse than LOCO model alone "
                            f"(diff CI [{lo:+.3f}, {hi:+.3f}])")

    # --- false positives of the specialist on LOCO val (no pipeline class present)
    n_fp = sum(1 for p in paths
               if np.any((spec_det[p][1] >= OPERATING_CONF) & np.isin(spec_det[p][2], list(pipeline_idx))))
    fp_rate = n_fp / len(paths)
    report["fp_image_rate"] = fp_rate
    print(f"\n=== specialist false positives on LOCO val ===\n{n_fp}/{len(paths)} images "
          f"({fp_rate:.2%}) with a pipeline-class detection >= {OPERATING_CONF} (max {MAX_FP_IMAGE_RATE:.0%})")
    if fp_rate > MAX_FP_IMAGE_RATE:
        failures.append(f"false positives on {fp_rate:.2%} of LOCO val images (max {MAX_FP_IMAGE_RATE:.0%})")

    # --- pipeline classes on their own held-out sets
    current_spec = production.get("specialist")
    has_current = current_spec is not None and os.path.exists(PRODUCTION_SPECIALIST)
    print("\n=== pipeline classes (held_out_val) ===")
    report["pipeline"] = {}
    for c in pipeline_reg:
        held_out = os.path.join(db.PROJECT_ROOT, c["held_out_val"])
        hgt = yolo_dir_to_coco_gt(os.path.join(held_out, "images"), os.path.join(held_out, "labels"),
                                  category_id=0, category_name=c["name"])
        hpaths = [img["path"] for img in hgt["images"]]
        cand = PerImageMatches(hgt, to_coco_results(predict_raw(specialist, hpaths), hgt, {c["class_idx"]: 0}))
        if has_current and c["name"] in current_spec.get("classes", []):
            cur = PerImageMatches(hgt, to_coco_results(predict_raw(PRODUCTION_SPECIALIST, hpaths), hgt,
                                                       {c["class_idx"]: 0}))
            a, b, lo, hi = paired_bootstrap(cur, cand, 0, 0)
            regressed = hi < 0
            print(f"{c['name']:<20} current={a:.3f} candidate={b:.3f} diff CI=[{lo:+.3f}, {hi:+.3f}]"
                  f"{'  REGRESSED' if regressed else ''}")
            report["pipeline"][c["name"]] = {"current": a, "candidate": b, "ci_diff": [lo, hi]}
            if regressed:
                failures.append(f"{c['name']}: candidate specialist significantly worse than current one")
        else:
            ap = cand.ap50(0)
            print(f"{c['name']:<20} candidate={ap:.3f} (first promotion, floor {MIN_AP50_FOR_NEW_CLASS})")
            report["pipeline"][c["name"]] = {"candidate": ap}
            if ap < MIN_AP50_FOR_NEW_CLASS:
                failures.append(f"{c['name']}: AP_50={ap:.3f} below first-promotion floor {MIN_AP50_FOR_NEW_CLASS}")

    passed = not failures
    print("\nPASSED" if passed else "\nNOT PROMOTED:\n" + "\n".join(f" - {f}" for f in failures))
    if passed and promote:
        shutil.copy2(specialist, PRODUCTION_SPECIALIST)
        production["specialist"] = {
            "weights": specialist, "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "classes": [c["name"] for c in pipeline_reg], "report": report,
        }
        with open(PRODUCTION_REGISTRY, "w") as f:
            json.dump(production, f, indent=2)
        print(f"PROMOTED specialist -> {PRODUCTION_SPECIALIST}")
    elif passed:
        print("(dry run -- pass --promote to write it to production)")
    return passed, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()
    _passed, report = run_gate(args.specialist, args.promote)
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
