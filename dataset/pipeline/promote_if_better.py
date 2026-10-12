"""
Decides whether a freshly fine-tuned candidate model should replace the
current production model, generalized over every class the CURRENT
production model already knows (read dynamically from
runs/detect/production/registry.json -- not a hardcoded "5 LOCO + iw_hub"
list). Promoting a candidate that regresses ANY known class -- even one
this pipeline only recently taught it -- defeats the point of the pipeline,
so every one of those classes is checked, not just the newest.

Promotion rule: for every class known to current production, the candidate
is rejected only if it is *demonstrably* worse -- the 95% confidence
interval of AP_50(candidate) - AP_50(production), from a paired bootstrap
over the same test images (two_model_gate.paired_bootstrap), lies entirely
below REGRESSION_TOLERANCE. A plain point-estimate threshold of -0.01 is
smaller than the run-to-run noise measured between training seeds (up to
~0.05 AP_50 on the weak classes), so it rejected candidates on noise.
AP_50 here is the same 0.0-1.0 fraction eval_loco.py has always reported
(e.g. "AP_50 OVERALL 0.257" in this repo's own history) -- NOT a 0-100
percentage. A class with no current production baseline yet (its first
promotion) only needs to clear MIN_AP50_FOR_NEW_CLASS.

Reuses the exact COCOeval pipeline every prior experiment in this repo used
(eval_loco.py's build_gt_subset/_run_coco_eval/evaluate) and the YOLO ->
COCO-detections pattern from eval_robot_finetune_experiments.py's
predict_on_loco/predict_iwhub, generalized to an arbitrary class subset.

Needs, alongside the candidate weights:
  - dataset/loco_val_paths.txt + dataset/annotations/loco-all-v1.json (the
    fixed LOCO held-out val split every prior experiment used)
  - for each source=pipeline class in class_registry.json, one or more
    YOLO-format {images,labels}/ folders that training never touches:
    "test_sets" (e.g. fixed_testset/iw_hub + fixed_testset/iw_hub_roboflow,
    real files), falling back to "held_out_val". The old held_out_val/iw_hub
    was symlinks into /tmp and lost on a reboot.

Usage:
    python promote_if_better.py --run-name loco_finetune_pipeline_v3 \
        --weights ../../runs/detect/yolo_runs/loco_finetune_pipeline_v3/weights/best.pt [--dry-run]
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time

import db

sys.path.insert(0, db.DATASET_ROOT)
import eval_loco  # noqa: E402

REGRESSION_TOLERANCE = -0.01  # AP_50 fraction (0.0-1.0 scale); reject only if the whole 95% CI of the drop is below it
MIN_AP50_FOR_NEW_CLASS = 0.10

# AP_50 is meant to be computed by sweeping the whole precision-recall curve
# (standard COCOeval practice uses conf~0.001 for exactly this reason) --
# using conf=0.5 as a pre-filter before detections even reach COCOeval
# silently discards real, lower-confidence true positives and understates
# every class's AP_50, disproportionately so for weaker/rarer classes whose
# correct detections skew lower-confidence to begin with (forklift's AP_50
# on the official LOCO val jumped from 0.144 to 0.183 once this was fixed,
# pallet_truck's nearly tripled from 0.077 to 0.206). conf is still exposed
# as a predict_yolo() parameter for callers that genuinely want a filtered
# detection list (e.g. live inference), just no longer defaulted that way
# for AP computation.
AP_CONFIDENCE_THRESHOLD = 0.001

PRODUCTION_DIR = os.path.join(db.PROJECT_ROOT, "runs", "detect", "production")
PRODUCTION_WEIGHTS = os.path.join(PRODUCTION_DIR, "best.pt")
PRODUCTION_REGISTRY = os.path.join(PRODUCTION_DIR, "registry.json")

LOCO_ANNOTATIONS = os.path.join(db.DATASET_ROOT, "annotations", "loco-all-v1.json")
LOCO_VAL_PATHS = os.path.join(db.DATASET_ROOT, "loco_val_paths.txt")
LOCO_RESULTS_CSV = os.path.join(db.DATASET_ROOT, "results", "loco_eval_results.csv")


def load_production_registry():
    if not os.path.exists(PRODUCTION_REGISTRY):
        return None
    with open(PRODUCTION_REGISTRY) as f:
        return json.load(f)


def known_classes(registry, production):
    """Classes the gate must evaluate: whatever production currently knows
    (or the 5 LOCO classes as a floor if there's no production model
    promoted yet), UNION every source=pipeline class in class_registry.json.

    Without the union, a brand-new pipeline class (e.g. iw_hub before its
    first successful promotion) is absent from production['classes'] and so
    never gets evaluated against its fixed held-out set by this gate at
    all -- its AP_50 only ever came from run_finetune.py's own internal
    training-batch validation split, which isn't guaranteed to be the same
    images across runs. Appending it here means it's checked against
    MIN_AP50_FOR_NEW_CLASS (via decide_promotion's pipeline_class_names
    branch) starting with its very first promotion attempt."""
    if production and production.get("classes"):
        classes = list(production["classes"])
    else:
        classes = [c["name"] for c in db.loco_classes(registry)]
    for c in db.pipeline_classes(registry):
        if c["name"] not in classes:
            classes.append(c["name"])
    return classes


def yolo_dir_to_coco_gt(images_dir, labels_dir, category_id, category_name):
    """Builds a single-category COCO-format GT dict straight from a YOLO
    images/+labels/ folder pair -- lets any future pipeline class use the
    same eval path as LOCO without a class-specific annotation converter."""
    import cv2

    images, annotations = [], []
    # Must start at 1: COCOeval stores a detection's matched gt *id* in
    # dtMatches and treats 0 as "unmatched", so an annotation with id 0 can
    # never count as a true positive.
    next_ann_id = 1
    for i, fname in enumerate(sorted(os.listdir(images_dir))):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({"id": i, "path": path, "width": w, "height": h})
        lbl_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                _cls, cx, cy, bw, bh = (float(x) for x in parts)
                x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
                annotations.append({
                    "id": next_ann_id, "image_id": i, "category_id": category_id,
                    "bbox": [x0, y0, bw * w, bh * h], "area": bw * w * bh * h, "iscrowd": 0,
                })
                next_ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": category_id, "name": category_name}]}


def predict_yolo(weights_path, gt, class_idx_to_cat_id, conf=AP_CONFIDENCE_THRESHOLD, imgsz=960, device=0, batch=32):
    """gt['images'] entries need an already-resolved 'path'. Keeps only
    detections whose class index is in class_idx_to_cat_id -- e.g. an
    iw_hub-aware model's other 5 heads are irrelevant when scoring against
    an iw_hub-only held-out set, and vice versa."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    image_paths = [img["path"] for img in gt["images"]]
    id_by_path = {img["path"]: img["id"] for img in gt["images"]}

    results = []
    start = time.perf_counter()
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        preds = model.predict(chunk, conf=conf, imgsz=imgsz, device=device, verbose=False)
        for path, pred in zip(chunk, preds):
            image_id = id_by_path[path]
            boxes = pred.boxes
            for xyxy, score, cls in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
            ):
                cls = int(cls)
                if cls not in class_idx_to_cat_id:
                    continue
                x0, y0, x1, y1 = xyxy
                results.append({
                    "image_id": image_id, "category_id": class_idx_to_cat_id[cls],
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(score),
                })
    return results, time.perf_counter() - start


def ap50_per_category(gt, results):
    """{category_id: AP_50}, via the same COCOeval call eval_loco.evaluate()
    uses internally -- so these numbers match anything already logged in
    results/loco_eval_results.csv."""
    from pycocotools.coco import COCO

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    if not results:
        return {c["id"]: 0.0 for c in gt["categories"]}
    coco_dt = coco_gt.loadRes(copy.deepcopy(results))
    return {
        c["id"]: eval_loco._run_coco_eval(coco_gt, coco_dt, cat_ids=[c["id"]])[1]
        for c in gt["categories"]
    }


def decide_promotion(target_classes, candidate_scores, production_scores, pipeline_class_names,
                      diff_ci, regression_tolerance=REGRESSION_TOLERANCE,
                      min_ap50_new_class=MIN_AP50_FOR_NEW_CLASS):
    """Pure decision logic, kept separate from I/O/inference so it can be
    unit-tested without a GPU. Returns (promote: bool, failures: list[str]).

    diff_ci: {class_name: (lo, hi)}, the 95% CI of candidate - production
    AP_50 for every class production knows. A class fails only when hi <
    regression_tolerance, i.e. the candidate is worse by more than the
    tolerance with 95% confidence.

    The min_ap50_new_class floor only applies to classes this PIPELINE is
    introducing (source=pipeline in class_registry.json, e.g. iw_hub) the
    first time they're promoted -- not to the original LOCO classes on the
    very first bootstrap promotion (no production model exists yet). LOCO's
    own baseline AP_50 (e.g. a historically weak class like pallet_truck)
    isn't something this gate should reject; it's just accepted as the
    starting reference for every future round to be compared against."""
    failures = []
    for name in target_classes:
        cand = candidate_scores.get(name)
        if cand is None:
            failures.append(f"{name}: could not be evaluated (missing/unreadable val data)")
            continue
        if name in production_scores:
            prod = production_scores[name]
            lo, hi = diff_ci[name]
            if hi < regression_tolerance:
                failures.append(
                    f"{name}: candidate AP_50={cand:.3f} vs production {prod:.3f}, 95% CI of the "
                    f"difference [{lo:+.3f}, {hi:+.3f}] entirely below {regression_tolerance} "
                    f"(demonstrably regresses this class)"
                )
        elif name in pipeline_class_names and cand < min_ap50_new_class:
            failures.append(
                f"{name}: candidate AP_50={cand:.3f} below floor {min_ap50_new_class} "
                f"required for a first-time promotion"
            )
    return (len(failures) == 0), failures


def _eval_sets(registry, target_classes, dataset_root):
    """Yields (gt, {class_name: category_id}, {class_idx: category_id}) for
    every val set the target classes live in: the LOCO val split for the
    LOCO classes, and each pipeline class's own test folders (see
    pipeline_class_gt). Skips (with a warning) a pipeline class with none."""
    loco_reg = {c["name"]: c for c in db.loco_classes(registry)}
    pipeline_reg = {c["name"]: c for c in db.pipeline_classes(registry)}

    loco_names = [n for n in target_classes if n in loco_reg]
    if loco_names:
        with open(LOCO_VAL_PATHS) as f:
            val_paths = [line.strip() for line in f if line.strip()]
        gt = eval_loco.build_gt_subset(LOCO_ANNOTATIONS, val_paths)
        for img in gt["images"]:
            img["path"] = os.path.join(dataset_root, img["path"].lstrip("/"))
        cat_name_to_id = {c["name"]: c["id"] for c in gt["categories"]}
        names = {n: cat_name_to_id[n] for n in loco_names if n in cat_name_to_id}
        yield gt, names, {loco_reg[n]["class_idx"]: cat for n, cat in names.items()}

    for name in target_classes:
        if name not in pipeline_reg:
            continue
        gt = pipeline_class_gt(pipeline_reg[name])
        if gt is not None:
            yield gt, {name: 0}, {pipeline_reg[name]["class_idx"]: 0}


def pipeline_class_gt(cls, category_id=0):
    """Single-category COCO GT over all of a pipeline class's test folders
    ("test_sets", else "held_out_val"), or None (with a warning) if none of
    them holds a readable image."""
    dirs = cls.get("test_sets") or [cls["held_out_val"]]
    merged = {"images": [], "annotations": [], "categories": [{"id": category_id, "name": cls["name"]}]}
    for rel in dirs:
        d = os.path.join(db.PROJECT_ROOT, rel)
        if not os.path.isdir(os.path.join(d, "images")):
            print(f"WARNING: no test set at {d} for class '{cls['name']}'")
            continue
        gt = yolo_dir_to_coco_gt(os.path.join(d, "images"), os.path.join(d, "labels"), category_id, cls["name"])
        img_offset, ann_offset = len(merged["images"]), len(merged["annotations"])
        for img in gt["images"]:
            merged["images"].append({**img, "id": img["id"] + img_offset})
        for ann in gt["annotations"]:
            merged["annotations"].append({**ann, "id": ann["id"] + ann_offset, "image_id": ann["image_id"] + img_offset})
    if not merged["images"]:
        print(f"WARNING: no readable test images for class '{cls['name']}' in {dirs} -- "
              f"can't verify it isn't regressing")
        return None
    return merged


def evaluate_model(weights_path, registry, target_classes, dataset_root):
    """Runs `weights_path` against every target class's val set, returning
    {class_name: AP_50}."""
    scores = {}
    for gt, names, idx_to_cat in _eval_sets(registry, target_classes, dataset_root):
        results, _elapsed = predict_yolo(weights_path, gt, idx_to_cat)
        ap50 = ap50_per_category(gt, results)
        scores.update({n: ap50.get(cat, 0.0) for n, cat in names.items()})
    return scores


def compare_with_production(candidate, production, registry, target_classes, production_known, dataset_root):
    """Scores the candidate on every target class and, for the classes
    production knows, production too -- on the same images -- plus the 95%
    paired-bootstrap CI of candidate - production. Returns (candidate_scores,
    production_scores, diff_ci)."""
    from two_model_gate import PerImageMatches, paired_bootstrap

    cand_scores, prod_scores, diff_ci = {}, {}, {}
    for gt, names, idx_to_cat in _eval_sets(registry, target_classes, dataset_root):
        cand_results, _ = predict_yolo(candidate, gt, idx_to_cat)
        cand_ap = ap50_per_category(gt, cand_results)
        cand_scores.update({n: cand_ap.get(cat, 0.0) for n, cat in names.items()})
        known = {n: cat for n, cat in names.items() if n in production_known}
        if not known:
            continue
        prod_results, _ = predict_yolo(production, gt, idx_to_cat)
        prod_ap = ap50_per_category(gt, prod_results)
        cand_m, prod_m = PerImageMatches(gt, cand_results), PerImageMatches(gt, prod_results)
        for n, cat in known.items():
            prod_scores[n] = prod_ap.get(cat, 0.0)
            _p, _c, lo, hi = paired_bootstrap(prod_m, cand_m, cat, cat)
            diff_ci[n] = (lo, hi)
    return cand_scores, prod_scores, diff_ci


def _gate_one(weights, registry, target_classes, production_known, has_production,
              regression_tolerance, min_ap50_new_class):
    """Scores one candidate and runs decide_promotion on it."""
    # Only evaluate production on classes ITS OWN registry claims to know --
    # evaluating it on a class it never learned (e.g. iw_hub before its first
    # promotion) always scores 0.0 (the model structurally can't predict a
    # class index it has no head for), and a candidate clearing that phantom
    # 0.0 baseline would trivially "pass" the regression check in
    # decide_promotion() instead of being held to MIN_AP50_FOR_NEW_CLASS via
    # its proper first-promotion floor check.
    if has_production:
        candidate_scores, production_scores, diff_ci = compare_with_production(
            weights, PRODUCTION_WEIGHTS, registry, target_classes, production_known, db.DATASET_ROOT)
    else:
        candidate_scores = evaluate_model(weights, registry, target_classes, db.DATASET_ROOT)
        production_scores, diff_ci = {}, {}

    print(f"\n=== {weights}: per-class AP_50 (95% CI of candidate - production) ===")
    for name in target_classes:
        cand = candidate_scores.get(name)
        prod = production_scores.get(name)
        ci = f"  CI [{diff_ci[name][0]:+.3f}, {diff_ci[name][1]:+.3f}]" if name in diff_ci else ""
        print(f"{name}: candidate={cand if cand is not None else float('nan'):.3f} "
              f"production={prod if prod is not None else float('nan'):.3f}{ci}")

    pipeline_class_names = {c["name"] for c in db.pipeline_classes(registry)}
    passed, failures = decide_promotion(
        target_classes, candidate_scores, production_scores, pipeline_class_names,
        diff_ci, regression_tolerance, min_ap50_new_class,
    )
    if not passed:
        print("fails the gate:")
        for f in failures:
            print(f" - {f}")
    return {"weights": weights, "passed": passed, "failures": failures,
            "scores": candidate_scores, "diff_ci": diff_ci}


def promote(run_name, weights, regression_tolerance=REGRESSION_TOLERANCE,
            min_ap50_new_class=MIN_AP50_FOR_NEW_CLASS, dry_run=False):
    """Gates one candidate, or several (e.g. the same batch trained with
    different seeds), against every class the current production model
    knows, and promotes the passing candidate with the highest mean AP_50
    over those classes. Returns True if one was promoted (or would be, with
    dry_run) -- used directly by auto_train_trigger.py, and by main() below
    for the CLI.

    Picking the best of several seeds on the same test set the gate reads
    biases the promoted model's scores slightly upward (a lucky seed looks
    better than the recipe is); each candidate still has to pass the gate
    on its own."""
    candidates = [weights] if isinstance(weights, str) else list(weights)
    registry = db.load_class_registry()
    production = load_production_registry()
    has_production = production is not None and os.path.exists(PRODUCTION_WEIGHTS)
    target_classes = known_classes(registry, production)
    production_known = set(production["classes"]) if has_production else set()

    print(f"classes the gate must not regress: {target_classes}")
    print(f"current production: {'none yet (first promotion)' if not has_production else PRODUCTION_WEIGHTS}")

    results = [_gate_one(w, registry, target_classes, production_known, has_production,
                         regression_tolerance, min_ap50_new_class) for w in candidates]
    passing = [r for r in results if r["passed"]]
    if not passing:
        print(f"\nNOT PROMOTED: none of the {len(results)} candidate(s) passed the gate")
        return False
    best = max(passing, key=lambda r: sum(r["scores"].get(n, 0.0) for n in target_classes) / len(target_classes))
    if dry_run:
        print(f"\nWOULD PROMOTE (dry run, production untouched): {run_name} -> {best['weights']}")
        return True

    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    shutil.copy2(best["weights"], PRODUCTION_WEIGHTS)
    with open(PRODUCTION_REGISTRY, "w") as f:
        json.dump({
            "run_name": run_name,
            "weights": best["weights"],
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "classes": target_classes,
            "ap50_by_class": best["scores"],
            "ap50_diff_ci95_vs_previous": {n: list(ci) for n, ci in best["diff_ci"].items()},
            "candidates": [{"weights": r["weights"], "passed": r["passed"], "ap50_by_class": r["scores"]}
                           for r in results],
        }, f, indent=2)
    print(f"\nPROMOTED: {run_name} ({best['weights']}) -> {PRODUCTION_WEIGHTS}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--weights", required=True, nargs="+",
                        help="Candidate weights; several (e.g. one per seed) -> the best passing one is promoted")
    parser.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    parser.add_argument("--min-ap50-new-class", type=float, default=MIN_AP50_FOR_NEW_CLASS)
    parser.add_argument("--dry-run", action="store_true", help="evaluate and decide, but never touch production")
    args = parser.parse_args()
    promote(args.run_name, args.weights, args.regression_tolerance, args.min_ap50_new_class, args.dry_run)


if __name__ == "__main__":
    main()
