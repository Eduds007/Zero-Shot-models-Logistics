"""
Decides whether a freshly fine-tuned candidate model should replace the
current production model, generalized over every class the CURRENT
production model already knows (read dynamically from
runs/detect/production/registry.json -- not a hardcoded "5 LOCO + iw_hub"
list). Promoting a candidate that regresses ANY known class -- even one
this pipeline only recently taught it -- defeats the point of the pipeline,
so every one of those classes is checked, not just the newest.

Promotion rule: for every class known to current production, the
candidate's AP_50 must be >= production's AP_50 - REGRESSION_TOLERANCE.
AP_50 here is the same 0.0-1.0 fraction eval_loco.py has always reported
(e.g. "AP_50 OVERALL 0.257" in this repo's own history) -- NOT a 0-100
percentage -- so the default tolerance of -0.01 means "at most 1
percentage point worse," absorbing run-to-run measurement noise without
allowing real forgetting. A class with no current production baseline yet
(its first promotion) only needs to clear MIN_AP50_FOR_NEW_CLASS.

Reuses the exact COCOeval pipeline every prior experiment in this repo used
(eval_loco.py's build_gt_subset/_run_coco_eval/evaluate) and the YOLO ->
COCO-detections pattern from eval_robot_finetune_experiments.py's
predict_on_loco/predict_iwhub, generalized to an arbitrary class subset.

Needs, alongside the candidate weights:
  - dataset/loco_val_paths.txt + dataset/annotations/loco-all-v1.json (the
    fixed LOCO held-out val split every prior experiment used)
  - for each source=pipeline class in class_registry.json, a YOLO-format
    <held_out_val>/{images,labels}/ folder that build_training_batch.py's
    replay sampling never touches -- a permanently held-out slice (e.g.
    carved out of the iw_hub Roboflow export). Populated once by hand; this
    script only reads it.

Usage:
    python promote_if_better.py --run-name loco_finetune_pipeline_v3 \
        --weights ../../runs/detect/yolo_runs/loco_finetune_pipeline_v3/weights/best.pt
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

REGRESSION_TOLERANCE = -0.01  # AP_50 fraction (0.0-1.0 scale); candidate may be at most 1 point worse
MIN_AP50_FOR_NEW_CLASS = 0.10

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
    next_ann_id = 0
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


def predict_yolo(weights_path, gt, class_idx_to_cat_id, conf=0.5, imgsz=960, device=0, batch=32):
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
                      regression_tolerance=REGRESSION_TOLERANCE,
                      min_ap50_new_class=MIN_AP50_FOR_NEW_CLASS):
    """Pure decision logic, kept separate from I/O/inference so it can be
    unit-tested without a GPU. Returns (promote: bool, failures: list[str]).

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
            if cand < prod + regression_tolerance:
                failures.append(
                    f"{name}: candidate AP_50={cand:.3f} < production AP_50={prod:.3f} "
                    f"- tolerance {regression_tolerance} (would regress this class)"
                )
        elif name in pipeline_class_names and cand < min_ap50_new_class:
            failures.append(
                f"{name}: candidate AP_50={cand:.3f} below floor {min_ap50_new_class} "
                f"required for a first-time promotion"
            )
    return (len(failures) == 0), failures


def evaluate_model(weights_path, registry, target_classes, dataset_root):
    """Runs `weights_path` against every target class's val set, returning
    {class_name: AP_50}. Skips (with a warning) any pipeline class whose
    held_out_val folder doesn't exist yet."""
    loco_reg = {c["name"]: c for c in db.loco_classes(registry)}
    pipeline_reg = {c["name"]: c for c in db.pipeline_classes(registry)}
    scores = {}

    loco_names = [n for n in target_classes if n in loco_reg]
    if loco_names:
        with open(LOCO_VAL_PATHS) as f:
            val_paths = [line.strip() for line in f if line.strip()]
        gt = eval_loco.build_gt_subset(LOCO_ANNOTATIONS, val_paths)
        for img in gt["images"]:
            img["path"] = os.path.join(dataset_root, img["path"].lstrip("/"))
        cat_name_to_id = {c["name"]: c["id"] for c in gt["categories"]}
        class_idx_to_cat_id = {loco_reg[n]["class_idx"]: cat_name_to_id[n]
                                for n in loco_names if n in cat_name_to_id}
        results, _elapsed = predict_yolo(weights_path, gt, class_idx_to_cat_id)
        ap50 = ap50_per_category(gt, results)
        scores.update({n: ap50.get(cat_name_to_id[n], 0.0) for n in loco_names if n in cat_name_to_id})

    for name in target_classes:
        if name not in pipeline_reg:
            continue
        cls = pipeline_reg[name]
        held_out = os.path.join(db.PROJECT_ROOT, cls["held_out_val"])
        images_dir, labels_dir = os.path.join(held_out, "images"), os.path.join(held_out, "labels")
        if not os.path.isdir(images_dir):
            print(f"WARNING: no held-out val set at {held_out} for class '{name}' -- "
                  f"can't verify it isn't regressing (populate this folder to close the gap)")
            continue
        gt = yolo_dir_to_coco_gt(images_dir, labels_dir, category_id=0, category_name=name)
        results, _elapsed = predict_yolo(weights_path, gt, {cls["class_idx"]: 0})
        scores[name] = ap50_per_category(gt, results).get(0, 0.0)

    return scores


def promote(run_name, weights, regression_tolerance=REGRESSION_TOLERANCE,
            min_ap50_new_class=MIN_AP50_FOR_NEW_CLASS):
    """Evaluates `weights` against every class the current production model
    knows, and promotes it to production iff none of them regress. Returns
    True if promoted, False otherwise -- used directly by
    auto_train_trigger.py, and by main() below for the CLI."""
    registry = db.load_class_registry()
    production = load_production_registry()
    has_production = production is not None and os.path.exists(PRODUCTION_WEIGHTS)
    target_classes = known_classes(registry, production)

    print(f"classes the gate must not regress: {target_classes}")
    print(f"current production: {'none yet (first promotion)' if not has_production else PRODUCTION_WEIGHTS}")

    candidate_scores = evaluate_model(weights, registry, target_classes, db.DATASET_ROOT)
    # Only evaluate production on classes ITS OWN registry claims to know --
    # evaluating it on a class it never learned (e.g. iw_hub before its first
    # promotion) always scores 0.0 (the model structurally can't predict a
    # class index it has no head for), and a candidate clearing that phantom
    # 0.0 baseline would trivially "pass" the regression check in
    # decide_promotion() instead of being held to MIN_AP50_FOR_NEW_CLASS via
    # its proper first-promotion floor check.
    production_known = set(production["classes"]) if has_production else set()
    production_scores = (
        evaluate_model(PRODUCTION_WEIGHTS, registry, [c for c in target_classes if c in production_known],
                       db.DATASET_ROOT)
        if has_production else {}
    )

    print("\n=== per-class AP_50 ===")
    for name in target_classes:
        cand = candidate_scores.get(name)
        prod = production_scores.get(name)
        print(f"{name}: candidate={cand if cand is not None else float('nan'):.3f} "
              f"production={prod if prod is not None else float('nan'):.3f}")

    pipeline_class_names = {c["name"] for c in db.pipeline_classes(registry)}
    should_promote, failures = decide_promotion(
        target_classes, candidate_scores, production_scores, pipeline_class_names,
        regression_tolerance, min_ap50_new_class,
    )

    if not should_promote:
        print("\nNOT PROMOTED:")
        for f in failures:
            print(f" - {f}")
        return False

    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    shutil.copy2(weights, PRODUCTION_WEIGHTS)
    with open(PRODUCTION_REGISTRY, "w") as f:
        json.dump({
            "run_name": run_name,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "classes": target_classes,
            "ap50_by_class": candidate_scores,
        }, f, indent=2)
    print(f"\nPROMOTED: {run_name} -> {PRODUCTION_WEIGHTS}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--weights", required=True, help="Path to the candidate's best.pt")
    parser.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    parser.add_argument("--min-ap50-new-class", type=float, default=MIN_AP50_FOR_NEW_CLASS)
    args = parser.parse_args()
    promote(args.run_name, args.weights, args.regression_tolerance, args.min_ap50_new_class)


if __name__ == "__main__":
    main()
