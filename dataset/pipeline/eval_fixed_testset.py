"""
Evaluates one or two models (an "old" one and/or a "new" candidate) against
the fixed test set built by build_fixed_testset.py: a stratified sample of
the official LOCO val split, a permanent slice of the robot's own real
lab-video iw_hub captures, and optionally a fixed sample of the Roboflow
held_out_val/iw_hub set (--add-roboflow).

The two iw_hub sources are scored in SEPARATE predict+COCOeval passes, not
merged into one multi-source category: merging them under one category id
would force every class-5 detection to be scored against combined GT
regardless of which source image it came from, hiding the in-domain
(real lab video) vs. cross-domain (Roboflow) gap that's the whole point of
keeping both sources around. LOCO's 5 classes get one shared pass since
their category ids are already disjoint and unambiguous.

Reuses yolo_dir_to_coco_gt/predict_yolo/ap50_per_category from
promote_if_better.py rather than reimplementing COCO-merging logic.

Usage:
    python eval_fixed_testset.py --old-weights runs/detect/production/best.pt \
        --new-weights runs/detect/yolo_runs/pipeline_finetune_v9/weights/best.pt
"""

import argparse
import json
import os
import sys

import db

sys.path.insert(0, db.PIPELINE_ROOT)
sys.path.insert(0, db.DATASET_ROOT)
import eval_loco  # noqa: E402
from promote_if_better import ap50_per_category, predict_yolo, yolo_dir_to_coco_gt  # noqa: E402

FIXED_TESTSET_DIR = os.path.join(db.PIPELINE_ROOT, "fixed_testset")
LOCO_PATHS_FILE = os.path.join(db.PIPELINE_ROOT, "fixed_testset_loco_paths.txt")
LOCO_ANNOTATIONS = os.path.join(db.DATASET_ROOT, "annotations", "loco-all-v1.json")

IW_HUB_CATEGORY_ID = 100  # outside LOCO's real category id range (3,5,7,10,11)


def load_gt_sources():
    """Returns {source_name: gt_dict}. 'loco' is always present; 'iw_hub'
    (real video) and 'iw_hub_roboflow' are present iff their directories
    exist (the latter is optional, added on demand via --add-roboflow)."""
    if not os.path.exists(LOCO_PATHS_FILE):
        raise SystemExit(f"{LOCO_PATHS_FILE} not found -- run build_fixed_testset.py first")

    with open(LOCO_PATHS_FILE) as f:
        loco_paths = [line.strip() for line in f if line.strip()]
    gt_loco = eval_loco.build_gt_subset(LOCO_ANNOTATIONS, loco_paths)
    for img in gt_loco["images"]:
        img["path"] = os.path.join(db.DATASET_ROOT, img["path"].lstrip("/"))
    sources = {"loco": gt_loco}

    for source_name, subdir in (("iw_hub", "iw_hub"), ("iw_hub_roboflow", "iw_hub_roboflow")):
        d = os.path.join(FIXED_TESTSET_DIR, subdir)
        if not os.path.isdir(os.path.join(d, "images")):
            continue
        sources[source_name] = yolo_dir_to_coco_gt(
            os.path.join(d, "images"), os.path.join(d, "labels"),
            category_id=IW_HUB_CATEGORY_ID, category_name="iw_hub",
        )

    print("fixed test set composition:")
    for name, gt in sources.items():
        print(f"  {name}: {len(gt['images'])} images")
    return sources


def evaluate_model(weights_path, sources, registry, batch=32):
    from ultralytics import YOLO

    model_nc = YOLO(weights_path).model.nc
    scores = {}

    loco_cat_name_to_id = {c["name"]: c["id"] for c in sources["loco"]["categories"]}
    loco_idx_to_cat = {c["class_idx"]: loco_cat_name_to_id[c["name"]]
                       for c in registry if c["source"] == "loco" and c["class_idx"] < model_nc}
    if loco_idx_to_cat:
        results, _ = predict_yolo(weights_path, sources["loco"], loco_idx_to_cat, batch=batch)
        ap50 = ap50_per_category(sources["loco"], results)
        cat_id_to_name = {v: k for k, v in loco_cat_name_to_id.items()}
        scores.update({cat_id_to_name[cat_id]: ap50.get(cat_id, 0.0) for cat_id in loco_idx_to_cat.values()})

    iw_hub_cls = db.class_by_name("iw_hub", registry)
    if iw_hub_cls["class_idx"] < model_nc:
        for source_name in ("iw_hub", "iw_hub_roboflow"):
            if source_name not in sources:
                continue
            results, _ = predict_yolo(weights_path, sources[source_name],
                                       {iw_hub_cls["class_idx"]: IW_HUB_CATEGORY_ID}, batch=batch)
            ap50 = ap50_per_category(sources[source_name], results)
            scores[source_name] = ap50.get(IW_HUB_CATEGORY_ID, 0.0)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-weights", default=None, help="e.g. runs/detect/production/best.pt")
    parser.add_argument("--new-weights", default=None, help="candidate checkpoint, e.g. a fresh fine-tune's best.pt")
    parser.add_argument("--batch", type=int, default=32, help="Inference batch size (lower if the GPU is shared/busy)")
    args = parser.parse_args()
    if not args.old_weights and not args.new_weights:
        raise SystemExit("pass at least one of --old-weights / --new-weights")

    registry = db.load_class_registry()
    sources = load_gt_sources()

    old_scores = evaluate_model(args.old_weights, sources, registry, args.batch) if args.old_weights else {}
    new_scores = evaluate_model(args.new_weights, sources, registry, args.batch) if args.new_weights else {}

    row_names = [c["name"] for c in registry if c["source"] == "loco"]
    row_names += [n for n in ("iw_hub", "iw_hub_roboflow") if n in sources]

    print("\n=== AP_50 on the fixed test set ===")
    print(f"{'classe':<24}{'antigo (sem iw_hub)':<22}{'novo (com iw_hub)':<22}")
    for name in row_names:
        old_val = f"{old_scores[name]:.3f}" if name in old_scores else "N/A (não conhece)"
        new_val = f"{new_scores[name]:.3f}" if name in new_scores else "N/A (não conhece)"
        label = name if name != "iw_hub" else "iw_hub (video real)"
        label = label if name != "iw_hub_roboflow" else "iw_hub (roboflow)"
        print(f"{label:<24}{old_val:<22}{new_val:<22}")


if __name__ == "__main__":
    main()
