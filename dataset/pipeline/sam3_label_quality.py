"""
How good is SAM3 as the pipeline's annotator? Measured against independent
human ground truth, at the pipeline's operating point (confidence >= 0.5):

  1. iw_hub on the Roboflow "iw.hub detection" export (509 images, one box
     each, CC BY 4.0; dataset/iwhub_roboflow/), with the class_registry.json
     prompt -- box precision / recall / mean IoU of the matched boxes / AP50,
     and what sam3_quality_gate.py would do with each frame (reject, human
     review, auto-approve). These images come from the robot's own lab, so
     this is SAM3's quality in the deployment environment.
  2. iw_hub false alarms on LOCO val images (warehouses with no iw_hub):
     the Roboflow images all contain the object, so precision alone can't
     show SAM3 inventing it where it isn't.
  3. The 5 LOCO classes on the official LOCO val split (2277 images), SAM3
     (eval_loco.py prompts) next to the production model at the same 0.5 --
     the evidence behind each class's pseudo_label_source.

Usage:
    python sam3_label_quality.py
"""

import json
import os
import sys

import numpy as np

import db
from promote_if_better import LOCO_ANNOTATIONS, LOCO_VAL_PATHS, PRODUCTION_WEIGHTS, ap50_per_category
from sam3_quality_gate import AUTO_APPROVE_THRESHOLD, CONFIDENCE_KEEP_BAR

sys.path.insert(0, db.DATASET_ROOT)
import eval_loco  # noqa: E402

ROBOFLOW = os.path.join(db.DATASET_ROOT, "iwhub_roboflow")
OUT = os.path.join(db.PIPELINE_ROOT, "sam3_label_quality.json")
CONF = 0.5
MATCH_IOU = 0.5
SCORE_FLOOR = 0.05  # kept so AP50 can sweep the whole precision-recall curve
N_NEGATIVE = 500


def roboflow_gt():
    images, anns = [], []
    for split in ("train", "valid", "test"):
        with open(os.path.join(ROBOFLOW, split, "_annotations.coco.json")) as f:
            d = json.load(f)
        offset = len(images)
        for img in d["images"]:
            images.append({"id": img["id"] + offset, "path": os.path.join(ROBOFLOW, split, img["file_name"]),
                           "width": img["width"], "height": img["height"]})
        for a in d["annotations"]:
            # ids from 1: COCOeval treats a matched gt id of 0 as unmatched
            anns.append({**a, "id": len(anns) + 1, "image_id": a["image_id"] + offset, "category_id": 0})
    return {"images": images, "annotations": anns, "categories": [{"id": 0, "name": "iw_hub"}]}


def loco_val_gt():
    with open(LOCO_VAL_PATHS) as f:
        gt = eval_loco.build_gt_subset(LOCO_ANNOTATIONS, [line.strip() for line in f if line.strip()])
    for img in gt["images"]:
        img["path"] = os.path.join(db.DATASET_ROOT, img["path"].lstrip("/"))
    return gt


def sam3_boxes(paths, prompts):
    """{path: {prompt_key: [(x0, y0, x1, y1, score), ...]}} down to SCORE_FLOOR."""
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=SCORE_FLOOR)
    out = {}
    for i, p in enumerate(paths):
        state = processor.set_image(Image.open(p).convert("RGB"))
        out[p] = {}
        for key, prompt in prompts.items():
            state = processor.set_text_prompt(prompt=prompt, state=state)
            out[p][key] = [(*b, s) for b, s in zip(state["boxes"].cpu().tolist(), state["scores"].cpu().tolist())]
        if (i + 1) % 250 == 0:
            print(f"[sam3 {i + 1}/{len(paths)}]", flush=True)
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def precision_recall(gt_by_img, pred_by_img):
    """Greedy one-to-one matching per image at MATCH_IOU, predictions by score."""
    tp = fp = n_gt = 0
    ious = []
    for img, gts in gt_by_img.items():
        n_gt += len(gts)
        free = list(gts)
        for box in sorted(pred_by_img.get(img, []), key=lambda b: -b[4]):
            best = max(free, key=lambda g: _iou(box, g), default=None)
            if best is not None and _iou(box, best) >= MATCH_IOU:
                tp += 1
                ious.append(_iou(box, best))
                free.remove(best)
            else:
                fp += 1
    return {"precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / n_gt if n_gt else float("nan"),
            "mean_iou_matched": float(np.mean(ious)) if ious else float("nan"), "tp": tp, "fp": fp, "n_gt": n_gt}


def _gt_boxes(gt, cat):
    by = {img["id"]: [] for img in gt["images"]}
    for a in gt["annotations"]:
        if a["category_id"] == cat and not a.get("iscrowd", 0):
            x, y, w, h = a["bbox"]
            by[a["image_id"]].append((x, y, x + w, y + h))
    return by


def _coco_results(gt, boxes_by_path, key, cat):
    id_by_path = {img["path"]: img["id"] for img in gt["images"]}
    return [{"image_id": id_by_path[p], "category_id": cat, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s}
            for p, d in boxes_by_path.items() for x0, y0, x1, y1, s in d.get(key, [])]


def evaluate(gt, boxes_by_path, key, cat):
    id_by_path = {img["path"]: img["id"] for img in gt["images"]}
    at_conf = {id_by_path[p]: [b for b in d.get(key, []) if b[4] >= CONF] for p, d in boxes_by_path.items()}
    res = precision_recall(_gt_boxes(gt, cat), at_conf)
    res["ap50"] = ap50_per_category(gt, _coco_results(gt, boxes_by_path, key, cat)).get(cat, 0.0)
    return res


def gate_outcomes(boxes_by_path, key):
    """What sam3_quality_gate.py does with each frame: mean confidence of the
    detections it keeps decides reject / review / auto-approve."""
    out = {"reject": 0, "review": 0, "auto_approve": 0}
    for d in boxes_by_path.values():
        scores = [b[4] for b in d.get(key, []) if b[4] >= CONF]
        mean = float(np.mean(scores)) if scores else 0.0
        if mean <= CONFIDENCE_KEEP_BAR:
            out["reject"] += 1
        elif mean > AUTO_APPROVE_THRESHOLD:
            out["auto_approve"] += 1
        else:
            out["review"] += 1
    return out


def production_boxes(gt):
    from two_model_gate import predict_raw

    dets = predict_raw(PRODUCTION_WEIGHTS, [img["path"] for img in gt["images"]], conf=SCORE_FLOOR)
    names = {c["class_idx"]: c["name"] for c in db.loco_classes()}
    return {p: {names[int(c)]: [(*b, float(s)) for b, s, c2 in zip(xyxy, scores, cls) if int(c2) == int(c)]
                for c in set(cls.tolist())} for p, (xyxy, scores, cls) in dets.items()}


def main():
    registry = db.load_class_registry()
    iw = db.class_by_name("iw_hub", registry)
    report = {"conf": CONF, "match_iou": MATCH_IOU, "iw_hub_prompt": iw["prompt"]}

    rf = roboflow_gt()
    rf_boxes = sam3_boxes([img["path"] for img in rf["images"]], {"iw_hub": iw["prompt"]})
    report["iw_hub_roboflow"] = {**evaluate(rf, rf_boxes, "iw_hub", 0), "n_images": len(rf["images"]),
                                 "gate": gate_outcomes(rf_boxes, "iw_hub")}
    print("iw_hub (Roboflow):", json.dumps(report["iw_hub_roboflow"]), flush=True)

    loco = loco_val_gt()
    prompts = {name: eval_loco.CATEGORY_PROMPTS[name] for name in eval_loco.CATEGORY_PROMPTS}
    paths = [img["path"] for img in loco["images"]]
    loco_boxes = sam3_boxes(paths, prompts)
    negatives = sam3_boxes(paths[:N_NEGATIVE], {"iw_hub": iw["prompt"]})
    fa = [sum(b[4] >= CONF for b in d["iw_hub"]) for d in negatives.values()]
    report["iw_hub_false_alarms_on_loco"] = {"n_images": len(fa), "images_with_false_box": int(sum(x > 0 for x in fa)),
                                             "false_boxes": int(sum(fa)),
                                             "gate": gate_outcomes(negatives, "iw_hub")}
    print("iw_hub false alarms (LOCO):", json.dumps(report["iw_hub_false_alarms_on_loco"]), flush=True)

    prod_boxes = production_boxes(loco)
    cat_by_name = {c["name"]: c["id"] for c in loco["categories"]}
    report["loco_classes"] = {}
    for name in prompts:
        report["loco_classes"][name] = {
            "sam3": evaluate(loco, loco_boxes, name, cat_by_name[name]),
            "production": evaluate(loco, prod_boxes, name, cat_by_name[name]),
        }
        r = report["loco_classes"][name]
        print(f"{name}: SAM3 P={r['sam3']['precision']:.3f} R={r['sam3']['recall']:.3f} AP50={r['sam3']['ap50']:.3f} | "
              f"production P={r['production']['precision']:.3f} R={r['production']['recall']:.3f} "
              f"AP50={r['production']['ap50']:.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"LABEL QUALITY DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
