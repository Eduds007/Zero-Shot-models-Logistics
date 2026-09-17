"""
Runs each class's SAM3 text prompt (from class_registry.json) against every
sam3_status='pending' row, using the exact confidence-gate heuristic already
validated in generate_sam3_loco_unannotated.py: mean detection confidence
> 0.5 keeps the frame. The SAM3 box(es) themselves become the YOLO
pseudo-label -- same pseudo-labeling convention already used everywhere
else in this dataset, no manual box drawing.

Usage:
    python sam3_quality_gate.py [--confidence-keep-bar 0.5] [--limit N]
"""

import argparse
import os
import statistics

import db

CONFIDENCE_KEEP_BAR = 0.5


def run_sam3_on_frame(processor, image, prompt):
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt=prompt, state=state)
    boxes = state["boxes"].cpu().tolist()
    scores = state["scores"].cpu().tolist()
    return list(zip(boxes, scores))


def write_pseudo_label(class_name, class_idx, local_path, detections, width, height):
    staging_dir = os.path.join(db.PIPELINE_ROOT, "staging_for_review", class_name)
    images_dir = os.path.join(staging_dir, "images")
    labels_dir = os.path.join(staging_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(local_path))[0]
    dst_img = os.path.join(images_dir, stem + ".jpg")
    if not os.path.exists(dst_img):
        os.symlink(os.path.abspath(local_path), dst_img)

    lines = []
    for (x0, y0, x1, y1), _score in detections:
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        bw, bh = (x1 - x0) / width, (y1 - y0) / height
        cx, cy, bw, bh = (max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh))
        lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return dst_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-keep-bar", type=float, default=CONFIDENCE_KEEP_BAR)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N pending rows (for testing)")
    args = parser.parse_args()

    conn = db.connect()
    registry = db.load_class_registry()
    pending = conn.execute(
        "SELECT id, class_name, local_path FROM frames WHERE sam3_status = 'pending'"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("nothing pending")
        return

    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.confidence_keep_bar)

    n_pass = n_reject = 0
    for i, (row_id, class_name, local_path) in enumerate(pending):
        cls = db.class_by_name(class_name, registry)
        image = Image.open(local_path).convert("RGB")
        w, h = image.size
        detections = run_sam3_on_frame(processor, image, cls["prompt"])
        scores = [s for _, s in detections]
        mean_conf = statistics.mean(scores) if scores else 0.0
        passes = mean_conf > args.confidence_keep_bar

        print(f"[{i + 1}/{len(pending)}] {os.path.basename(local_path)} ({class_name}): "
              f"{len(detections)} det, mean_conf={mean_conf:.3f} -> {'PASS' if passes else 'reject'}")

        if not passes:
            conn.execute(
                "UPDATE frames SET sam3_status = 'reject', sam3_mean_confidence = ? WHERE id = ?",
                (mean_conf, row_id),
            )
            n_reject += 1
            continue

        write_pseudo_label(class_name, cls["class_idx"], local_path, detections, w, h)
        conn.execute(
            "UPDATE frames SET sam3_status = 'pass', sam3_mean_confidence = ?, review_status = 'pending' "
            "WHERE id = ?",
            (mean_conf, row_id),
        )
        n_pass += 1

    conn.commit()
    print(f"\n=== summary ===\npass: {n_pass}\nreject: {n_reject}")


if __name__ == "__main__":
    main()
