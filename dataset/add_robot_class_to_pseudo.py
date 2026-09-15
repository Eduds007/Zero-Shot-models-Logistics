"""
Adds a 6th class ("iw_hub", the yellow AMR) to the existing 5-class SAM3
pseudo-labels for the video frames (dataset/sam3_v6_pseudo_split), using the
best prompt found in dataset/compare_amr_prompts.py: "yellow robot platform"
(confidence_threshold=0.5, top-1 box only -- there is only ever one robot in
frame).

Does NOT touch dataset/sam3_v6_pseudo_split/ (already published as v2 on HF,
5 classes only). Writes a parallel copy with the robot class added:
    dataset/sam3_v6_pseudo_split_with_robot/images/{train,val}  (symlinks)
    dataset/sam3_v6_pseudo_split_with_robot/labels/{train,val}  (5 LOCO
        classes copied as-is, plus a 6th "5 ..." line when the robot is
        detected above threshold)

Usage:
    python add_robot_class_to_pseudo.py
"""

import os
import shutil

from eval_loco import CATEGORY_PROMPTS  # noqa: F401 (kept for parity/context)

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "sam3_v6_pseudo_split")
DST = os.path.join(ROOT, "sam3_v6_pseudo_split_with_robot")

ROBOT_PROMPT = "yellow robot platform"
ROBOT_CLASS_ID = 5
ROBOT_THRESHOLD = 0.5


def main():
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("Loading SAM3...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=ROBOT_THRESHOLD)

    n_with_robot = 0
    n_total = 0

    for split in ("train", "val"):
        src_img_dir = os.path.join(SRC, "images", split)
        src_lbl_dir = os.path.join(SRC, "labels", split)
        dst_img_dir = os.path.join(DST, "images", split)
        dst_lbl_dir = os.path.join(DST, "labels", split)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for fname in sorted(os.listdir(src_img_dir)):
            if not fname.endswith(".jpg"):
                continue
            n_total += 1
            src_img = os.path.join(src_img_dir, fname)
            dst_img = os.path.join(dst_img_dir, fname)
            if not os.path.exists(dst_img):
                os.symlink(os.path.realpath(src_img), dst_img)

            stem = os.path.splitext(fname)[0]
            src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
            existing_lines = []
            if os.path.exists(src_lbl):
                with open(src_lbl) as f:
                    existing_lines = [ln for ln in f.read().splitlines() if ln.strip()]

            image = Image.open(src_img).convert("RGB")
            w, h = image.size
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt=ROBOT_PROMPT, state=state)
            boxes = state["boxes"].cpu().tolist()
            scores = state["scores"].cpu().tolist()

            if boxes:
                # top-1 only -- there is only one robot in the scene
                pairs = sorted(zip(boxes, scores), key=lambda bs: bs[1], reverse=True)
                box, score = pairs[0]
                x0, y0, x1, y1 = box
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                bw, bh = (x1 - x0) / w, (y1 - y0) / h
                cx, cy, bw, bh = (max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh))
                existing_lines.append(f"{ROBOT_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_with_robot += 1
                print(f"[{split}] {fname}: robot conf={score:.3f} -> ADDED")
            else:
                print(f"[{split}] {fname}: no robot detected above {ROBOT_THRESHOLD}")

            with open(os.path.join(dst_lbl_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(existing_lines) + ("\n" if existing_lines else ""))

    print("\n=== summary ===")
    print(f"total video frames: {n_total}")
    print(f"frames with robot box added (conf>{ROBOT_THRESHOLD}): {n_with_robot}")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
