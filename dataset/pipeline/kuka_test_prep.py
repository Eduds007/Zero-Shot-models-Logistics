"""
Test set for the KUKA-arm experiment, drawn only from the 2026-10-05 capture
(training uses the 09-16/09-17 frames, so no near-duplicate video frames
cross the split). SAM3 ("robotic arm") pre-labels the frames; a human then
corrects every box, so the gate measures the arm against human labels, not
against SAM3.

Frames go to review when SAM3's best box scores >= REVIEW_FLOOR -- below the
pipeline's 0.5, so frames where SAM3 is unsure (or misses the arm) are
reviewed too. Only the boxes at >= 0.5 are pre-drawn.

Writes kuka_arm_test_review/{images,labels,data.yaml} and a zip of it, in
YOLO format (one class, robotic_arm) for Roboflow / labelImg.

Usage:
    python kuka_test_prep.py
"""

import os
import shutil

import cv2

import db
from sam3_label_quality import CONF, sam3_boxes

PROMPT = "robotic arm"
TEST_DAY = "20261005"
REVIEW_FLOOR = 0.3
OUT = os.path.join(db.PIPELINE_ROOT, "kuka_arm_test_review")


def main():
    conn = db.connect()
    paths = [r[0] for r in conn.execute("SELECT DISTINCT local_path FROM frames ORDER BY local_path")
             if f"capture_{TEST_DAY}_" in r[0] and os.path.exists(r[0])]
    boxes = sam3_boxes(paths, {"arm": PROMPT})
    review = [p for p in paths if max((b[4] for b in boxes[p]["arm"]), default=0) >= REVIEW_FLOOR]
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "images"))
    os.makedirs(os.path.join(OUT, "labels"))
    n_boxes = 0
    for p in review:
        h, w = cv2.imread(p).shape[:2]
        shutil.copy(p, os.path.join(OUT, "images"))
        lines = [f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
                 for x0, y0, x1, y1, s in boxes[p]["arm"] if s >= CONF]
        n_boxes += len(lines)
        with open(os.path.join(OUT, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("train: images\nval: images\nnc: 1\nnames: ['robotic_arm']\n")
    shutil.make_archive(OUT, "zip", OUT)
    print(f"{len(paths)} frames from {TEST_DAY}; {len(review)} to review (best box >= {REVIEW_FLOOR}), "
          f"{n_boxes} pre-drawn boxes (>= {CONF}) -> {OUT}.zip", flush=True)


if __name__ == "__main__":
    main()
