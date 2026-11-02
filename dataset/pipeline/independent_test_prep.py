"""
Independent test sets from a capture session the pipeline never saw (frames
recorded by capture_node.py alone, without sync_node.py, so they never
reach dataset/incoming/ or the review queue).

  1. drops near-duplicates (consecutive frames whose 64x36 grayscale
     thumbnails differ by less than DUP_DIFF -- the robot standing still);
  2. runs SAM3 with each pipeline class's registry prompt;
  3. per class, writes a review folder for labelImg with every kept frame
     (not only those SAM3 finds the object in -- that would leave SAM3's
     misses, the hard cases, out of the test), boxes >= 0.5 pre-drawn, one
     class (index 0).

A human then corrects every box; the corrected folder becomes
fixed_testset/<class>_<session>, as kuka_test_prep.py did for the arm.

Usage:
    python independent_test_prep.py <frames dir> <session name>
"""

import os
import shutil
import sys

import cv2
import numpy as np

import db
from sam3_label_quality import CONF, sam3_boxes

DUP_DIFF = 4.0


def dedupe(paths):
    kept, prev = [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)
        if prev is None or np.abs(thumb - prev).mean() >= DUP_DIFF:
            kept.append(p)
        prev = thumb
    return kept


def main(frames_dir, session):
    out_root = os.path.join(db.PIPELINE_ROOT, "independent_test", session)
    paths = sorted(os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    kept = dedupe(paths)
    print(f"{len(paths)} frames, {len(kept)} after dropping near-duplicates", flush=True)
    classes = db.pipeline_classes()
    boxes = sam3_boxes(kept, {c["name"]: c["prompt"] for c in classes})
    for cls in classes:
        name = cls["name"]
        out = os.path.join(out_root, name)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(os.path.join(out, "images"))
        os.makedirs(os.path.join(out, "labels"))
        review = kept
        n_boxes = 0
        for p in review:
            h, w = cv2.imread(p).shape[:2]
            shutil.copy(p, os.path.join(out, "images"))
            lines = [f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
                     for x0, y0, x1, y1, s in boxes[p][name] if s >= CONF]
            n_boxes += len(lines)
            with open(os.path.join(out, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        with open(os.path.join(out, "labels", "classes.txt"), "w") as f:
            f.write(name + "\n")
        shutil.make_archive(out, "zip", out)
        print(f"{name}: {len(review)} frames to review, {sum(1 for p in review if any(b[4] >= CONF for b in boxes[p][name]))} with a SAM3 box, "
              f"{n_boxes} pre-drawn boxes (>= {CONF}) -> {out}.zip", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
