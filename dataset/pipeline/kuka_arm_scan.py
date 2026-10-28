"""
Before the KUKA arm becomes a second pipeline class: does it already show up
in the frames the robot has captured (every frame in pipeline.db), and does
SAM3 find it with a plain-words prompt? Frames that contain the arm would be
trained as background for it unless relabeled (missing annotation).

For each candidate prompt: how many frames have a SAM3 box at >= 0.5, and a
contact sheet of the highest-scoring boxes (green = box at >= 0.5) to check
by eye whether they are the arm or false alarms.

Usage:
    python kuka_arm_scan.py
"""

import json
import os

import cv2
import numpy as np

import db
from sam3_label_quality import CONF, sam3_boxes

PROMPTS = ["orange industrial robot arm", "robotic arm", "industrial robot", "orange robot arm"]
OUT_DIR = os.path.join(db.PIPELINE_ROOT, "kuka_arm_scan")
N_SHEET = 24
TILE = 320


def contact_sheet(dets, path):
    """dets: [(image_path, (x0, y0, x1, y1, score))], best first."""
    tiles = []
    for p, (x0, y0, x1, y1, s) in dets[:N_SHEET]:
        img = cv2.imread(p)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0) if s >= CONF else (0, 0, 255), 4)
        img = cv2.resize(img, (TILE, int(TILE * img.shape[0] / img.shape[1])))
        img = cv2.copyMakeBorder(img, 0, max(0, 240 - img.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:240]
        cv2.putText(img, f"{s:.2f} {os.path.basename(p)[8:23]}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def main():
    conn = db.connect()
    paths = [r[0] for r in conn.execute("SELECT DISTINCT local_path FROM frames ORDER BY local_path")]
    paths = [p for p in paths if os.path.exists(p)]
    print(f"{len(paths)} frames", flush=True)
    boxes = sam3_boxes(paths, {q: q for q in PROMPTS})
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"n_frames": len(paths), "conf": CONF, "prompts": {}}
    for i, q in enumerate(PROMPTS):
        dets = sorted(((p, b) for p in paths for b in boxes[p][q]), key=lambda d: -d[1][4])
        hits = sorted({p for p, b in dets if b[4] >= CONF})
        report["prompts"][q] = {"frames_with_box": len(hits), "boxes": sum(b[4] >= CONF for _p, b in dets),
                                "frames": hits}
        contact_sheet(dets, os.path.join(OUT_DIR, f"sheet_{i}.jpg"))
        print(f"{q!r}: {len(hits)} frames with a box >= {CONF} -> sheet_{i}.jpg", flush=True)
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("KUKA SCAN DONE", flush=True)


if __name__ == "__main__":
    main()
