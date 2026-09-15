"""
Builds a mixed fine-tune dataset: the 128 SAM3 video-frame pseudo-labels
(with the iw_hub class, dataset/sam3_v6_pseudo_split_with_robot) PLUS a
random sample of official LOCO train/val images of matching size, so the
video frames are no longer ~100% of what the frozen fine-tune's head sees.

Rationale (per conversation): the video-frame set alone is small (102/26)
and its class distribution is very different from LOCO's (dominated by
small_load_carrier, near-absent pallet_truck/stillage in some splits) --
that mismatch, not the new class, was shown to be the cause of the
LOCO-val forgetting in yolov8_v10_robot_frozen/yolov8_v11_control_frozen
(see loco_eval_results.csv). Diluting the video sample 50/50 with real
LOCO images (which have no iw_hub boxes -- correct negatives for that
class) should pull the fine-tune's effective distribution back toward
LOCO's, without diluting away the new class's only positive examples
(the video frames are always included in full).

Output (not committed, local only):
    dataset/sam3_v6_pseudo_mixed_with_robot/images/{train,val}  (symlinks:
        102+102 train, 26+26 val)
    dataset/sam3_v6_pseudo_mixed_with_robot/labels/{train,val}  (video
        frames: existing 6-class labels incl. iw_hub; LOCO images: existing
        5-class labels, unmodified -- no iw_hub line, which is the correct
        negative since LOCO has no AMR)

Usage:
    python build_mixed_robot_dataset.py
"""

import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_SRC = os.path.join(ROOT, "sam3_v6_pseudo_split_with_robot")
LOCO_SRC = os.path.join(ROOT, "yolo")  # official LOCO YOLO conversion, images/{train,val} + labels/{train,val}
DST = os.path.join(ROOT, "sam3_v6_pseudo_mixed_with_robot")

SEED = 0


def link_dir(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, filenames):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    for fname in filenames:
        src_img = os.path.join(src_img_dir, fname)
        dst_img = os.path.join(dst_img_dir, fname)
        if not os.path.exists(dst_img):
            os.symlink(os.path.realpath(src_img), dst_img)

        stem = os.path.splitext(fname)[0]
        src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
        dst_lbl = os.path.join(dst_lbl_dir, stem + ".txt")
        if not os.path.exists(dst_lbl):
            if os.path.exists(src_lbl):
                os.symlink(os.path.realpath(src_lbl), dst_lbl)
            else:
                open(dst_lbl, "w").close()  # LOCO images with 0 boxes of the 5 classes, if any


def main():
    random.seed(SEED)
    counts = {}

    for split in ("train", "val"):
        video_img_dir = os.path.join(VIDEO_SRC, "images", split)
        video_lbl_dir = os.path.join(VIDEO_SRC, "labels", split)
        video_files = sorted(os.listdir(video_img_dir))

        loco_img_dir = os.path.join(LOCO_SRC, "images", split)
        loco_lbl_dir = os.path.join(LOCO_SRC, "labels", split)
        loco_all_files = sorted(f for f in os.listdir(loco_img_dir) if f.endswith(".jpg"))
        loco_sample = random.sample(loco_all_files, len(video_files))

        dst_img_dir = os.path.join(DST, "images", split)
        dst_lbl_dir = os.path.join(DST, "labels", split)
        link_dir(video_img_dir, video_lbl_dir, dst_img_dir, dst_lbl_dir, video_files)
        link_dir(loco_img_dir, loco_lbl_dir, dst_img_dir, dst_lbl_dir, loco_sample)

        counts[split] = (len(video_files), len(loco_sample))
        print(f"[{split}] {len(video_files)} video frames + {len(loco_sample)} LOCO images "
              f"= {len(video_files) + len(loco_sample)} total")

    print(f"\nwrote {DST}")
    print(f"counts: {counts}")


if __name__ == "__main__":
    main()
