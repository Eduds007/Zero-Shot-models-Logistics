"""
Compares candidate text prompts for detecting the yellow AMR (iw.hub tow robot)
in the lab video, on top of the frame-selection/false-positive work already
documented in classify_video_sam3.py's EXTRA_PROMPTS comment:
  - "autonomous mobile robot" / "AGV" / "tugger" / "AMR" -> zero detections
  - "yellow mobile robot" -> fires, but also matches a tracked mini dumper
    elsewhere in the video (false positive, conf~0.4-0.8)
  - "yellow robot platform" -> current best: hits the real AMR (conf 0.58-0.70)
    while staying below the mini dumper's score (conf 0.37)

Runs each candidate prompt (low confidence threshold, so nothing is filtered
out yet) over all frames in assets/frames_2fps, and:
  1. writes a per-frame, per-prompt CSV of every detection (box + score) for
     manual review, and
  2. writes one annotated comparison video (one color per prompt, top-1 box
     per prompt per frame only, to keep it readable) so the results can be
     watched directly instead of just read as numbers.

Nothing here touches LOCO categories/eval_loco.py or the pseudo-label
pipelines -- this is a prompt-exploration script only, not wired into any
dataset generation.

Usage:
    python compare_amr_prompts.py --frames-dir ../assets/frames_2fps --out-dir /tmp/amr_prompts
"""

import argparse
import csv
import glob
import os

CANDIDATE_PROMPTS = {
    "yellow_robot_platform": "yellow robot platform",   # current baseline (best so far)
    "yellow_mobile_robot": "yellow mobile robot",        # known false-positive baseline
    "yellow_amr": "yellow AMR",
    "amr_yellow": "AMR yellow",
    "iwhub": "iwhub",
    "iw_dot_hub": "iw.hub",
    "iw_hub_spaced": "iw hub",
    "yellow_iwhub": "yellow iwhub robot",
}

# Fixed BGR color per prompt so it's consistent frame to frame.
PROMPT_COLORS = {
    "yellow_robot_platform": (0, 255, 255),
    "yellow_mobile_robot": (255, 0, 255),
    "yellow_amr": (0, 165, 255),
    "amr_yellow": (255, 128, 0),
    "iwhub": (0, 0, 255),
    "iw_dot_hub": (0, 255, 0),
    "iw_hub_spaced": (255, 255, 0),
    "yellow_iwhub": (128, 0, 255),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--low-threshold", type=float, default=0.05,
                         help="Confidence floor for logging -- kept low so weak "
                              "hits are still visible for comparison.")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    print(f"Found {len(frame_paths)} frames in {args.frames_dir}")

    import cv2
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("Loading SAM3...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.low_threshold)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "amr_prompt_comparison.csv")
    video_path = os.path.join(args.out_dir, "amr_prompt_comparison.mp4")

    writer = None
    csv_rows = [["frame", "prompt_key", "prompt_text", "rank", "score", "x0", "y0", "x1", "y1"]]
    top1_scores = {k: [] for k in CANDIDATE_PROMPTS}

    for i, path in enumerate(frame_paths):
        frame_bgr = cv2.imread(path)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        state = processor.set_image(image)

        for key, prompt in CANDIDATE_PROMPTS.items():
            state = processor.set_text_prompt(prompt=prompt, state=state)
            boxes = state["boxes"].cpu().tolist()
            scores = state["scores"].cpu().tolist()
            pairs = sorted(zip(boxes, scores), key=lambda bs: bs[1], reverse=True)

            for rank, (box, score) in enumerate(pairs):
                x0, y0, x1, y1 = box
                csv_rows.append([os.path.basename(path), key, prompt, rank, f"{score:.4f}",
                                  f"{x0:.1f}", f"{y0:.1f}", f"{x1:.1f}", f"{y1:.1f}"])

            top1_scores[key].append(pairs[0][1] if pairs else 0.0)

            if pairs:
                box, score = pairs[0]
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                color = PROMPT_COLORS[key]
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
                text = f"{key} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame_bgr, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
                cv2.putText(frame_bgr, text, (x0 + 2, max(0, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
        writer.write(frame_bgr)

        if (i + 1) % 20 == 0 or (i + 1) == len(frame_paths):
            print(f"[{i + 1}/{len(frame_paths)}] processed")

    if writer is not None:
        writer.release()

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print("\n=== top-1 score summary (per prompt, across all frames) ===")
    print(f"{'prompt_key':22s} {'prompt_text':22s} {'mean':>6s} {'max':>6s} {'frames>0.5':>11s}")
    for key, prompt in CANDIDATE_PROMPTS.items():
        scores = top1_scores[key]
        mean_s = sum(scores) / len(scores) if scores else 0.0
        max_s = max(scores) if scores else 0.0
        above = sum(1 for s in scores if s > 0.5)
        print(f"{key:22s} {prompt:22s} {mean_s:6.3f} {max_s:6.3f} {above:11d}")

    print(f"\nWrote {video_path} and {csv_path}")


if __name__ == "__main__":
    main()
