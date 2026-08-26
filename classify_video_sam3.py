"""
Run SAM3 (image predictor, text-prompted) frame-by-frame over a video, detecting
every LOCO category (plus extra classes not in LOCO) in each frame, and write an
annotated copy of the video with boxes + labels drawn.

Reuses the same category -> text-prompt mapping as dataset/eval_loco.py so the
video demo and the LOCO benchmark stay in sync for the shared categories.

Usage:
    python classify_video_sam3.py --video assets/base_video.mp4 --output assets/output_sam3.mp4
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dataset"))
from eval_loco import CATEGORY_PROMPTS  # noqa: E402

# Classes to detect that aren't part of the LOCO benchmark (no GT annotations for
# them), so they live here instead of eval_loco.CATEGORY_PROMPTS.
# "autonomous mobile robot" / "AGV" / "tugger" / "AMR" all returned zero detections
# on this video's actual AMR (a low-profile yellow tow robot) -- SAM3 doesn't ground
# those phrases here. "yellow mobile robot" does fire, but also matches the tracked
# mini dumper elsewhere in the video (false positive at conf~0.4-0.8). "yellow robot
# platform" was the only phrasing found that hits the real AMR (conf 0.58-0.70) while
# staying below the mini dumper's score (conf 0.37) at the default threshold.
EXTRA_PROMPTS = {
    "amr": "yellow robot platform",
}

ALL_PROMPTS = {**CATEGORY_PROMPTS, **EXTRA_PROMPTS}

# Fixed BGR color per category, so a class reads the same color in every frame.
CATEGORY_COLORS = {
    "small_load_carrier": (255, 128, 0),
    "forklift": (0, 0, 255),
    "pallet": (0, 200, 0),
    "stillage": (255, 0, 255),
    "pallet_truck": (0, 200, 255),
    "amr": (0, 255, 255),
}


def draw_detections(frame_bgr, category, boxes, scores):
    color = CATEGORY_COLORS[category]
    for box, score in zip(boxes, scores):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
        text = f"{category} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(frame_bgr, text, (x0 + 2, max(0, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/base_video.mp4")
    parser.add_argument("--output", default="assets/output_sam3.mp4")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--frame-skip", type=int, default=1,
                         help="Run detection every N frames; skipped frames reuse the last detections.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop early, for a quick test run.")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("Loading SAM3...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
    print(f"Categories: {ALL_PROMPTS}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    last_detections = {}  # category -> (boxes, scores), reused on skipped frames
    frame_idx = 0
    start_time = time.perf_counter()

    while frame_idx < total_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_skip == 0:
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            state = processor.set_image(image)
            last_detections = {}
            for category, prompt in ALL_PROMPTS.items():
                state = processor.set_text_prompt(prompt=prompt, state=state)
                boxes = state["boxes"].cpu().tolist()
                scores = state["scores"].cpu().tolist()
                last_detections[category] = (boxes, scores)

        for category, (boxes, scores) in last_detections.items():
            draw_detections(frame_bgr, category, boxes, scores)

        writer.write(frame_bgr)

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.perf_counter() - start_time
            print(f"[{frame_idx}/{total_frames}] {elapsed:.1f}s elapsed, "
                  f"{frame_idx / elapsed:.2f} fps effective")

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - start_time
    print(f"\nSaved annotated video to {args.output}")
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.2f} fps effective)")


if __name__ == "__main__":
    main()
