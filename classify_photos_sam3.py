"""
Simulates a real "camera sends N images/sec to a SAM3 server" setup: first cuts
the video into actual JPEG files sampled at --fps (evenly spaced in time, not a
frame-skip over the full video decode), then runs SAM3 only on those photos --
exactly as if they had arrived one by one from a camera, with no access to the
frames in between.

Reuses the same category -> text-prompt mapping as dataset/eval_loco.py (plus the
extra non-LOCO classes from classify_video_sam3.py).

Usage:
    python classify_photos_sam3.py --video assets/base_video.mp4 --fps 2 \
        --frames-dir assets/frames_2fps --output assets/output_sam3_2fps_photos.mp4
"""

import argparse
import importlib.util
import os
import time

import cv2

# Loaded by file path rather than via sys.path: the repo root also holds the "sam3"
# submodule checkout, and adding it to sys.path (for a normal sibling import) makes
# Python resolve "import sam3" to that empty namespace package instead of the
# properly pip-installed one.
_video_script_path = os.path.join(os.path.dirname(__file__), "classify_video_sam3.py")
_spec = importlib.util.spec_from_file_location("classify_video_sam3", _video_script_path)
_classify_video_sam3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_classify_video_sam3)
ALL_PROMPTS = _classify_video_sam3.ALL_PROMPTS
draw_detections = _classify_video_sam3.draw_detections


def extract_photos(video_path, frames_dir, target_fps):
    """Cuts the video into JPEGs sampled at target_fps, evenly spaced in time.

    Returns the sorted list of written file paths.
    """
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame indices evenly spaced in time at target_fps (e.g. source=29.86fps,
    # target=2fps -> indices 0, ~14.93, ~29.86, ... rounded to the nearest frame)
    step = source_fps / target_fps
    k = 0
    written = []
    while True:
        frame_idx = round(k * step)
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(frames_dir, f"photo_{k:05d}.jpg")
        cv2.imwrite(out_path, frame)
        written.append(out_path)
        k += 1
    cap.release()
    return written, source_fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/base_video.mp4")
    parser.add_argument("--fps", type=float, default=2.0, help="Photos captured per second.")
    parser.add_argument("--frames-dir", default="assets/frames_2fps")
    parser.add_argument("--output", default="assets/output_sam3_2fps_photos.mp4",
                         help="Reassembled video, played back at --fps so it reflects the real capture cadence.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Cutting {args.video} into photos at {args.fps} fps...")
    t0 = time.perf_counter()
    photo_paths, source_fps = extract_photos(args.video, args.frames_dir, args.fps)
    extract_elapsed = time.perf_counter() - t0
    print(f"Wrote {len(photo_paths)} photos to {args.frames_dir}/ in {extract_elapsed:.1f}s "
          f"(source video was {source_fps:.2f} fps)")

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

    writer = None
    print(f"\nAnalyzing {len(photo_paths)} photos (only these -- no other frame is touched)...")
    start_time = time.perf_counter()
    for i, photo_path in enumerate(photo_paths):
        frame_bgr = cv2.imread(photo_path)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        state = processor.set_image(image)
        for category, prompt in ALL_PROMPTS.items():
            state = processor.set_text_prompt(prompt=prompt, state=state)
            boxes = state["boxes"].cpu().tolist()
            scores = state["scores"].cpu().tolist()
            draw_detections(frame_bgr, category, boxes, scores)

        if writer is None:
            height, width = frame_bgr.shape[:2]
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
        writer.write(frame_bgr)

        if (i + 1) % 20 == 0 or (i + 1) == len(photo_paths):
            elapsed = time.perf_counter() - start_time
            print(f"[{i + 1}/{len(photo_paths)}] {elapsed:.1f}s elapsed, "
                  f"{(i + 1) / elapsed:.2f} photos/sec")

    if writer is not None:
        writer.release()

    inference_elapsed = time.perf_counter() - start_time
    n = len(photo_paths)
    print(f"\nSaved {args.output} (plays back at {args.fps} fps)")
    print(f"Extraction: {extract_elapsed:.1f}s for {n} photos")
    print(f"Analysis:   {inference_elapsed:.1f}s for {n} photos "
          f"({inference_elapsed / n:.3f}s/photo, {n / inference_elapsed:.2f} photos/sec)")


if __name__ == "__main__":
    main()
