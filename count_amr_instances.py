"""
Counts *distinct* AMR instances in the video, as opposed to raw per-frame
detections (which would count the same physical robot once per frame it's
visible in).

Uses SAM3's video predictor (dense tracking): a text prompt is propagated
across the video, and the model assigns a persistent object id to each
instance, discovering new ones as they enter the scene. The count of distinct
AMRs is the number of unique object ids seen.

The video predictor loads every frame (resized to 1008x1008) into one big
float32 GPU tensor up front -- for this video's 2329 frames that's ~27GB,
more than a single 24GB GPU can hold. So the video is processed in
overlapping chunks (each one fits comfortably in memory), and object ids
that reappear with a similar box in the overlap between two consecutive
chunks are merged back into a single instance.

Usage:
    python count_amr_instances.py --video assets/base_video.mp4
"""

import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict

import cv2

_video_script_path = os.path.join(os.path.dirname(__file__), "classify_video_sam3.py")
_spec = importlib.util.spec_from_file_location("classify_video_sam3", _video_script_path)
_classify_video_sam3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_classify_video_sam3)
PROMPT = _classify_video_sam3.EXTRA_PROMPTS["amr"]


def make_chunks(total_frames, chunk_size, overlap):
    chunks = []
    start = 0
    while True:
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))
        if end == total_frames:
            break
        start = end - overlap
    return chunks


def extract_subclip(video_path, out_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
    cap.release()


def box_iou_xywh(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def run_chunk(predictor, subclip_path, chunk_start, prompt):
    """Returns {obj_id: {frame_idx_abs: box_xywh_normalized}}."""
    resp = predictor.handle_request(request=dict(type="start_session", resource_path=subclip_path))
    session_id = resp["session_id"]
    predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
    )

    per_obj = defaultdict(dict)
    for resp in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id)):
        out = resp["outputs"]
        frame_idx_abs = chunk_start + resp["frame_index"]
        for obj_id, box in zip(out["out_obj_ids"].tolist(), out["out_boxes_xywh"].tolist()):
            per_obj[obj_id][frame_idx_abs] = box

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return per_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/base_video.mp4")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--min-frames", type=int, default=3,
                         help="Minimum number of frames a (merged) instance must appear in to count "
                              "as real rather than a one-off false positive.")
    parser.add_argument("--iou-match", type=float, default=0.15,
                         help="Min box IoU (checked at matching/nearby frames) to treat two chunks' "
                              "object ids as the same instance.")
    parser.add_argument("--max-gap-frames", type=int, default=20,
                         help="If two chunks' tracks don't share an exact frame in the overlap region "
                              "(e.g. one track starts a few frames after the other ends), still compare "
                              "their nearest boxes if the gap between them is within this many frames.")
    parser.add_argument("--tmp-dir", default="/tmp/sam3_amr_chunks")
    parser.add_argument("--cache", default="/tmp/sam3_amr_chunks/chunk_results.json",
                         help="Raw per-chunk detections are cached here so --iou-match/--max-gap-frames "
                              "can be re-tuned without re-running the (slow) video predictor.")
    parser.add_argument("--from-cache", action="store_true",
                         help="Skip inference entirely and just re-run the merge step on --cache.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    chunks = make_chunks(total_frames, args.chunk_size, args.overlap)
    print(f"Video: {total_frames} frames @ {fps:.2f}fps -> {len(chunks)} chunks: {chunks}")

    if args.from_cache:
        with open(args.cache) as f:
            raw = json.load(f)
        chunk_results = [
            (c_start, c_end, {int(oid): {int(fr): box for fr, box in frames.items()} for oid, frames in per_obj.items()})
            for c_start, c_end, per_obj in raw
        ]
        elapsed = 0.0
    else:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from sam3.model_builder import build_sam3_video_predictor

        print("Loading SAM3 video predictor...")
        predictor = build_sam3_video_predictor(gpus_to_use=[0])

        os.makedirs(args.tmp_dir, exist_ok=True)

        # global_id -> {frame_idx_abs: box}; tracks are first collected per-chunk with
        # ids local to that chunk, then merged across chunk boundaries below.
        chunk_results = []  # list of (chunk_start, chunk_end, {obj_id: {frame_idx: box}})
        start_time = time.perf_counter()
        for i, (c_start, c_end) in enumerate(chunks):
            print(f"\n[chunk {i + 1}/{len(chunks)}] frames [{c_start}, {c_end})")
            subclip_path = os.path.join(args.tmp_dir, f"chunk_{i:03d}.mp4")
            extract_subclip(args.video, subclip_path, c_start, c_end)
            per_obj = run_chunk(predictor, subclip_path, c_start, args.prompt)
            os.remove(subclip_path)
            n_dets = sum(len(v) for v in per_obj.values())
            print(f"  -> {len(per_obj)} object id(s) in this chunk, {n_dets} detections total")
            chunk_results.append((c_start, c_end, per_obj))

        predictor.shutdown()
        elapsed = time.perf_counter() - start_time

        with open(args.cache, "w") as f:
            json.dump(chunk_results, f)
        print(f"\nCached raw per-chunk detections to {args.cache}")

    # merge object ids across consecutive chunks via union-find. Two tracks are the
    # same instance if, within --max-gap-frames of each other in time (not necessarily
    # the exact same frame -- a track can end a few frames before the other starts,
    # e.g. right at a chunk boundary while its detection status was still "warming up"),
    # their boxes have IoU >= --iou-match.
    uf = UnionFind()
    node_key = lambda ci, oid: (ci, oid)  # noqa: E731
    all_frames = defaultdict(dict)  # (chunk_idx, obj_id) -> {frame_idx: box}
    for ci, (c_start, c_end, per_obj) in enumerate(chunk_results):
        for oid, frames in per_obj.items():
            uf.find(node_key(ci, oid))
            all_frames[node_key(ci, oid)] = frames

    for ci in range(len(chunk_results) - 1):
        c_start, c_end, per_obj = chunk_results[ci]
        n_start, n_end, n_per_obj = chunk_results[ci + 1]
        window_lo, window_hi = n_start - args.max_gap_frames, c_end + args.max_gap_frames
        for oid_a, frames_a in per_obj.items():
            fa_near = {f: b for f, b in frames_a.items() if window_lo <= f <= window_hi}
            if not fa_near:
                continue
            for oid_b, frames_b in n_per_obj.items():
                fb_near = {f: b for f, b in frames_b.items() if window_lo <= f <= window_hi}
                best_iou = 0.0
                for fa, box_a in fa_near.items():
                    for fb, box_b in fb_near.items():
                        if abs(fa - fb) <= args.max_gap_frames:
                            best_iou = max(best_iou, box_iou_xywh(box_a, box_b))
                if best_iou >= args.iou_match:
                    uf.union(node_key(ci, oid_a), node_key(ci + 1, oid_b))

    merged = defaultdict(dict)  # root -> {frame_idx: box} (union of all frames)
    for key, frames in all_frames.items():
        root = uf.find(key)
        merged[root].update(frames)

    confirmed = {root: frames for root, frames in merged.items() if len(frames) >= args.min_frames}

    print(f"\nProcessed {total_frames} frames across {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"\nRaw object ids across all chunks (before merge): {len(all_frames)}")
    print(f"Distinct AMR instances after merging chunk overlaps: {len(merged)}")
    print(f"Distinct AMR instances (>= {args.min_frames} frames, likely real): {len(confirmed)}")
    print("\nPer-instance detail:")
    for root, frames in sorted(merged.items(), key=lambda kv: min(kv[1])):
        fs = sorted(frames.keys())
        tag = "CONFIRMED" if root in confirmed else "noise?"
        t0, t1 = fs[0] / fps, fs[-1] / fps
        print(f"  instance {root}: {len(fs)} frames, frame {fs[0]}-{fs[-1]} (~{t0:.1f}s-{t1:.1f}s) [{tag}]")


if __name__ == "__main__":
    main()
