"""
Iterative self-training on new, unlabeled robot-lab video (Mayershofer et
al., "Improving visual object detection using synthetic self-training",
Logistics Journal 2021): each round, the previous round's model pseudo-labels
every class (5 LOCO + iw_hub) on the video's frames, the model is retrained
on the base batch plus those frames, and the new model becomes the next
round's teacher. The same frames are re-labeled every round by an improving
teacher, as in classic self-training; new videos can be added as extra
rounds.

The base batch (round 0: v13 composition + LOCO pseudo-labels on the 83
approved iw_hub frames) is reused unchanged every round, and every round
restarts from the pristine LOCO baseline (run_finetune.finetune, freeze=21),
so rounds differ only in how the video frames are labeled.

Evaluation per round: the 5 LOCO classes on the official LOCO val, and
iw_hub on fixed_testset (30 lab-video frames + 50 Roboflow images) -- the
509-image held_out_val/iw_hub was lost (its images were symlinks into /tmp).

Usage:
    python self_training.py --video ~/videos/lab.mp4 --base-batch training_batches/pipeline_r0_pseudo \
        --teacher ../../runs/detect/yolo_runs/pipeline_r0_pseudo/weights/best.pt --rounds 3
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

import db
from copy_paste_augment import _write_yolo_labels
from promote_if_better import evaluate_model, yolo_dir_to_coco_gt
from run_finetune import YOLO_RUNS_DIR, finetune
from two_model_gate import PerImageMatches, merge_detections, predict_raw, to_coco_results

OUT_ROOT = os.path.join(db.PIPELINE_ROOT, "self_training")
SAMPLE_FPS = 3.0
KEEP_SHARPEST = 0.7  # handheld video: drop the blurriest 30% of sampled frames
CONF = 0.5


def extract_frames(video, out_dir, sample_fps=SAMPLE_FPS, keep=KEEP_SHARPEST):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / sample_fps))
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            blur = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            frames.append((blur, idx, frame))
        idx += 1
    cutoff = np.quantile([b for b, _, _ in frames], 1 - keep)
    stem = os.path.splitext(os.path.basename(video))[0].replace(" ", "_")
    paths = []
    for blur, i, frame in frames:
        if blur >= cutoff:
            p = os.path.join(out_dir, f"video_{stem}_{i:05d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
    print(f"{video}: {idx} frames, sampled {len(frames)} at {sample_fps} fps, kept {len(paths)} sharpest")
    return paths


def label_frames(teacher, frame_paths, conf=CONF, loco_teacher=None, loco_idx=(), pipeline_idx=()):
    """{path: [(cls, cx, cy, w, h), ...]} for every class the teacher knows.

    With loco_teacher, the LOCO classes come from loco_teacher (e.g. the
    production model, which hasn't forgotten them) and only the pipeline
    classes from teacher -- otherwise a teacher that already forgot e.g.
    pallet_truck passes the forgetting on through its labels. A LOCO box
    overlapping a pipeline-class box (IoU > MERGE_IOU) is dropped, same rule
    as the two-model merge."""
    dets = predict_raw(teacher, frame_paths, conf=conf)
    if loco_teacher:
        loco_dets = predict_raw(loco_teacher, frame_paths, conf=conf)
        for p in frame_paths:
            (pb, ps, pc), (lb, ls, lc) = dets[p], loco_dets[p]
            own = np.isin(pc, list(pipeline_idx))
            lkeep = np.isin(lc, list(loco_idx))
            dets[p] = merge_detections((lb[lkeep], ls[lkeep], lc[lkeep]), (pb[own], ps[own], pc[own]),
                                       pipeline_idx, spec_conf=conf)
    out, counts = {}, {}
    for p, (xyxy, _scores, classes) in dets.items():
        h, w = cv2.imread(p).shape[:2]
        boxes = []
        for (x0, y0, x1, y1), c in zip(xyxy, classes):
            boxes.append((int(c), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
            counts[int(c)] = counts.get(int(c), 0) + 1
        out[p] = boxes
    return out, counts


def build_round_batch(base_batch, round_dir, labels):
    if os.path.exists(round_dir):
        shutil.rmtree(round_dir)
    shutil.copytree(base_batch, round_dir, symlinks=True)
    for f in os.listdir(round_dir):  # ultralytics label caches belong to the base batch
        if f.endswith(".cache"):
            os.remove(os.path.join(round_dir, f))
    for sub in ("labels/train", "labels/val"):
        cache = os.path.join(round_dir, sub + ".cache")
        if os.path.exists(cache):
            os.remove(cache)
    for p, boxes in labels.items():
        stem = os.path.splitext(os.path.basename(p))[0]
        shutil.copy2(p, os.path.join(round_dir, "images", "train", stem + ".jpg"))
        _write_yolo_labels(os.path.join(round_dir, "labels", "train", stem + ".txt"), boxes)
    yaml_path = os.path.join(round_dir, "data.yaml")
    with open(yaml_path) as f:
        text = f.read()
    with open(yaml_path, "w") as f:
        f.write("\n".join(f"path: {round_dir}" if line.startswith("path:") else line
                          for line in text.splitlines()) + "\n")
    return yaml_path


def eval_iw_hub(weights):
    scores = {}
    for name, sub in (("lab_video_30", "fixed_testset/iw_hub"), ("roboflow_50", "fixed_testset/iw_hub_roboflow")):
        d = os.path.join(db.PIPELINE_ROOT, sub)
        gt = yolo_dir_to_coco_gt(os.path.join(d, "images"), os.path.join(d, "labels"), 0, "iw_hub")
        paths = [i["path"] for i in gt["images"]]
        scores[name] = PerImageMatches(gt, to_coco_results(predict_raw(weights, paths), gt, {5: 0})).ap50(0)
    return scores


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--frames-dir", help="already-extracted frames (e.g. capture_node output)")
    parser.add_argument("--base-batch", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--loco-teacher", default=None,
                        help="label the LOCO classes with this model instead of --teacher (first round only, "
                             "unless --loco-teacher-every-round)")
    parser.add_argument("--loco-teacher-every-round", action="store_true",
                        help="keep --loco-teacher for the LOCO classes in every round: a student that already "
                             "forgot a class labels fewer of it each round and amplifies the forgetting")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--tag", default="", help="suffix for run names / results file, keeps experiments apart")
    parser.add_argument("--batch", type=int, default=-1, help="fixed batch size (-1 = AutoBatch)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-seeds", type=int, nargs="*", default=[],
                        help="after the last round, retrain its batch with these seeds too")
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    registry = db.load_class_registry()
    names = {c["class_idx"]: c["name"] for c in registry}
    loco_names = [c["name"] for c in db.loco_classes(registry)]
    loco_idx = {c["class_idx"] for c in db.loco_classes(registry)}
    pipeline_idx = {c["class_idx"] for c in db.pipeline_classes(registry)}
    if args.frames_dir:
        frames = sorted(os.path.join(args.frames_dir, f) for f in os.listdir(args.frames_dir) if f.endswith(".jpg"))
    else:
        stem = os.path.splitext(os.path.basename(args.video))[0].replace(" ", "_")
        frames = extract_frames(args.video, os.path.join(OUT_ROOT, "frames", stem))
    suffix = f"_{args.tag}" if args.tag else ""

    results_path = os.path.join(OUT_ROOT, f"results{suffix}.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else []

    def save(entry, key):
        nonlocal results
        results = [x for x in results if (x["round"], x.get("seed", 0)) != key] + [entry]
        with open(results_path, "w") as f:
            json.dump(sorted(results, key=lambda x: (x["round"], x.get("seed", 0))), f, indent=2)

    def train_eval(data_yaml, run_name, seed):
        weights = finetune(data_yaml, run_name, batch=args.batch, seed=seed)
        return weights, {"loco": evaluate_model(weights, registry, loco_names, db.DATASET_ROOT),
                         "iw_hub": eval_iw_hub(weights)}

    teacher, data_yaml = args.teacher, None
    for r in range(args.start_round, args.start_round + args.rounds):
        run_name = f"self_training{suffix}_r{r}"
        loco_teacher = args.loco_teacher if (r == args.start_round or args.loco_teacher_every_round) else None
        print(f"\n===== round {r}: teacher {teacher}"
              f"{f' (LOCO classes from {loco_teacher})' if loco_teacher else ''} =====", flush=True)
        labels, counts = label_frames(teacher, frames, loco_teacher=loco_teacher,
                                      loco_idx=loco_idx, pipeline_idx=pipeline_idx)
        print(f"pseudo-labels on {len(frames)} frames: { {names[k]: v for k, v in sorted(counts.items())} }")
        data_yaml = build_round_batch(os.path.join(db.PIPELINE_ROOT, args.base_batch),
                                      os.path.join(OUT_ROOT, "batches", run_name), labels)
        weights, scores = train_eval(data_yaml, run_name, args.seed)
        entry = {"round": r, "seed": args.seed, "teacher": teacher, "loco_teacher": loco_teacher,
                 "weights": weights, "pseudo_labels": {names[k]: v for k, v in counts.items()}, **scores}
        save(entry, (r, args.seed))
        print(f"ROUND {r} RESULT: {json.dumps(scores)}", flush=True)
        teacher = weights

    last = args.start_round + args.rounds - 1
    for seed in args.extra_seeds:
        weights, scores = train_eval(data_yaml, f"self_training{suffix}_r{last}_s{seed}", seed)
        save({"round": last, "seed": seed, "weights": weights, **scores}, (last, seed))
        print(f"ROUND {last} SEED {seed} RESULT: {json.dumps(scores)}", flush=True)
    print("SELF TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
