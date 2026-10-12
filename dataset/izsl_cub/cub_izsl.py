"""
Side experiment: our incremental-detection recipe on the CUB benchmark of
Wei et al., "Incremental Zero-Shot Learning" (IEEE T-Cybernetics 2022),
reported with their u / s / H so it lines up with their Table II.

Protocol (theirs): the standard ZSL "proposed split" of CUB (xlsa17: 150
seen, 50 unseen classes); the seen classes split into 5 steps of 30 by label
order; u / s = per-class mean accuracy on test_unseen / test_seen with the
generalized (seen + unseen) label set, measured after the last step;
H = 2us / (u + s). Average forgetting as in their Table IV.

Our method, adapted (detection, since CUB has one box per bird):
  - step 1 = the base model: YOLOv8m (COCO-pretrained) fully trained on the
    first 30 classes, like the LOCO baseline;
  - steps 2-5 = pipeline rounds: each restarts from the step-1 model with
    the pipeline's run_finetune recipe (freeze=21, 30 fixed epochs, last.pt,
    batch 8, seed 0) on the new classes plus replay of the old ones:
      full  -- the pipeline's rule (up to 150 images per class = all of CUB)
      k5    -- 5 stored images per old class
      zero  -- no stored images (the setting of the Table II methods)
  - unseen classes: SAM3 with each unseen species name as text prompt;
  - prediction per test image: the top YOLO detection if its confidence is
    >= 0.5 (the robot's operating threshold), else SAM3's best unseen
    class, else wrong. Nothing is tuned on the test set.

Old-class pseudo-labels don't apply: every CUB image holds a single bird.

Usage:
    python cub_izsl.py prepare | train | sam3 | evaluate
"""

import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))

CUB = os.path.join(HERE, "CUB_200_2011")
SPLITS = os.path.join(HERE, "xlsa17", "data", "CUB")
WORK = os.path.join(HERE, "work")
STEPS = 5
VARIANTS = {"full": 150, "k5": 5, "zero": 0}
GATE_CONF = 0.5
IMGSZ = 640
BATCH = 8
SEED = 0
BASE_EPOCHS = 100
SAM3_FLOOR = 0.05


# ------------------------------------------------------------------ data

def load_meta():
    path = os.path.join(WORK, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    import scipy.io
    from PIL import Image

    res = scipy.io.loadmat(os.path.join(SPLITS, "res101.mat"))
    att = scipy.io.loadmat(os.path.join(SPLITS, "att_splits.mat"))
    files = ["images/" + str(f[0][0]).split("/images/")[-1] for f in res["image_files"]]
    labels = res["labels"].ravel().astype(int)  # 1..200, CUB class ids
    names = [str(n[0][0]) for n in att["allclasses_names"]]  # index = class id - 1

    with open(os.path.join(CUB, "images.txt")) as f:
        id_by_path = {line.split()[1]: line.split()[0] for line in f}
    with open(os.path.join(CUB, "bounding_boxes.txt")) as f:
        box_by_id = {p[0]: [float(x) for x in p[1:]] for p in (line.split() for line in f)}

    def entries(loc_key):
        out = []
        for i in att[loc_key].ravel().astype(int) - 1:
            rel = files[i]
            x, y, w, h = box_by_id[id_by_path[rel[len("images/"):]]]
            iw, ih = Image.open(os.path.join(CUB, rel)).size
            out.append({"path": os.path.join(CUB, rel), "cls": int(labels[i]),
                        "box": [(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih]})
        return out

    trainval, test_seen, test_unseen = entries("trainval_loc"), entries("test_seen_loc"), entries("test_unseen_loc")
    seen = sorted({e["cls"] for e in trainval})
    unseen = sorted({e["cls"] for e in test_unseen})
    assert len(seen) == 150 and len(unseen) == 50 and not set(seen) & set(unseen)
    chunk = len(seen) // STEPS
    steps = [seen[i * chunk:(i + 1) * chunk] for i in range(STEPS)]  # "by label order"
    meta = {"names": {str(c): names[c - 1] for c in range(1, 201)}, "steps": steps, "unseen": unseen,
            "trainval": trainval, "test_seen": test_seen, "test_unseen": test_unseen}
    os.makedirs(WORK, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f)
    return meta


def yolo_index(meta):
    """Seen class id -> detector index, in step order (stable across steps)."""
    order = [c for step in meta["steps"] for c in step]
    return {c: i for i, c in enumerate(order)}


def build_step(meta, t, variant):
    """Train set of step t (1-based): all images of the step's classes plus
    replay of the older classes per the variant."""
    idx = yolo_index(meta)
    known = [c for step in meta["steps"][:t] for c in step]
    by_cls = {}
    for e in meta["trainval"]:
        by_cls.setdefault(e["cls"], []).append(e)
    rng = random.Random(SEED)
    chosen = [e for c in meta["steps"][t - 1] for e in by_cls[c]]
    if t > 1:
        cap = VARIANTS[variant]
        for c in known[:-len(meta["steps"][t - 1])]:
            pool = sorted(by_cls[c], key=lambda e: e["path"])
            chosen += rng.sample(pool, min(cap, len(pool)))
    d = os.path.join(WORK, "batches", "base" if t == 1 else f"{variant}_step{t}")
    for sub in ("images/train", "labels/train"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for e in chosen:
        stem = os.path.splitext(os.path.basename(e["path"]))[0]
        img = os.path.join(d, "images/train", stem + ".jpg")
        if not os.path.lexists(img):
            os.symlink(e["path"], img)
        with open(os.path.join(d, "labels/train", stem + ".txt"), "w") as f:
            f.write(f"{idx[e['cls']]} " + " ".join(f"{v:.6f}" for v in e["box"]) + "\n")
    yaml = os.path.join(d, "data.yaml")
    with open(yaml, "w") as f:
        # val = train: only for ultralytics' logs -- epochs are fixed and last.pt is used
        f.write(f"path: {d}\ntrain: images/train\nval: images/train\n")
        f.write("names: " + json.dumps({idx[c]: meta["names"][str(c)] for c in known}) + "\n")
    print(f"{os.path.basename(d)}: {len(chosen)} images, {len(known)} classes", flush=True)
    return yaml


# ------------------------------------------------------------------ train

def run_name(variant, t):
    return "cub_izsl_base" if t == 1 else f"cub_izsl_{variant}_step{t}"


def weights_for(variant, t):
    from run_finetune import YOLO_RUNS_DIR

    return os.path.join(YOLO_RUNS_DIR, run_name(variant, t), "weights", "last.pt")


def train(meta):
    from ultralytics import YOLO

    import run_finetune

    if not os.path.exists(weights_for("full", 1)):
        coco_pretrained = os.path.join(run_finetune.db.DATASET_ROOT, "yolov8m.pt")
        YOLO(coco_pretrained).train(
            data=build_step(meta, 1, "full"), epochs=BASE_EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
            patience=0, project=run_finetune.YOLO_RUNS_DIR, name=run_name("full", 1), exist_ok=True)
    for variant in VARIANTS:
        for t in range(2, STEPS + 1):
            if not os.path.exists(weights_for(variant, t)):
                run_finetune.finetune(build_step(meta, t, variant), run_name(variant, t),
                                      baseline=weights_for("full", 1), imgsz=IMGSZ, batch=BATCH, seed=SEED,
                                      patience=0)
            print(f"TRAINED {variant} step {t}", flush=True)


# ------------------------------------------------------------------ predict

def yolo_top(meta, weights, entries):
    """[(class id or None, confidence)] of the top detection per image."""
    from ultralytics import YOLO

    model = YOLO(weights)
    inv = {i: c for c, i in yolo_index(meta).items()}
    out = []
    paths = [e["path"] for e in entries]
    for i in range(0, len(paths), 32):
        for r in model.predict(paths[i:i + 32], conf=0.001, imgsz=IMGSZ, verbose=False):
            if len(r.boxes):
                j = int(r.boxes.conf.argmax())
                out.append((inv[int(r.boxes.cls[j])], float(r.boxes.conf[j])))
            else:
                out.append((None, 0.0))
    return out


def sam3_unseen(meta):
    """Best unseen class + score per test image, cached."""
    path = os.path.join(WORK, "sam3_unseen.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=SAM3_FLOOR)
    prompts = {c: meta["names"][str(c)].split(".", 1)[-1].replace("_", " ").lower() for c in meta["unseen"]}
    todo = [e["path"] for e in meta["test_seen"] + meta["test_unseen"] if e["path"] not in cache]
    for i, p in enumerate(todo):
        state = processor.set_image(Image.open(p).convert("RGB"))
        best = (None, 0.0)
        for c, prompt in prompts.items():
            state = processor.set_text_prompt(prompt=prompt, state=state)
            scores = state["scores"]
            if len(scores) and float(scores.max()) > best[1]:
                best = (c, float(scores.max()))
        cache[p] = best
        if (i + 1) % 200 == 0 or i + 1 == len(todo):
            with open(path, "w") as f:
                json.dump(cache, f)
            print(f"[sam3 {i + 1}/{len(todo)}]", flush=True)
    return cache


# ------------------------------------------------------------------ metrics

def per_class_acc(entries, preds):
    hits = {}
    for e, p in zip(entries, preds):
        hits.setdefault(e["cls"], []).append(p == e["cls"])
    return float(np.mean([np.mean(v) for v in hits.values()])) * 100


def evaluate(meta):
    sam3 = json.load(open(os.path.join(WORK, "sam3_unseen.json")))
    report = {"paper_table_ii_cub": {"ours_paper": [38.12, 45.63, 41.54], "SFT": [34.34, 34.06, 34.19],
                                     "EWC": [36.58, 34.79, 35.67], "JL": [43.70, 57.70, 49.70]}}
    for variant in VARIANTS:
        # per-step seen accuracy per task (YOLO alone), for the forgetting of their Table IV
        acc = {}  # acc[(l, j)] = accuracy on task j after step l
        for l in range(1, STEPS + 1):
            top = yolo_top(meta, weights_for(variant, l), meta["test_seen"])
            for j in range(1, l + 1):
                task = set(meta["steps"][j - 1])
                sel = [(e, p) for e, (p, _c) in zip(meta["test_seen"], top) if e["cls"] in task]
                acc[(l, j)] = per_class_acc([e for e, _ in sel], [p for _, p in sel])
        f_k = [max(acc[(l, j)] - acc[(STEPS, j)] for l in range(j, STEPS)) for j in range(1, STEPS)]

        final = weights_for(variant, STEPS)
        def gzsl(entries):
            preds = []
            for e, (c, conf) in zip(entries, yolo_top(meta, final, entries)):
                if c is not None and conf >= GATE_CONF:
                    preds.append(c)
                else:
                    preds.append(sam3[e["path"]][0])
            return per_class_acc(entries, preds)
        u, s = gzsl(meta["test_unseen"]), gzsl(meta["test_seen"])
        h = 2 * u * s / (u + s) if u + s else 0.0
        seen_mean = np.mean([acc[(STEPS, j)] for j in range(1, STEPS + 1)])
        report[variant] = {"u": u, "s": s, "H": h, "seen_mean_acc_yolo": float(seen_mean),
                           "avg_forgetting": float(np.mean(f_k)),
                           "per_step_task_acc": {f"{l}->{j}": a for (l, j), a in acc.items()}}
        print(f"RESULT {variant}: u={u:.2f} s={s:.2f} H={h:.2f} | seen acc {seen_mean:.2f} "
              f"forgetting {np.mean(f_k):.2f}", flush=True)
    with open(os.path.join(WORK, "results.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    {"prepare": lambda m: print(f"steps: {[len(s) for s in m['steps']]}, unseen {len(m['unseen'])}, "
                                f"trainval {len(m['trainval'])}, test_seen {len(m['test_seen'])}, "
                                f"test_unseen {len(m['test_unseen'])}"),
     "train": train, "sam3": sam3_unseen, "evaluate": evaluate}[sys.argv[1]](load_meta())
