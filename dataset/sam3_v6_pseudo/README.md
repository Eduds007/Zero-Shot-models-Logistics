---
pretty_name: LOCO SAM3 v6 Pseudo Labels
tags:
  - object-detection
  - pseudo-labels
  - logistics
  - sam3
---

# LOCO SAM3 v6 Pseudo Labels

Pseudo-labeled object detection dataset for 5 logistics object classes (small load
carrier, forklift, pallet, stillage, pallet truck), generated automatically from a
2fps video of a logistics/lab facility.

**These are NOT human-verified ground-truth annotations.** Every box comes from
SAM3 (`facebook/sam3`, zero-shot, text-prompted), run with the exact configuration
of run `sam3_v6` in this project's benchmark
(`dataset/results/loco_eval_results.csv`). Treat this as a pseudo-labeled / weakly
supervised dataset, not a validated benchmark.

## Source video and frame filtering

- Source: a 2fps-extracted frame sequence of a single walkthrough video
  (`assets/frames_2fps/`, 156 frames total).
- **Blur filter**: frames with Laplacian variance < 120 were dropped as
  motion-blurred before running the model at all (13 frames dropped).
- **Confidence filter**: of the remaining frames, only those whose detections
  (pooled across all 5 categories) averaged a confidence score > 0.5 were kept
  (15 more frames dropped, mostly frames with zero detections at all).
- **128 of the original 156 frames** made it into this dataset.

## Labeling method (sam3_v6 config)

For each kept frame, SAM3 was prompted once per category with:

| class_id | class name | prompt | confidence_threshold |
|---|---|---|---|
| 0 | small_load_carrier | "plastic crate" | 0.5 |
| 1 | forklift | "forklift" | 0.05, then capped to the top-2 highest-score detections |
| 2 | pallet | "pallet" | 0.5 |
| 3 | stillage | "wire mesh crate" | 0.5 |
| 4 | pallet_truck | "pallet jack" | 0.5 |

The forklift threshold was deliberately lowered and capped (rather than left at
0.5) because at 0.5 SAM3 found almost no forklifts in this project's LOCO
benchmark; see run `sam3_v6` in `dataset/results/loco_eval_results.csv` for the
zero-shot accuracy numbers this config produces on the (human-annotated) LOCO
dataset itself, for context on how reliable these classes are.

## Format

Standard YOLO detection format, one label file per image, no train/val split:

```
images/photo_XXXXX.jpg
labels/photo_XXXXX.txt   # one line per box: <class_id> <x_center> <y_center> <width> <height>, all normalized [0,1]
```

## Class distribution (128 images, 2253 boxes)

| class_id | class name | boxes |
|---|---|---|
| 0 | small_load_carrier | 1239 |
| 1 | forklift | 173 |
| 2 | pallet | 686 |
| 3 | stillage | 40 |
| 4 | pallet_truck | 115 |

## License

Not yet specified by the dataset owner.
