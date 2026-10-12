# Continual-learning pipeline (FML Tugger)

The robot captures frames, the workstation finds a new class in them with
SAM3, a human reviews the uncertain ones, and a single YOLO model is
retrained so it learns the new class without forgetting the LOCO ones. A
model is only deployed if it passes a statistical non-regression gate; the
robot then picks it up by itself.

```
ROBOT (fml-22, ~/ros2nav_ws/src)            WORKSTATION (this repo)
capture_node ─► capture/pending/
sync_node ──────rsync──────────────────►  dataset/incoming/<robot>/
                                          run_pipeline_cycle.py (hourly cron):
                                            1. ingest_from_robot.py
                                            2. sam3_quality_gate.py ─► approved_pending_training/ (clear)
                                                                    └► staging_for_review/       (uncertain)
review_sync_node ◄───────rsync────────────  staging_for_review/
review_node (Lichtblick: approve/reject)
review_sync_node ───────rsync──────────►  review_decisions/ + apply_synced_decisions.py
                                            3. auto_train_trigger.py (>= 50 new approvals):
                                               build_training_batch ─► run_finetune x3 seeds
                                               ─► promote_if_better (95% CI gate)
model_sync_node ◄────────rsync────────────  runs/detect/production/{best.pt,registry.json}
inference_node (hot-swaps the model) ─► perception/detections_image
```

## Training recipe (build_training_batch.py)

Every round restarts from the pristine LOCO model
(`runs/detect/yolo_runs/loco_full_v3`, YOLOv8m, `freeze=21`) and trains on:

| part | labels |
|---|---|
| LOCO replay (10 LOCO images per pipeline image, rare classes guaranteed) | human (LOCO) |
| approved frames of each pipeline class (up to 150 per class) | new class: SAM3 + review; LOCO classes: production model |
| copy-paste of each pipeline class onto LOCO replay images | inherited |
| environment frames (captures SAM3 rejected for every pipeline class, up to 1000) | LOCO classes: production model |
| LOCO bank: 5000 unannotated LOCO images from the train warehouses 2, 3, 5 | per class `pseudo_label_source` in `class_registry.json`: SAM3 (small_load_carrier, stillage, pallet_truck) or production (forklift, pallet) |

All pseudo-labels use a fixed confidence of 0.5 (`pseudo_labels.py`).
3 seeds, batch 8, 30 fixed epochs, `last.pt` (no early stopping: the val
split holds images the base model was trained on). `assert_no_test_leak()`
aborts a batch that contains any test image. Why this recipe:
`CATASTROPHIC_FORGETTING_RESULTS.md`.

## Promotion gate (promote_if_better.py)

- LOCO classes: scored on the official LOCO val split (2277 images,
  warehouses 1 and 4 -- never seen in training). A class fails only if the
  95% paired-bootstrap CI of AP50(candidate) − AP50(production) lies
  entirely below −0.01.
- Pipeline classes: scored on `test_sets` in `class_registry.json`
  (`fixed_testset/iw_hub` lab frames + Roboflow images); a first promotion
  needs AP50 ≥ 0.10.
- The best passing seed (mean AP50) is promoted; `--dry-run` never touches
  production.

Known status: the recipe keeps forklift, pallet, small_load_carrier and
iw_hub, but pallet_truck and stillage still fail the gate, so rounds don't
promote yet.

## Running it

Workstation cron (installed for go39sin; `crontab -l`):

```
0 * * * * cd $HOME/Desktop/Zero-Shot-models-Logistics/dataset/pipeline && PYTORCH_ALLOC_CONF=expandable_segments:True ../../.venv/bin/python run_pipeline_cycle.py >> logs/pipeline_cycle.log 2>&1
```

Each step is a no-op when there's nothing to do. The GPU is shared: if SAM3
runs out of memory the cycle logs it and the next hour resumes (progress is
committed every 25 frames). Manual runs:

```
../../.venv/bin/python run_pipeline_cycle.py [--threshold 50] [--dry-run]
../../.venv/bin/python promote_if_better.py --run-name X --weights a.pt b.pt c.pt --dry-run
```

Robot (`ssh workstation` must work: `~/.ssh/config` Host `workstation` with
HostName/User, key via ssh-copy-id -- all nodes default to that alias):

```
ros2 launch tug_bringup_ros2 robot.launch.py           # camera, motors, foxglove bridge
ros2 launch dataset_capture capture.launch.py          # capture + upload
ros2 launch dataset_capture review.launch.py           # review queue in Lichtblick
ros2 launch logistics_perception inference.launch.py   # live detection + model pulls
```

## Adding a new class

Add an entry with `"source": "pipeline"`, a SAM3 `prompt`, `replay_dir`
and `test_sets` to `class_registry.json`; captures are checked against it
from the next cycle on. It also needs a fixed test set (a slice of its
approved frames carved out of training, like `fixed_testset/iw_hub`) before
its first promotion -- `build_fixed_testset.py` does this for iw_hub only and
would need generalizing.
