"""
Runs one fine-tune round with the exact recipe already validated in
finetune_robot_frozen_mixed.py (freeze=21: backbone+neck frozen, only the
last C2f block + Detect head train; 30 epochs; imgsz 960) -- always
starting from the pristine LOCO-only baseline checkpoint, never a previous
pipeline round's weights, so drift never compounds across rounds (see
build_training_batch.py's docstring for the full rationale).

Exposed as a function (not just a __main__ block, unlike
finetune_robot_frozen_mixed.py) so auto_train_trigger.py can call it
directly.

Optional teacher_weights turns on this ultralytics build's native
feature-response knowledge distillation (ultralytics/nn/distill_model.py,
`model.train(distill_model=..., dis=...)`) against the current production
model, as an extra anti-catastrophic-forgetting lever on top of replay
sampling. It matches student vs. teacher NECK features (not classification
logits, so it doesn't care that the student has more output classes than
the teacher), weighted by the teacher's own confidence at each spatial
location -- so it pushes hard wherever the teacher recognizes something it
already knows (LOCO objects) and barely at all where it doesn't (e.g. a
novel iw_hub region). Under freeze=21 specifically, this lands almost
entirely on the one unfrozen neck layer (the last C2f block, layer 21):
the two frozen neck feature levels (15, 18) are byte-identical to the
teacher's own frozen weights already (both restart from the same pristine
baseline every round), so distillation pressure there is trivially ~0 --
layer 21 is where retraining actually happens, and therefore where
distillation actually has teeth.

Usage:
    python run_finetune.py --data path/to/data.yaml --run-name loco_finetune_pipeline_v3
    python run_finetune.py --data path/to/data.yaml --run-name loco_finetune_pipeline_v6 \
        --teacher-weights ../../runs/detect/production/best.pt
"""

import argparse
import os

import db

BASELINE = os.path.join(db.PROJECT_ROOT, "runs", "detect", "yolo_runs", "loco_full_v3", "weights", "best.pt")
YOLO_RUNS_DIR = os.path.join(db.PROJECT_ROOT, "runs", "detect", "yolo_runs")
FREEZE_LAYERS = 21
DIS_WEIGHT = 6.0  # ultralytics' own default distillation loss weight (cfg/default.yaml)
CLS_PW = 0.0  # ultralytics' own default (disabled); 1.0 = full inverse-frequency class weighting


def finetune(data_yaml, run_name, baseline=BASELINE, epochs=30, imgsz=960,
             teacher_weights=None, dis_weight=DIS_WEIGHT, freeze=FREEZE_LAYERS, cls_pw=CLS_PW,
             mask_loco_cls_grad=False):
    from ultralytics import YOLO

    if not os.path.exists(baseline):
        raise FileNotFoundError(
            f"LOCO-only baseline checkpoint not found at {baseline} -- every pipeline fine-tune "
            f"round starts from this pristine checkpoint (never a previous round's weights); "
            f"it needs to be provisioned once (it's gitignored, same as every other run/ artifact)."
        )
    if teacher_weights and not os.path.exists(teacher_weights):
        raise FileNotFoundError(f"--teacher-weights path not found: {teacher_weights}")

    model = YOLO(baseline)

    trainer_cls = None
    if mask_loco_cls_grad:
        loco_idx = sorted(c["class_idx"] for c in db.loco_classes(db.load_class_registry()))
        trainer_cls = _make_grad_mask_trainer(loco_idx)

    model.train(
        data=data_yaml,
        epochs=epochs,
        patience=10,
        batch=-1,
        imgsz=imgsz,
        cls=1.0,
        cls_pw=cls_pw,
        freeze=freeze,
        project=YOLO_RUNS_DIR,
        name=run_name,
        exist_ok=True,
        distill_model=teacher_weights,
        dis=dis_weight,
        trainer=trainer_cls,
    )
    return os.path.join(YOLO_RUNS_DIR, run_name, "weights", "best.pt")


def _make_grad_mask_trainer(loco_idx):
    """Poor-man's dual head: a DetectionTrainer subclass that zeroes the
    gradient on the Detect head's classification-branch output channels for
    every source=loco class, via a native torch.Tensor.register_hook on the
    final 1x1 conv of each detection-scale's cv3 branch.

    Must hook get_model(), not the pre-train() YOLO(baseline).model object:
    YOLO.train() always rebuilds a fresh DetectionModel sized to the new
    dataset's nc inside trainer.get_model() (self.trainer.model =
    self.trainer.get_model(...); self.model = self.trainer.model) and trains
    THAT object -- a hook registered on the baseline's own 5-class Detect
    head before calling .train() would sit on a discarded object and never
    fire. register_hook fires during backward(), independent of
    ultralytics' callback timing, and survives being wrapped by
    DistillationModel afterward (wrapping doesn't clone parameters), so
    this composes safely with both freeze=N and distill_model.

    Rationale: freeze=N still lets the one unfrozen block's classifier
    weights for the OLD classes drift toward whatever the new class's
    gradient wants (a second channel of interference on top of the
    shared-feature drift distillation already targets). This removes that
    one specific channel by construction: the LOCO channels' own weights
    literally cannot update from either loss term, they can only shift
    indirectly if the shared input features (also touched by distillation)
    change. Doesn't touch box regression (already class-agnostic in
    YOLOv8's decoupled head) or the backbone/neck.
    """
    import torch

    from ultralytics.models.yolo.detect import DetectionTrainer

    def _apply_mask(model):
        for branch in model.model[-1].cv3:  # model.model[-1] is the Detect head
            final_conv = branch[-1]  # 1x1 conv, out_channels == nc
            mask = torch.ones(final_conv.weight.shape[0], 1, 1, 1)
            mask[loco_idx] = 0.0
            final_conv.weight.register_hook(lambda grad, m=mask: grad * m.to(grad.device))
            if final_conv.bias is not None:
                bias_mask = torch.ones(final_conv.bias.shape[0])
                bias_mask[loco_idx] = 0.0
                final_conv.bias.register_hook(lambda grad, m=bias_mask: grad * m.to(grad.device))

    class _GradMaskDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            _apply_mask(model)
            return model

    return _GradMaskDetectionTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--freeze", type=int, default=FREEZE_LAYERS,
                         help="Number of leading layers to freeze (default 21: backbone+neck frozen, "
                              "only the last C2f block + Detect head train)")
    parser.add_argument("--teacher-weights", default=None,
                         help="Path to a teacher checkpoint (e.g. runs/detect/production/best.pt) to enable "
                              "knowledge distillation against it. Omit to disable (default).")
    parser.add_argument("--dis-weight", type=float, default=DIS_WEIGHT,
                         help="Distillation loss weight, only used when --teacher-weights is set (default 6.0, "
                              "ultralytics' own default)")
    parser.add_argument("--cls-pw", type=float, default=CLS_PW,
                         help="Class-weighting power for the classification loss, [0.0-1.0] (default 0.0 = "
                              "disabled, ultralytics' own default; 1.0 = full inverse-frequency weighting)")
    parser.add_argument("--mask-loco-cls-grad", action="store_true",
                         help="Zero the gradient on the Detect head's classification-branch output channels "
                              "for every source=loco class (see _mask_loco_classifier_gradients docstring)")
    args = parser.parse_args()
    weights = finetune(args.data, args.run_name, args.baseline, args.epochs, args.imgsz,
                        args.teacher_weights, args.dis_weight, args.freeze, args.cls_pw,
                        args.mask_loco_cls_grad)
    print(f"finetune complete: {weights}")


if __name__ == "__main__":
    main()
