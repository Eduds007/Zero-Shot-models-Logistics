"""
Evaluate NVIDIA LocateAnything-3B (a generative vision-language grounding
model, transformers, trust_remote_code) on the same LOCO sample/GT/CSV
pipeline as eval_loco.py, so results are directly comparable to the
SAM3/Grounding DINO/OWLv2 runs.

Unlike those detectors, LocateAnything-3B has no importable convenience
class (its model card shows a "LocateAnythingWorker" as copy-paste example
code, not a package) and is generative: it answers in free text containing
"<box><x1><y1><x2><y2></box>" tokens (coordinates normalized to [0,1000]),
with NO per-box confidence score anywhere in its documented output. Every
parsed box here is therefore given a placeholder score of 1.0 -- AP_50/
AP_50-95/AP_75/AP_95 in the resulting CSV rows are NOT a true ranked
precision-recall-curve metric (there's no score to rank by), they collapse
to precision/recall at a single fixed operating point. This is spelled out
in the --message text used for the real run; don't reuse these AP numbers
as if they were directly comparable to sam3/gdino/owlv2's real ranked AP.

Also unlike Grounding DINO/OWLv2's single combined text query per image,
this prompts ONE CATEGORY AT A TIME (ground_multi() per LOCO category, 5
calls/image) -- the model card doesn't document how a multi-category answer
would disambiguate which box belongs to which category, so per-category
calls sidestep that ambiguity entirely (mirrors eval_loco.py's own SAM3
per-category loop). Slower (5x redundant image encoding per image) but
correct.

Smoke-tested with do_sample=False (greedy) first, for reproducibility -- but
that caused a real, reproducible failure mode in this model's custom
generate() implementation: after emitting a couple of plausible boxes it gets
stuck and repeats one degenerate near-full-image box hundreds of times until
max_new_tokens is exhausted (confirmed via raw-output inspection; a classic
greedy-decoding repetition trap, not fixed by repetition_penalty=1.1 alone).
Switched to do_sample=True + temperature/top_p (the model card's own demo
defaults) so decoding can escape the loop, plus a belt-and-suspenders dedup
in parse_boxes() that drops a box once the identical (x1,y1,x2,y2) has
already appeared, in case sampling still degenerates occasionally.

Usage:
    python eval_loco_locate_anything.py --sample-list /tmp/loco_sample_paths.txt \
        --message "..."
"""

import argparse
import os
import re
import time

import torch
from PIL import Image

from eval_loco import CATEGORY_PROMPTS, build_gt_subset, evaluate


def parse_boxes(answer, image_width, image_height):
    """From the model card: <box><x1><y1><x2><y2></box>, ints normalized to
    [0,1000] -> pixel xyxy. Drops a box once the identical raw (x1,y1,x2,y2)
    has already been seen -- guards against the degenerate-repeat failure
    mode described in the module docstring (a stuck decode repeating one
    box hundreds of times)."""
    boxes = []
    seen_raw = set()
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        raw = m.groups()
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        x1, y1, x2, y2 = [int(g) for g in raw]
        boxes.append((
            x1 / 1000 * image_width, y1 / 1000 * image_height,
            x2 / 1000 * image_width, y2 / 1000 * image_height,
        ))
    return boxes


def ground_multi(model, tokenizer, processor, image, phrase, max_new_tokens, device, dtype):
    """One category, one image -> raw generated answer text."""
    prompt = f"Locate all the instances that match the following description: {phrase}."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        response = model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="hybrid",
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
    return response[0] if isinstance(response, tuple) else response


def run_predictions(gt, dataset_root, model_id, max_new_tokens):
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    torch.manual_seed(0)  # do_sample=True is stochastic; seed for run-to-run repeatability
    device = "cuda"
    dtype = torch.bfloat16

    print(f"Loading {model_id} (trust_remote_code=True)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # The vision encoder (MoonViT) correctly falls back flash_attention_2 -> sdpa
    # on its own if flash_attn isn't available, and its flash_attention_2 path
    # works fine here (~7.8GB after load, vs. OOM without flash-attn). But the
    # *text* model's forward() (modeling_qwen2.py) only implements "magi" (Hopper/
    # Blackwell only) and "sdpa" -- it raises NotImplementedError if flash_attn is
    # installed and auto-selection picks "flash_attention_2" for it (a real gap in
    # this custom modeling code, not something we're choosing to avoid). Force sdpa
    # for text explicitly; leave vision_config alone so it keeps its own working
    # flash_attention_2 default.
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.text_config._attn_implementation = "sdpa"

    model = AutoModel.from_pretrained(
        model_id, config=config, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    print(f"Loaded. VRAM allocated after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

    results = []
    start_time = time.perf_counter()
    for i, img_meta in enumerate(gt["images"]):
        image_path = os.path.join(dataset_root, img_meta["path"].lstrip("/"))
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        n_this_image = 0
        for cat_name in cat_id_to_name.values():
            phrase = CATEGORY_PROMPTS.get(cat_name, cat_name.replace("_", " "))
            answer = ground_multi(model, tokenizer, processor, image, phrase,
                                   max_new_tokens, device, dtype)
            for x0, y0, x1, y1 in parse_boxes(answer, w, h):
                results.append({
                    "image_id": img_meta["id"],
                    "category_id": name_to_cat_id[cat_name],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": 1.0,
                })
                n_this_image += 1

        print(f"[{i + 1}/{len(gt['images'])}] {img_meta['path']}: {n_this_image} detections")

    torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start_time
    return results, elapsed_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default="/tmp/loco_sample_paths.txt")
    parser.add_argument("--annotations", default="annotations/loco-all-v1.json")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--model-id", default="nvidia/LocateAnything-3B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--results-csv", default="results/loco_eval_results_zeroshot.csv")
    parser.add_argument("--run", default=None,
                         help="Run name, e.g. 'locateanything_v1'. Auto-incremented if omitted.")
    parser.add_argument(
        "--message", required=True,
        help="Required. What does this run represent? Should document the score=1.0 "
             "placeholder caveat -- see module docstring.",
    )
    args = parser.parse_args()

    print(f"=== {args.message} ===\n")

    with open(args.sample_list) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    gt = build_gt_subset(args.annotations, image_paths)
    print(f"Evaluating on {len(gt['images'])} images, {len(gt['annotations'])} GT boxes")

    results, elapsed_sec = run_predictions(gt, args.dataset_root, args.model_id, args.max_new_tokens)
    print(f"Inference: {elapsed_sec:.1f}s total, {elapsed_sec / len(gt['images']):.3f}s/image")

    import eval_loco
    eval_loco.MODEL_NAME = (
        f"{args.model_id} (transformers, trust_remote_code, vision=flash_attention_2 "
        f"+ text=sdpa, do_sample=True/temp=0.7/top_p=0.9/rep_penalty=1.1/seed=0, "
        f"5x ground_multi() calls/image [1/category], max_new_tokens={args.max_new_tokens}, "
        f"dedup identical boxes, score=1.0 placeholder -- no real confidence exposed)"
    )
    evaluate(gt, results, args.message, elapsed_sec, run_name=args.run,
             run_prefix="locateanything", csv_path=args.results_csv)


if __name__ == "__main__":
    main()
