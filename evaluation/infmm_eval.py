#!/usr/bin/env python3
"""
Evaluate Infi-MM/infimm-vicuna13b on an annotated pairs dataset (A.jpg labeled, B.jpg augmented).

For each pair:
  - Send BOTH images (A first, B second) with a prompt to Infi-MM/infimm-vicuna13b.
  - Parse the model's final output for a single uppercase letter A–Z.
  - Save a big JSON with raw responses (full generated output) and per-pair correctness.
  - Print and save an accuracy summary.

Parallelization:
  - Shards pairs across all available GPUs (one process per GPU).
  - Each process loads a single model instance on its GPU and iterates its shard.
  - CPU fallback if no GPU is available (single process).

Requirements:
  pip install -U transformers accelerate torch pillow tqdm
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any, Tuple, Any as AnyType
from dataclasses import dataclass
from tqdm import tqdm
import torch
from PIL import Image
from multiprocessing import get_context

from transformers import AutoProcessor, AutoModelForCausalLM

# ---------------------------
# Data loading
# ---------------------------
PROMPT = (
    "Here are two photos of a static scene from different views. However, in between taking the photos, "
    "I edited the image such that, for one object, its positioning is inconsistent with the camera motion "
    "between the two frames. Which letter marks the modified object? Please use the labels from the first image. "
)

@dataclass
class PairItem:
    pair_id: str
    pair_dir: str
    A_path: str
    B_path: str
    gt_letter: str  # may be None if missing
    meta: Dict[str, Any]


def load_dataset(dataset_root: str) -> List[PairItem]:
    ann_path = os.path.join(dataset_root, "annotations.json")
    with open(ann_path, "r") as f:
        anns = json.load(f)

    items: List[PairItem] = []
    for rec in anns:
        pair_id = rec["pair_id"]
        pair_dir = os.path.join(dataset_root, pair_id)
        A_path = os.path.join(pair_dir, "A.jpg")
        B_path = os.path.join(pair_dir, "B.jpg")
        if not (os.path.isfile(A_path) and os.path.isfile(B_path)):
            continue
        gt = rec.get("moved_object_letter")
        if isinstance(gt, str) and gt:
            gt = gt.strip().upper()[0]
            if not ("A" <= gt <= "Z"):
                gt = None
        else:
            gt = None
        items.append(PairItem(pair_id, pair_dir, A_path, B_path, gt, rec))
    return items

# ---------------------------
# InfiMM (Vicuna13B) inference
# ---------------------------

def build_prompts(a_path: str, b_path: str, prompt: str) -> List[Dict[str, Any]]:
    """Builds InfiMM-style chat prompts with image entries and a strict formatting instruction."""
    strict = (
        f"{prompt}\n\n"
        "Format your reply exactly as:\n"
        "Final Answer: <LETTER>\n"
        "…where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
    )
    # InfiMM expects: [{"role": "...", "content": [ <image or text entries> ]}]
    # Images can be file paths or PIL, we pass file paths here.
    return [
        {
            "role": "user",
            "content": [
                {"image": a_path},   # A (first)
                {"image": b_path},   # B (second)
                strict,
            ],
        }
    ]

FINAL_RE = re.compile(r"Final Answer:\s*([A-Z])")

def parse_letter_from_generated(generated_text: str) -> str:
    if not isinstance(generated_text, str):
        return ""
    m = FINAL_RE.search(generated_text)
    if m:
        return m.group(1)
    lines = [ln.strip() for ln in generated_text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last) == 1 and "A" <= last <= "Z":
            return last
        for ch in last:
            if "A" <= ch <= "Z":
                return ch
    for ch in generated_text:
        if "A" <= ch <= "Z":
            return ch
    return ""


from transformers import AutoProcessor, AutoModelForCausalLM

def load_model_and_processor(model_id: str, device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # Processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # IMPORTANT: disable meta-tensor init & device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,   # <-- key fix
        device_map=None            # <-- ensure not loaded to "meta"
    )

    # Move after weights are materialized
    model.to(device).eval()
    return model, processor, dtype


def run_inference(
    items: List[PairItem],
    model_id: str,
    device: str,
    prompt: str,
    max_new_tokens: int = 8,
) -> List[Dict[str, Any]]:
    model, processor, dtype = load_model_and_processor(model_id, device)
    results: List[Dict[str, Any]] = []

    for it in tqdm(items, desc=f"{device}"):
        # Light validation that the images are readable
        try:
            Image.open(it.A_path).convert("RGB")
            Image.open(it.B_path).convert("RGB")
        except Exception as e:
            results.append({
                "pair_id": it.pair_id,
                "error": f"Image open failed: {repr(e)}",
                "pred_letter": "",
                "gt_letter": it.gt_letter,
                "correct": False
            })
            continue

        # 1) Build InfiMM chat prompt with two images (A then B)
        prompts = build_prompts(it.A_path, it.B_path, prompt)

        # 2) Tokenize/prepare inputs via processor (handles images from paths)
        inputs = processor(prompts)

        # 3) Move to the same device as the model; cast images to bf16/fp16 when present
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "batch_images" in inputs and hasattr(inputs["batch_images"], "to"):
            # InfiMM example uses bf16 for images
            target_img_dtype = torch.bfloat16 if dtype in (torch.bfloat16, torch.float16) else torch.float32
            inputs["batch_images"] = inputs["batch_images"].to(target_img_dtype)

        # 4) Generate (InfiMM uses custom args in trust_remote_code)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                min_generation_length=0,
                max_generation_length=int(max_new_tokens),
            )

        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_texts[0] if generated_texts else ""

        pred = parse_letter_from_generated(generated_text)
        correct = (pred == it.gt_letter) if it.gt_letter else False
        results.append({
            "pair_id": it.pair_id,
            "pred_letter": pred,
            "gt_letter": it.gt_letter,
            "correct": bool(correct),
            "raw_text": generated_text,
            "a_path": it.A_path,
            "b_path": it.B_path,
        })
    return results

# ---------------------------
# Sharding & Multiprocessing
# ---------------------------

def shard(lst: List[AnyType], n: int) -> List[List[AnyType]]:
    if n <= 1:
        return [lst]
    return [lst[i::n] for i in range(n)]

def _worker(args) -> List[Dict[str, Any]]:
    shard_items, model_id, device, prompt, max_new_tokens = args
    return run_inference(
        items=shard_items,
        model_id=model_id,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

# ---------------------------
# Accuracy & I/O
# ---------------------------

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in results if r.get("gt_letter")]
    if not evaluated:
        return {"num_pairs": len(results), "num_evaluated": 0, "accuracy": None}
    correct = sum(1 for r in evaluated if r.get("correct"))
    acc = correct / max(1, len(evaluated))
    return {
        "num_pairs": len(results),
        "num_evaluated": len(evaluated),
        "num_correct": correct,
        "accuracy": acc
    }

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Infi-MM/infimm-vicuna13b on annotated pairs (A,B)."
    )
    ap.add_argument(
        "--dataset-root", default="/nfs_share4/code/om/hypersim/3d_eval_FULL",
        help="Root folder containing pair subfolders + annotations.json"
    )
    ap.add_argument(
        "--model-id", default="Infi-MM/infimm-vicuna13b",
        help="HF model id for InfiMM Vicuna-13B"
    )
    ap.add_argument(
        "--output-dir", default="/nfs_share4/code/om/hypersim/results_eval_full",
        help="Directory where to save results and summary with model name"
    )
    ap.add_argument(
        "--prompt", default=PROMPT,
        help="Text instruction sent with A and B"
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=8,
        help="Max new tokens for generation"
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root
    items = load_dataset(dataset_root)
    if not items:
        raise SystemExit(f"No pairs found under {dataset_root}")

    # Compute output paths including sanitized model name
    os.makedirs(args.output_dir, exist_ok=True)
    model_stamp = args.model_id.replace("/", "_")
    out_results = os.path.join(
        args.output_dir,
        f"{model_stamp}_results.json"
    )
    out_summary = os.path.join(
        args.output_dir,
        f"{model_stamp}_summary.json"
    )

    # GPU detection
    try:
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0

    if num_gpus <= 0:
        print("No GPUs found. Running on CPU…")
        all_results = run_inference(
            items=items,
            model_id=args.model_id,
            device="cpu",
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        shards = shard(items, num_gpus)
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_gpus) as pool:
            mapped = pool.map(
                _worker,
                [
                    (
                        shards[i], args.model_id, devices[i],
                        args.prompt, args.max_new_tokens
                    )
                    for i in range(num_gpus)
                ],
                chunksize=1
            )
        all_results = [r for sub in mapped for r in sub]

    # Save big JSON
    with open(out_results, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summarize & save
    summary = summarize(all_results)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved per-pair results to: {out_results}")
    print(f"Saved summary to: {out_summary}")

if __name__ == "__main__":
    main()
