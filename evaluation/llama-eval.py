#!/usr/bin/env python3
"""
Evaluate Llama 3.2 Vision (Instruct) on an annotated pairs dataset (A.jpg labeled, B.jpg augmented).

For each pair:
  - Send BOTH images (A first, B second) with a prompt to Llama 3.2 Vision (Instruct).
  - Parse the model's final output for a single uppercase letter A–Z.
  - Save a big JSON with raw responses (full generated output) and per-pair correctness.
  - Print and save an accuracy summary.

Parallelization:
  - Shards pairs across all available GPUs (one process per GPU).
  - Each process loads a single model instance on its GPU and iterates its shard.
  - CPU fallback if no GPU is available (single process).

Requirements:
  pip install -U "transformers>=4.43" accelerate torch pillow tqdm
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

from transformers import AutoProcessor, AutoModelForVision2Seq

# ---------------------------
# Data loading
# ---------------------------
PROMPT = (
    "Here are two photos of a static scene from different views. However, in between taking the photos, "
    "I edited the image such that, for one object, its positioning is inconsistent with the camera motion "
    "between the two frames. Which letter marks the modified object? Please use the labels from the first image."
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
# Llama 3.2 Vision inference
# ---------------------------

def build_messages(prompt: str, imgA: Image.Image, imgB: Image.Image) -> List[Dict[str, Any]]:
    strict = (
        f"{prompt}\n\n"
        "Format your reply exactly as:\n"
        "Final Answer: <LETTER>\n"
        "…where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
    )
    # Llama 3.2 Vision Instruct supports the same multi-turn chat template via processor.apply_chat_template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": imgA},
                {"type": "image", "image": imgB},
                {"type": "text",  "text": strict},
            ],
        }
    ]
    return messages

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


def load_model_and_processor(model_id: str, device: str):
    # Prefer bf16 on capable CUDA, else fp16 on CUDA, else fp32 on CPU
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    # Llama 3.2 Vision works with AutoProcessor and AutoModelForVision2Seq in transformers >= 4.43
    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # device_map can be "auto", "cpu", or "cuda:IDX". We pass through the user/device pipeline.
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, processor


def run_inference(
    items: List[PairItem],
    model_id: str,
    device: str,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0
) -> List[Dict[str, Any]]:
    model, processor = load_model_and_processor(model_id, device)
    results: List[Dict[str, Any]] = []

    for it in tqdm(items, desc=f"{device}"):
        try:
            imgA = Image.open(it.A_path).convert("RGB")
            imgB = Image.open(it.B_path).convert("RGB")
        except Exception as e:
            results.append({
                "pair_id": it.pair_id,
                "error": f"Image open failed: {repr(e)}",
                "pred_letter": "",
                "gt_letter": it.gt_letter,
                "correct": False
            })
            continue

        messages = build_messages(prompt, imgA, imgB)
        # Apply the chat template for Llama 3.2 Vision
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Important: pass BOTH images in the same order as in messages
        inputs = processor(
            text=prompt_text,
            images=[imgA, imgB],
            return_tensors="pt"
        ).to(model.device)

        input_len = int(inputs["input_ids"].shape[1])

        # Generation: standard greedy or sampling depending on temperature
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                repetition_penalty=1.0,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        gen_ids = outputs[0, input_len:]
        generated_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

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
    shard_items, model_id, device, prompt, max_new_tokens, temperature = args
    return run_inference(
        items=shard_items,
        model_id=model_id,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
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
        description="Evaluate Llama 3.2 Vision (Instruct) on annotated pairs (A,B)."
    )
    ap.add_argument(
        "--dataset-root", default="/nfs_share4/code/om/hypersim/3d_eval_FULL",
        help="Root folder containing pair subfolders + annotations.json"
    )
    ap.add_argument(
        "--model-id",
        default="/nfs_share4/code/om/Llama3.2-Vision-Finetune/output/ft_on_our_eval",
        help="HF model id for Llama 3.2 Vision Instruct (e.g., meta-llama/Llama-3.2-11B-Vision-Instruct)"
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
        "--max-new-tokens", type=int, default=64,
        help="Max new tokens for generation"
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy)"
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
            temperature=args.temperature
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
                        args.prompt, args.max_new_tokens, args.temperature
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
