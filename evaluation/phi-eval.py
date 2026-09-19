#!/usr/bin/env python3
"""
Evaluate microsoft/Phi-3.5-vision-instruct on an annotated pairs dataset (A.jpg labeled, B.jpg augmented).

For each pair:
  - Send BOTH images (A first, B second) with a prompt to the model.
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
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import torch
from PIL import Image
from multiprocessing import get_context

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)

AnyType = Any

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
# Chat messages (Phi-3.5 format)
# ---------------------------

def build_messages(prompt: str, num_images: int = 2) -> List[Dict[str, Any]]:
    """
    Build Phi-3.5-vision-instruct style chat messages.

    Uses <|image_1|> and <|image_2|> markers in the user content.
    The images list passed to the processor must match this order.
    """
    strict = (
        f"{prompt}\n\n"
        "Format your reply exactly as:\n"
        "Final Answer: <LETTER>\n"
        "…where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
    )

    placeholder = "".join(f"<|image_{i+1}|>\n" for i in range(num_images))
    user_content = placeholder + strict

    messages = [
        {
            "role": "user",
            "content": user_content,
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

# ---------------------------
# Model loading for Phi-3.5-vision-instruct
# ---------------------------

def _device_map_from_str(device: str):
    """
    Convert a simple device string to a device_map suitable for HF loading.

    device: "cpu" | "cuda" | "cuda:0" | "cuda:1" ...
    """
    d = (device or "").lower()
    if d == "cpu":
        return "cpu"
    if d.startswith("cuda"):
        idx = 0
        if ":" in d:
            try:
                idx = int(d.split(":", 1)[1])
            except Exception:
                idx = 0
        return {"": idx}
    return "auto"


def load_model_and_processor(model_id: str, device: str):
    """
    Load Phi-3.5-vision-instruct with its AutoProcessor.

    Uses AutoModelForCausalLM + AutoProcessor.
    Respects the given device (cpu / cuda:N).
    No flash attention, no custom cache_implementation tweaks.
    """
    use_cuda = (device != "cpu") and torch.cuda.is_available()
    dtype = (
        torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_cuda else torch.float32)
    )
    device_map = _device_map_from_str(device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Disable KV cache usage to avoid DynamicCache-related issues
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.use_cache = False
        except Exception:
            pass

    model.eval()
    return model, processor

# ---------------------------
# Inference
# ---------------------------

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

        # Build Phi-3.5 style messages & prompt
        messages = build_messages(prompt, num_images=2)

        prompt_text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Trim trailing training artifact token if present
        if prompt_text.endswith("<|endoftext|>"):
            prompt_text = prompt_text[:-len("<|endoftext|>")]

        # Encode with both images in order [A, B]
        inputs = processor(
            prompt_text,
            [imgA, imgB],
            return_tensors="pt"
        )

        model_device = next(model.parameters()).device
        inputs = {
            k: (v.to(model_device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        input_len = int(inputs["input_ids"].shape[1])

        eos_id = processor.tokenizer.eos_token_id
        pad_id = processor.tokenizer.pad_token_id or eos_id

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": (temperature > 0.0),
            "temperature": temperature if temperature > 0.0 else 1.0,
            "repetition_penalty": 1.0,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
            # Important for compatibility with older transformers
            "use_cache": False,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        gen_ids = outputs[0, input_len:]

        generated_text = processor.batch_decode(
            gen_ids.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

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
        description="Evaluate Phi-3.5-vision-instruct on annotated pairs (A,B)."
    )
    ap.add_argument(
        "--dataset-root", default="/nfs_share4/code/om/hypersim/3d_eval_FULL",
        help="Root folder containing pair subfolders + annotations.json"
    )
    ap.add_argument(
        "--model-id", default="microsoft/Phi-3.5-vision-instruct",
        help="HF model id (e.g., microsoft/Phi-3.5-vision-instruct)"
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
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy)"
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root
    items = load_dataset(dataset_root)
    if not items:
        raise SystemExit(f"No pairs found under {dataset_root}")

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
        print("No GPUs found. Running on CPU...")
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
                        shards[i],
                        args.model_id,
                        devices[i],
                        args.prompt,
                        args.max_new_tokens,
                        args.temperature,
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
