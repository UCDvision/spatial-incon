#!/usr/bin/env python3
"""
Evaluate OpenFlamingo on an annotated pairs dataset (A.jpg labeled, B.jpg augmented).

For each pair:
  - Send BOTH images (A first, B second) with a strict prompt to an OpenFlamingo checkpoint.
  - Parse the model's final output for a single uppercase letter A–Z.
  - Save a big JSON with raw responses and per-pair correctness.
  - Print and save an accuracy summary.

Parallelization:
  - Shards pairs across all available GPUs (one process per GPU).
  - Each process loads a single model instance on its GPU and iterates its shard.
  - CPU fallback if no GPU is available (single process).

Tested with OpenFlamingo 4B/9B checkpoints on HF Hub.

Requirements:
  pip install -U torch torchvision pillow tqdm transformers accelerate huggingface_hub sentencepiece einops open-clip-torch
  pip install -U open_flamingo
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any, Any as AnyType
from dataclasses import dataclass
from tqdm import tqdm
import torch
from PIL import Image
from multiprocessing import get_context

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from open_flamingo import create_model_and_transforms

# ---------------------------
# Data loading
# ---------------------------
PROMPT = (
    "Here are two photos of a static scene from different views. However, in between taking the photos, "
    "I edited the image such that, for one object, its positioning is inconsistent with the camera motion "
    "between the two frames. Which letter marks the modified object? Please use the labels from the first image."
)

STRICT_SUFFIX = (
    "\n\nFormat your reply exactly as:\n"
    "Final Answer: <LETTER>\n"
    "…where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
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
# OpenFlamingo inference helpers
# ---------------------------

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


def load_openflamingo(model_id: str, device: str):
    """
    Builds an OpenFlamingo model & transforms, then loads weights from HF Hub.
    Default configs target the common 4B/9B variants (CLIP ViT-L/14 + RPJ/MPT).
    """
    # Choose language model & tokenizer path based on checkpoint family
    if "mpt7b" in model_id.lower():
        lang_path = "mosaicml/mpt-7b"
    elif "rpj" in model_id.lower() or "redpajama" in model_id.lower():
        lang_path = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
    else:
        # Fallback; user can override if needed
        lang_path = "togethercomputer/RedPajama-INCITE-Base-3B-v1"

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=lang_path,
        tokenizer_path=lang_path,
        cross_attn_every_n_layers=2,  # per OpenFlamingo defaults
    )

    # Load checkpoint weights from the model repo (file: checkpoint.pt)
    ckpt_path = hf_hub_download(repo_id=model_id, filename="checkpoint.pt")
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)

    # Device/dtype
    if device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model.to(device=device, dtype=dtype)
    model.eval()

    # For generation, padding should be on the left (per model cards)
    tokenizer.padding_side = "left"

    return model, image_processor, tokenizer, dtype


def build_inputs_openflamingo(
    imgA: Image.Image, imgB: Image.Image, tokenizer: AutoTokenizer, image_processor, prompt: str
):
    """
    OpenFlamingo expects:
      - vision_x: tensor [B, num_media, num_frames, 3, H, W]
      - lang_x: tokenized text containing <image> markers and <|endofchunk|> separators
    We keep A->B order. We place the instruction after the two images.
    """
    with torch.no_grad():
        xa = image_processor(imgA).unsqueeze(0)  # [1,3,H,W]
        xb = image_processor(imgB).unsqueeze(0)
        vision_x = torch.cat([xa, xb], dim=0)    # [2,3,H,W]
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)  # [1,2,1,3,H,W]

    # Two images, then the instruction
    # We use <image> tokens twice, then the strict instruction. End the
    # image-associated empty chunks with <|endofchunk|> to match examples. :contentReference[oaicite:1]{index=1}
    text = "<image><|endofchunk|><image>" + "\n" + prompt + STRICT_SUFFIX

    lang = tokenizer([text], return_tensors="pt")
    return vision_x, lang


def run_inference(
    items: List[PairItem],
    model_id: str,
    device: str,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    model, image_processor, tokenizer, dtype = load_openflamingo(model_id, device)
    results: List[Dict[str, Any]] = []

    do_sample = temperature > 0.0

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

        vision_x, lang = build_inputs_openflamingo(imgA, imgB, tokenizer, image_processor, prompt)

        # Move to model device/dtype
        vision_x = vision_x.to(next(model.parameters()).device, dtype=dtype)
        input_ids = lang["input_ids"].to(next(model.parameters()).device)
        attn = lang["attention_mask"].to(next(model.parameters()).device)

        with torch.no_grad():
            generated_ids = model.generate(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                repetition_penalty=1.0,
                num_beams=None if do_sample else 1,
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

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
        description="Evaluate OpenFlamingo on annotated pairs (A,B)."
    )
    ap.add_argument(
        "--dataset-root", default="/nfs_share4/code/om/hypersim/3d_eval_FULL",
        help="Root folder containing pair subfolders + annotations.json"
    )
    ap.add_argument(
        "--model-id", default="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
        help="HF repo id for an OpenFlamingo checkpoint (e.g., OpenFlamingo-4B-vitl-rpj3b)"
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

    items = load_dataset(args.dataset_root)
    if not items:
        raise SystemExit(f"No pairs found under {args.dataset_root}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_stamp = args.model_id.replace("/", "_")
    out_results = os.path.join(args.output_dir, f"{model_stamp}_results.json")
    out_summary = os.path.join(args.output_dir, f"{model_stamp}_summary.json")

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
                    (shards[i], args.model_id, devices[i],
                     args.prompt, args.max_new_tokens, args.temperature)
                    for i in range(num_gpus)
                ],
                chunksize=1
            )
        all_results = [r for sub in mapped for r in sub]

    with open(out_results, "w") as f:
        json.dump(all_results, f, indent=2)

    summary = summarize(all_results)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved per-pair results to: {out_results}")
    print(f"Saved summary to: {out_summary}")

if __name__ == "__main__":
    main()
