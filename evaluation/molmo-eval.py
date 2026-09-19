#!/usr/bin/env python3
"""
Evaluate Molmo-7B-D-0924 on an annotated pairs dataset (A.jpg labeled, B.jpg augmented).

Implements fixes (1–4) and avoids KeyError: 'position_ids' by using Molmo's native
packing + generation path:
  - Use processor.process(images=[...], text=...) to build the batch.
  - Use model.generate_from_batch(..., GenerationConfig, tokenizer=processor.tokenizer).

Requirements:
  pip install -U transformers accelerate torch pillow tqdm einops torchvision
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any, Tuple, Any as AnyType
from dataclasses import dataclass
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
from multiprocessing import get_context

from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

# ---------------------------
# Data loading
# ---------------------------
PROMPT = (
    # (1) Keep the format constraint up front and terse
    "You will answer with ONE capital letter (A–Z) only.\n"
    "Format exactly:\n"
    "Final Answer: <LETTER>\n\n"
    "Task: Here are two photos of a static scene from different views. However, in between taking the photos, "
    "I edited the image such that, for one object, its positioning is inconsistent with the camera motion "
    "between the two frames. Which letter marks the modified object? Use labels from the FIRST image. "
    "In the stitched image I provided, A is on the LEFT and B is on the RIGHT."
)

FINAL_RE = re.compile(r"Final Answer:\s*([A-Z])")

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
# Precision selection
# ---------------------------

def select_cuda_precision() -> Tuple[torch.dtype, torch.dtype, str]:
    """
    Returns (model_dtype, autocast_dtype, label) for CUDA.
    Prefers BF16 when supported; else FP16.
    """
    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if bf16_ok:
        return torch.bfloat16, torch.bfloat16, "bf16"
    else:
        return torch.float16, torch.float16, "fp16"

# ---------------------------
# Image stitching helpers
# ---------------------------

def pad_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Pads img to target_h (no resizing) with black bars to preserve content."""
    if img.height == target_h:
        return img
    top_pad = (target_h - img.height) // 2
    bottom_pad = target_h - img.height - top_pad
    return ImageOps.expand(img, border=(0, top_pad, 0, bottom_pad), fill=(0, 0, 0))

def concat_side_by_side(imgA: Image.Image, imgB: Image.Image) -> Image.Image:
    """
    Return a single RGB image with A on the LEFT and B on the RIGHT.
    If heights differ, pad (do not resize) to the max height to avoid distortion.
    """
    if imgA.mode != "RGB":
        imgA = imgA.convert("RGB")
    if imgB.mode != "RGB":
        imgB = imgB.convert("RGB")

    H = max(imgA.height, imgB.height)
    imgA_p = pad_to_height(imgA, H)
    imgB_p = pad_to_height(imgB, H)

    out = Image.new("RGB", (imgA_p.width + imgB_p.width, H))
    out.paste(imgA_p, (0, 0))
    out.paste(imgB_p, (imgA_p.width, 0))
    return out

# ---------------------------
# Helpers (1–4)
# ---------------------------

def build_prompt(prompt: str) -> str:
    # Keep the final-line format to ease parsing
    return f"{prompt}\nRespond now.\nFinal Answer: <LETTER>"

def parse_letter_from_generated(generated_text: str) -> str:
    if not isinstance(generated_text, str):
        return ""
    m = FINAL_RE.search(generated_text)
    if m:
        return m.group(1)
    # Conservative fallbacks
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
    """
    Load processor and model. Prefer BF16 on CUDA if supported, else FP16; use FP32 on CPU.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    model_dtype = torch.bfloat16 if (use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else (torch.float16 if use_cuda else torch.float32)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        device_map=(device if use_cuda else None),
    )

    if not getattr(model, "hf_device_map", None):
        model.to(device if use_cuda else "cpu")

    model.eval()
    return model, processor

def run_inference(
    items: List[PairItem],
    model_id: str,
    device: str,
    prompt: str,
    max_new_tokens: int = 8,      # (2) short
    temperature: float = 0.0      # (2) deterministic; ignored when do_sample=False
) -> List[Dict[str, Any]]:
    model, processor = load_model_and_processor(model_id, device)
    results: List[Dict[str, Any]] = []

    tokenizer = getattr(processor, "tokenizer", None)
    eos_id = tokenizer.eos_token_id if (tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None) else None

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    autocast_dtype = select_cuda_precision()[1] if use_cuda else None

    # (2)(3) Generation config: no sampling, short, EOS
    gen_cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,             # force greedy
        temperature=None,            # don't re-enable sampling implicitly
        eos_token_id=eos_id
    )

    for it in tqdm(items, desc=f"{device}"):
        try:
            imgA = Image.open(it.A_path)
            if imgA.mode != "RGB":
                imgA = imgA.convert("RGB")
            imgB = Image.open(it.B_path)
            if imgB.mode != "RGB":
                imgB = imgB.convert("RGB")
        except Exception as e:
            results.append({
                "pair_id": it.pair_id,
                "error": f"Image open failed: {repr(e)}",
                "pred_letter": "",
                "gt_letter": it.gt_letter,
                "correct": False
            })
            continue

        # Build stitched image (A on LEFT, B on RIGHT)
        combined = concat_side_by_side(imgA, imgB)
        text = build_prompt(prompt)

        # (1) Use Molmo's native packing. This ensures position_ids etc. are present.
        proc = processor.process(images=[combined], text=text)

        # Batchize and move to device
        inputs = {
            k: (v.to(model.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in proc.items()
        }

        # Record pre-length for (4) span decoding
        input_len = int(inputs["input_ids"].size(1)) if isinstance(inputs.get("input_ids"), torch.Tensor) else None

        # Generate via Molmo's helper (avoids KeyError: 'position_ids')
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                output_ids = model.generate_from_batch(inputs, gen_cfg, tokenizer=processor.tokenizer)
        else:
            output_ids = model.generate_from_batch(inputs, gen_cfg, tokenizer=processor.tokenizer)

        # (4) Decode only continuation beyond input length
        if input_len is not None:
            new_tokens = output_ids[0, input_len:]
        else:
            new_tokens = output_ids[0]

        generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

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
        description="Evaluate Molmo-7B-D-0924 on annotated pairs (A,B) by stitching them into one image."
    )
    ap.add_argument(
        "--dataset-root", default="/nfs_share4/code/om/hypersim/3d_eval_FULL",
        help="Root folder containing pair subfolders + annotations.json"
    )
    ap.add_argument(
        "--model-id", default="allenai/Molmo-7B-D-0924",
        help="HF model id for Molmo-7B-D-0924"
    )
    ap.add_argument(
        "--output-dir", default="/nfs_share4/code/om/hypersim/results_eval_full",
        help="Directory where to save results and summary with model name"
    )
    ap.add_argument(
        "--prompt", default=PROMPT,
        help="Text instruction sent with stitched A|B image"
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max new tokens for generation (short; 8–16 is typical)"
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="Ignored when do_sample=False; kept for CLI parity"
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
        print("No GPUs found. Running on CPU (fp32)…")
        all_results = run_inference(
            items=items,
            model_id=args.model_id,
            device="cpu",
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
    else:
        # Inform about chosen precision for visibility
        dtype_label = "bf16" if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else "fp16"
        print(f"CUDA detected on {num_gpus} device(s). Using {dtype_label} autocast and weights.")
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
