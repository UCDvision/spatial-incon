#!/usr/bin/env python3
"""
Evaluate InternVL3-style chat models on an annotated pairs dataset
(A.jpg labeled, B.jpg augmented).

For each pair:
  - Send BOTH images (A first, B second) with a prompt to an InternVL chat model.
  - Parse the model's final output for a single uppercase letter A–Z.
  - Save a big JSON with raw responses (full generated output) and per-pair correctness.
  - Print and save an accuracy summary.

Parallelization:
  - Shards pairs across all available GPUs (one process per GPU).
  - Each process loads a single model instance on its GPU and iterates its shard.
  - CPU fallback if no GPU is available (single process).

Requirements:
  pip install -U transformers accelerate torch torchvision pillow tqdm
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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer

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
# InternVL inference
# ---------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_question(prompt: str) -> str:
    strict = (
        f"{prompt}\n\n"
        "Format your reply exactly as:\n"
        "Final Answer: <LETTER>\n"
        "where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
    )
    return f"Image-1: <image>\nImage-2: <image>\n{strict}"

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


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_to_pixel_values(
    image: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def load_model_and_tokenizer(model_id: str, device: str):
    # dtype: bf16 on CUDA when available; on CPU use float32
    if device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval()
    if device != "cpu":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer, dtype


def run_inference(
    items: List[PairItem],
    model_id: str,
    device: str,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    max_tiles_per_image: int = 12,
) -> List[Dict[str, Any]]:
    model, tokenizer, dtype = load_model_and_tokenizer(model_id, device)
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

        pixel_values_a = image_to_pixel_values(
            imgA, max_num=max_tiles_per_image
        )
        pixel_values_b = image_to_pixel_values(
            imgB, max_num=max_tiles_per_image
        )
        pixel_values = torch.cat((pixel_values_a, pixel_values_b), dim=0)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        num_patches_list = [pixel_values_a.size(0), pixel_values_b.size(0)]
        question = build_question(prompt)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_config["temperature"] = temperature

        with torch.no_grad():
            generated_text = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
            )

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
    (
        shard_items, model_id, device, prompt, max_new_tokens, temperature,
        max_tiles_per_image,
    ) = args
    return run_inference(
        items=shard_items,
        model_id=model_id,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_tiles_per_image=max_tiles_per_image,
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
        description="Evaluate InternVL3-style chat models on annotated pairs (A,B)."
    )
    ap.add_argument(
        "--dataset-root", default="/nfs_share4/code/om/hypersim/3d_eval_FULL",
        help="Root folder containing pair subfolders + annotations.json"
    )
    ap.add_argument(
        "--model-id", default="sensenova/SenseNova-SI-1.4-InternVL3-8B",
        help="HF model id for an InternVL chat model"
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
    ap.add_argument(
        "--max-tiles-per-image", type=int, default=12,
        help="Maximum dynamic-resolution tiles per image before thumbnail"
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
            temperature=args.temperature,
            max_tiles_per_image=args.max_tiles_per_image,
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
                        args.prompt, args.max_new_tokens, args.temperature,
                        args.max_tiles_per_image,
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
