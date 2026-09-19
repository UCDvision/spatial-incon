#!/usr/bin/env python3
"""
Evaluate Qwen2.5/3-VL on an annotated pairs dataset (A.jpg labeled, B.jpg augmented).

Minimal GPU-parallel fixes in this version:
  - Uses ALL available GPUs via CUDA_VISIBLE_DEVICES / torch.cuda.device_count().
  - Replaces Manager().Queue() with a fast ctx.Queue() (avoids Manager bottleneck).
  - Uses non-daemon worker processes (daemon can cause premature termination / join issues).
  - Keeps the rest of your logic (streaming results, periodic saves, checkpoint sweep) the same.
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any, Tuple, Callable, Optional, Any as AnyType
from dataclasses import dataclass
from tqdm import tqdm
import torch
from PIL import Image
from multiprocessing import get_context
from pathlib import Path
from collections import Counter

# Compatibility shim for older Torch builds used with newer Transformers/Qwen codepaths.
if not hasattr(torch, "compiler"):
    class _TorchCompilerShim:
        @staticmethod
        def is_compiling():
            return False
    torch.compiler = _TorchCompilerShim()
elif not hasattr(torch.compiler, "is_compiling"):
    def _torch_is_compiling_fallback():
        return False
    torch.compiler.is_compiling = _torch_is_compiling_fallback

from transformers import AutoProcessor, AutoModelForVision2Seq, LogitsProcessor

# ---------------------------
# Data loading
# ---------------------------
PROMPT = (
    "Here are two photos of a static scene from different views. However, in between taking the photos, "
    "I edited the image such that, for one object, its positioning is inconsistent with the camera motion "
    "between the two frames. Which letter marks the modified object? Please use the labels from the first image. If an object is present in one frame but not the other, please ignore it."
)

PROMPT = (
    "Here are two photos of a static scene from different views. However, in between taking the photos, "
    "for one object, it was moved so its positioning is inconsistent with the camera motion "
    "between the two frames. Which letter marks the modified object? Please use the labels from the first image. If an object is present in one frame but not the other, please ignore it."
)


STRUCTURED_PROMPT_STEPS = [
    (
        "describe_image_a",
        "Describe Image A: briefly describe the scene, camera viewpoint, visible labeled objects, and relevant occlusions.",
    ),
    (
        "describe_image_b",
        "Describe Image B: briefly describe the same scene from the second viewpoint, including visible corresponding objects and occlusions.",
    ),
    (
        "list_correspondences",
        "List correspondences: list each labeled object from Image A that has a clear corresponding object in Image B. Ignore objects that are present in only one frame.",
    ),
    (
        "compare_correspondences",
        "Compare each corresponding object: for every correspondence, compare its pose, scale, orientation, lighting, and occlusion. Distinguish changes expected from camera motion from changes that look physically inconsistent.",
    ),
    (
        "decide",
        "Decide: choose the single Image A label whose object is most inconsistent with the camera motion.",
    ),
]

STRUCTURED_PROMPT_SUFFIX = "Use this structured visual reasoning procedure before choosing the label:\n" + "\n".join(
    f"{i}. {instruction}" for i, (_, instruction) in enumerate(STRUCTURED_PROMPT_STEPS, start=1)
)

STRUCTURED_PROMPT = f"{PROMPT}\n\n{STRUCTURED_PROMPT_SUFFIX}"

NONE_OF_THE_ABOVE_TOKEN = "NONE_OF_THE_ABOVE"

@dataclass
class PairItem:
    pair_id: str
    pair_dir: str
    A_path: str
    B_path: str
    gt_letter: str  # may be None if missing
    meta: Dict[str, Any]


@dataclass(frozen=True)
class GenerationSettings:
    do_sample: bool = False
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0


RANDOM_GENERATION_SETTINGS = GenerationSettings(
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.0,
    presence_penalty=1.5,
)


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        for batch_idx in range(input_ids.shape[0]):
            seen_token_ids = torch.unique(input_ids[batch_idx])
            scores[batch_idx, seen_token_ids] -= self.penalty
        return scores


def load_dataset(dataset_root: str) -> List[PairItem]:
    ann_path = os.path.join(dataset_root, "annotations.json")
    with open(ann_path, "r") as f:
        anns = json.load(f)

    items: List[PairItem] = []
    for rec in tqdm(anns):
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
# Small utils (layout / plots / discovery)
# ---------------------------

def sanitize_tag(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def discover_dataset_roots(dataset_root: str) -> List[Tuple[str, str]]:
    root = Path(dataset_root)
    if (root / "annotations.json").is_file():
        return [(root.name, str(root))]
    children = sorted(p for p in root.iterdir() if p.is_dir() and (p / "annotations.json").is_file())
    return [(p.name, str(p)) for p in children]

def discover_checkpoints(variant_dir: str, ckpt_regex: re.Pattern) -> List[Tuple[int, str, str]]:
    if not os.path.isdir(variant_dir):
        return []
    found = []
    for name in os.listdir(variant_dir):
        path = os.path.join(variant_dir, name)
        if not os.path.isdir(path):
            continue
        m = ckpt_regex.match(name)
        if m:
            step = None
            for g in m.groups():
                if g is not None:
                    try:
                        step = int(g)
                        break
                    except ValueError:
                        pass
            if step is None:
                digs = re.findall(r'\d+', name)
                step = int(digs[0]) if digs else -1
            found.append((step, name, path))
    found.sort(key=lambda x: x[0])
    return found

def try_plot_xy(xs: List[int], ys: List[float], title: str, out_path: Path, fmt: str = "png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Checkpoint (iteration/step)")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        out_path = out_path.with_suffix(f".{fmt}")
        plt.savefig(out_path)
        plt.close()
        return str(out_path)
    except Exception:
        return None

# ---------------------------
# Qwen VL inference helpers
# ---------------------------

def build_messages(
    prompt: str,
    imgA: Image.Image,
    imgB: Image.Image,
    include_none_of_the_above: bool = False,
    structured_prompting: bool = False,
) -> List[Dict[str, Any]]:
    if include_none_of_the_above:
        if structured_prompting:
            strict = (
                f"{prompt}\n\n"
                "If none of the labeled objects is the modified object, choose None of the above.\n\n"
                "After the structured reasoning, end your reply with exactly one of these final lines:\n"
                "Final Answer: <LETTER>\n"
                f"Final Answer: {NONE_OF_THE_ABOVE_TOKEN}\n"
                "where <LETTER> is ONE capital letter A to Z only, and "
                f"{NONE_OF_THE_ABOVE_TOKEN} means None of the above."
            )
        else:
            strict = (
                f"{prompt}\n\n"
                "If none of the labeled objects is the modified object, choose None of the above.\n\n"
                "Format your reply exactly as one of:\n"
                "Final Answer: <LETTER>\n"
                f"Final Answer: {NONE_OF_THE_ABOVE_TOKEN}\n"
                "…where <LETTER> is ONE capital letter A to Z only, and "
                f"{NONE_OF_THE_ABOVE_TOKEN} means None of the above. Do not write anything else."
            )
    else:
        if structured_prompting:
            strict = (
                f"{prompt}\n\n"
                "After the structured reasoning, end your reply with exactly this final line:\n"
                "Final Answer: <LETTER>\n"
                "where <LETTER> is ONE capital letter A to Z only. DO NOT put None as your final answer. "
            )
        else:
            strict = (
                f"{prompt}\n\n"
                "Format your reply exactly as:\n"
                "Final Answer: <LETTER>\n"
                "…where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
            )
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
ANSWER_HINT_RE = re.compile(
    r"(?:modified|inconsistent|answer|label|object)\b\D{0,40}\b([A-Z])\b",
    re.IGNORECASE,
)
FINAL_NONE_RE = re.compile(
    r"Final Answer:\s*(NONE_OF_THE_ABOVE|NONE(?:\s+OF\s+THE\s+ABOVE)?)\b",
    re.IGNORECASE,
)
NONE_TEXT_RE = re.compile(r"\bNONE(?:\s+OF\s+THE\s+ABOVE)?\b", re.IGNORECASE)

def parse_letter_from_generated(
    generated_text: str,
    allow_none_of_the_above: bool = False,
    structured_prompting: bool = False,
) -> str:
    if not isinstance(generated_text, str):
        return ""
    if allow_none_of_the_above:
        m_none = FINAL_NONE_RE.search(generated_text)
        if m_none:
            return NONE_OF_THE_ABOVE_TOKEN
    m = FINAL_RE.search(generated_text)
    if m:
        return m.group(1)
    lines = [ln.strip() for ln in generated_text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if allow_none_of_the_above and NONE_TEXT_RE.fullmatch(last):
            return NONE_OF_THE_ABOVE_TOKEN
        if len(last) == 1 and "A" <= last <= "Z":
            return last
        if allow_none_of_the_above and NONE_TEXT_RE.search(last):
            return NONE_OF_THE_ABOVE_TOKEN
        if structured_prompting:
            hint_matches = ANSWER_HINT_RE.findall(last)
            for candidate in reversed(hint_matches):
                candidate = candidate.upper()
                if "A" <= candidate <= "Z":
                    return candidate
        else:
            for ch in last:
                if "A" <= ch <= "Z":
                    return ch
    if allow_none_of_the_above and NONE_TEXT_RE.search(generated_text):
        return NONE_OF_THE_ABOVE_TOKEN
    if structured_prompting:
        hint_matches = ANSWER_HINT_RE.findall(generated_text)
        for candidate in reversed(hint_matches):
            candidate = candidate.upper()
            if "A" <= candidate <= "Z":
                return candidate
        return ""
    for ch in generated_text:
        if "A" <= ch <= "Z":
            return ch
    return ""


def load_model_and_processor(model_id: str, device: str):
    dtype = (
        torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    return model, processor


def _special_token_ids(tokenizer):
    """
    Obtain token ids for markers used in the Qwen quickstart strategy.
    """
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    THINK_CLOSE_ID = think_close_ids[-1] if think_close_ids else None
    IM_END_ID = im_end_ids[-1] if im_end_ids else None
    return THINK_CLOSE_ID, IM_END_ID


def _generate_kwargs(settings: GenerationSettings) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "do_sample": bool(settings.do_sample),
        "repetition_penalty": float(settings.repetition_penalty),
    }
    if settings.do_sample:
        kwargs["temperature"] = float(settings.temperature)
        if settings.top_p is not None:
            kwargs["top_p"] = float(settings.top_p)
        if settings.top_k is not None:
            kwargs["top_k"] = int(settings.top_k)
    return kwargs


def _presence_logits_processor(settings: GenerationSettings):
    if settings.presence_penalty == 0.0:
        return None
    return [PresencePenaltyLogitsProcessor(settings.presence_penalty)]


def _majority_vote(preds: List[str]) -> str:
    non_empty = [p for p in preds if p]
    if not non_empty:
        return ""
    counts = Counter(non_empty)
    max_count = max(counts.values())
    tied = {pred for pred, count in counts.items() if count == max_count}
    for pred in non_empty:
        if pred in tied:
            return pred
    return non_empty[0]


def _two_pass_budgeted_generation(
    model,
    processor,
    messages,
    images,
    *,
    thinking_budget: int,
    max_new_tokens: int,
    generation_settings: GenerationSettings,
) -> str:
    """
    Qwen quickstart strategy:
      - Pass 1: generate up to `thinking_budget`.
      - If `</think>` not reached and not finished, append the early-stopping text and resume.
    """
    base_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs0 = processor(text=base_text, images=images, return_tensors="pt").to(model.device)
    input_len0 = int(inputs0["input_ids"].shape[1])

    with torch.no_grad():
        generate_kwargs = _generate_kwargs(generation_settings)
        presence_logits_processor = _presence_logits_processor(generation_settings)
        if presence_logits_processor is not None:
            generate_kwargs["logits_processor"] = presence_logits_processor
        out1 = model.generate(
            **inputs0,
            max_new_tokens=max(1, int(thinking_budget)),
            **generate_kwargs,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    gen_ids1 = out1[0, input_len0:]
    out_ids1 = gen_ids1.tolist()
    partial_text = processor.tokenizer.decode(gen_ids1, skip_special_tokens=True)

    THINK_CLOSE_ID, IM_END_ID = _special_token_ids(processor.tokenizer)

    def _has_tok(seq, tok_id):
        return (tok_id is not None) and (tok_id in seq)

    if _has_tok(out_ids1, IM_END_ID) or _has_tok(out_ids1, THINK_CLOSE_ID) or ("</think>" in partial_text):
        return partial_text

    early_stopping_text = (
        "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
    )
    continue_text = base_text + partial_text + early_stopping_text
    inputs2 = processor(text=continue_text, images=images, return_tensors="pt").to(model.device)
    input_len2 = int(inputs2["input_ids"].shape[1])

    residual = max(1, int(max_new_tokens) - len(out_ids1))

    with torch.no_grad():
        generate_kwargs = _generate_kwargs(generation_settings)
        presence_logits_processor = _presence_logits_processor(generation_settings)
        if presence_logits_processor is not None:
            generate_kwargs["logits_processor"] = presence_logits_processor
        out2 = model.generate(
            **inputs2,
            max_new_tokens=residual,
            **generate_kwargs,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    gen_ids2 = out2[0, input_len2:]
    tail_text = processor.tokenizer.decode(gen_ids2, skip_special_tokens=True)
    return partial_text + tail_text


def _generate_text(
    model,
    processor,
    messages,
    images,
    *,
    max_new_tokens: int,
    thinking_budget: Optional[int],
    use_thinking_budget: bool,
    generation_settings: GenerationSettings,
) -> str:
    if use_thinking_budget:
        return _two_pass_budgeted_generation(
            model=model,
            processor=processor,
            messages=messages,
            images=images,
            thinking_budget=int(thinking_budget),
            max_new_tokens=int(max_new_tokens),
            generation_settings=generation_settings,
        )

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt_text,
        images=images,
        return_tensors="pt"
    ).to(model.device)
    input_len = int(inputs["input_ids"].shape[1])
    generate_kwargs = _generate_kwargs(generation_settings)
    presence_logits_processor = _presence_logits_processor(generation_settings)
    if presence_logits_processor is not None:
        generate_kwargs["logits_processor"] = presence_logits_processor
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    gen_ids = outputs[0, input_len:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True)


def build_structured_step_messages(
    base_prompt: str,
    imgA: Image.Image,
    imgB: Image.Image,
    step_index: int,
    prior_steps: List[Dict[str, str]],
    include_none_of_the_above: bool = False,
) -> List[Dict[str, Any]]:
    step_name, instruction = STRUCTURED_PROMPT_STEPS[step_index]
    prior_text = ""
    if prior_steps:
        prior_text = "\n\nPrior step outputs:\n" + "\n\n".join(
            f"{i}. {step['step_name']}:\n{step['text']}"
            for i, step in enumerate(prior_steps, start=1)
        )

    if step_name == "decide":
        if include_none_of_the_above:
            step_prompt = (
                f"{base_prompt}\n\n"
                "If none of the labeled objects is the modified object, choose None of the above.\n\n"
                f"{prior_text}\n\n"
                f"Now run only this final structured step:\n{instruction}\n\n"
                "Format your reply exactly as one of:\n"
                "Final Answer: <LETTER>\n"
                f"Final Answer: {NONE_OF_THE_ABOVE_TOKEN}\n"
                "where <LETTER> is ONE capital letter A to Z only, and "
                f"{NONE_OF_THE_ABOVE_TOKEN} means None of the above. Do not write anything else."
            )
        else:
            step_prompt = (
                f"{base_prompt}"
                f"{prior_text}\n\n"
                f"Now run only this final structured step:\n{instruction}\n\n"
                "Format your reply exactly as:\n"
                "Final Answer: <LETTER>\n"
                "where <LETTER> is ONE capital letter A to Z only. "
                "You must choose the best Image A label even if uncertain. "
                "Do not answer None, no object, unknown, or N/A. Do not write anything else."
            )
    else:
        step_prompt = (
            f"{base_prompt}"
            f"{prior_text}\n\n"
            f"Run only structured step {step_index + 1} of {len(STRUCTURED_PROMPT_STEPS)}:\n"
            f"{instruction}\n\n"
            "Do not decide the final answer yet, and do not state that the final answer is None."
        )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": imgA},
                {"type": "image", "image": imgB},
                {"type": "text", "text": step_prompt},
            ],
        }
    ]


def run_structured_multi_query(
    model,
    processor,
    *,
    base_prompt: str,
    imgA: Image.Image,
    imgB: Image.Image,
    max_new_tokens: int,
    thinking_budget: Optional[int],
    use_thinking_budget: bool,
    generation_settings: GenerationSettings,
    include_none_of_the_above: bool = False,
) -> Tuple[str, List[Dict[str, str]]]:
    trace: List[Dict[str, str]] = []
    for step_index, (step_name, _) in enumerate(STRUCTURED_PROMPT_STEPS):
        step_messages = build_structured_step_messages(
            base_prompt=base_prompt,
            imgA=imgA,
            imgB=imgB,
            step_index=step_index,
            prior_steps=trace,
            include_none_of_the_above=include_none_of_the_above,
        )
        step_text = _generate_text(
            model,
            processor,
            step_messages,
            [imgA, imgB],
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
            use_thinking_budget=use_thinking_budget,
            generation_settings=generation_settings,
        )
        trace.append({
            "step": str(step_index + 1),
            "step_name": step_name,
            "text": step_text,
        })
    return trace[-1]["text"] if trace else "", trace


def run_inference(
    items: List[PairItem],
    model_id: str,
    device: str,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    thinking_budget: Optional[int] = None,
    include_none_of_the_above: bool = False,
    structured_prompting: bool = False,
    enable_randomness: bool = False,
    k_times: int = 1,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    disable_tqdm: bool = False,
) -> List[Dict[str, Any]]:
    model, processor = load_model_and_processor(model_id, device)
    results: List[Dict[str, Any]] = []
    k_times = max(1, int(k_times))
    generation_settings = (
        RANDOM_GENERATION_SETTINGS
        if enable_randomness or k_times > 1
        else GenerationSettings(do_sample=(float(temperature) > 0.0), temperature=float(temperature))
    )

    use_thinking_budget = (
        thinking_budget is not None and thinking_budget > 0 and ("thinking" in model_id.lower())
    )

    for it in tqdm(items, desc=f"{device}", disable=disable_tqdm):
        try:
            imgA = Image.open(it.A_path).convert("RGB")
            imgB = Image.open(it.B_path).convert("RGB")
        except Exception as e:
            res = {
                "pair_id": it.pair_id,
                "error": f"Image open failed: {repr(e)}",
                "pred_letter": "",
                "gt_letter": it.gt_letter,
                "correct": False,
                "a_path": it.A_path,
                "b_path": it.B_path,
            }
            results.append(res)
            if on_result:
                on_result(res)
            continue

        generated_texts: List[str] = []
        structured_traces: List[List[Dict[str, str]]] = []
        preds: List[str] = []
        try:
            for _ in range(k_times):
                if structured_prompting:
                    generated_text, structured_trace = run_structured_multi_query(
                        model,
                        processor,
                        base_prompt=prompt,
                        imgA=imgA,
                        imgB=imgB,
                        max_new_tokens=max_new_tokens,
                        thinking_budget=thinking_budget,
                        use_thinking_budget=use_thinking_budget,
                        generation_settings=generation_settings,
                        include_none_of_the_above=include_none_of_the_above,
                    )
                    structured_traces.append(structured_trace)
                else:
                    messages = build_messages(
                        prompt,
                        imgA,
                        imgB,
                        include_none_of_the_above,
                        structured_prompting,
                    )
                    generated_text = _generate_text(
                        model,
                        processor,
                        messages,
                        [imgA, imgB],
                        max_new_tokens=max_new_tokens,
                        thinking_budget=thinking_budget,
                        use_thinking_budget=use_thinking_budget,
                        generation_settings=generation_settings,
                    )

                generated_texts.append(generated_text)
                preds.append(parse_letter_from_generated(
                    generated_text,
                    include_none_of_the_above,
                    structured_prompting,
                ))
        except Exception as e:
            res = {
                "pair_id": it.pair_id,
                "error": f"Generation failed: {repr(e)}",
                "pred_letter": "",
                "gt_letter": it.gt_letter,
                "correct": False,
                "a_path": it.A_path,
                "b_path": it.B_path,
            }
            results.append(res)
            if on_result:
                on_result(res)
            continue

        pred = _majority_vote(preds)
        correct = (pred == it.gt_letter) if it.gt_letter else False
        oracle_correct = any(p == it.gt_letter for p in preds) if it.gt_letter else False
        res = {
            "pair_id": it.pair_id,
            "pred_letter": pred,
            "gt_letter": it.gt_letter,
            "correct": bool(correct),
            "raw_text": generated_texts[0] if len(generated_texts) == 1 else generated_texts,
            "a_path": it.A_path,
            "b_path": it.B_path,
        }
        if structured_prompting:
            res["structured_queries"] = (
                structured_traces[0] if len(structured_traces) == 1 else structured_traces
            )
        if k_times > 1:
            res.update({
                "k_times": k_times,
                "sample_pred_letters": preds,
                "majority_vote_pred_letter": pred,
                "majority_vote_correct": bool(correct),
                "oracle_correct": bool(oracle_correct),
            })
        results.append(res)
        if on_result:
            on_result(res)

    return results

# ---------------------------
# Sharding & Multiprocessing
# ---------------------------

def shard(lst: List[AnyType], n: int) -> List[List[AnyType]]:
    if n <= 1:
        return [lst]
    return [lst[i::n] for i in range(n)]


def _worker_stream(
    shard_items,
    model_id,
    device,
    prompt,
    max_new_tokens,
    temperature,
    thinking_budget,
    include_none_of_the_above,
    structured_prompting,
    enable_randomness,
    k_times,
    q,
):
    try:
        def push(res: Dict[str, Any]):
            q.put(("result", res), block=True)

        run_inference(
            items=shard_items,
            model_id=model_id,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
            include_none_of_the_above=include_none_of_the_above,
            structured_prompting=structured_prompting,
            enable_randomness=enable_randomness,
            k_times=k_times,
            on_result=push,
            disable_tqdm=True,
        )
    except Exception as e:
        q.put(("error", {"device": device, "error": repr(e)}), block=True)
    finally:
        q.put(("done", {"device": device}), block=True)

# ---------------------------
# Accuracy & I/O
# ---------------------------

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in results if r.get("gt_letter")]
    if not evaluated:
        return {"num_pairs": len(results), "num_evaluated": 0, "accuracy": None}
    correct = sum(1 for r in evaluated if r.get("correct"))
    acc = correct / max(1, len(evaluated))
    summary = {
        "num_pairs": len(results),
        "num_evaluated": len(evaluated),
        "num_correct": correct,
        "accuracy": acc
    }
    if any("oracle_correct" in r for r in evaluated):
        oracle_correct = sum(1 for r in evaluated if r.get("oracle_correct"))
        summary.update({
            "majority_vote_num_correct": correct,
            "majority_vote_accuracy": acc,
            "oracle_num_correct": oracle_correct,
            "oracle_accuracy": oracle_correct / max(1, len(evaluated)),
        })
    return summary


def _save_now(all_results: List[Dict[str, Any]], out_results_path: str, out_summary_path: str):
    def atomic_json_dump(obj: Any, path: str):
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)

    atomic_json_dump(all_results, out_results_path)
    summary = summarize(all_results)
    atomic_json_dump(summary, out_summary_path)
    return summary

# ---------------------------
# Single evaluation runner (used by both normal and checkpoint modes)
# ---------------------------

def evaluate_one_model(
    *,
    items: List[PairItem],
    model_id: str,
    output_dir: str,
    out_results_path: str,
    out_summary_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    thinking_budget: int,
    include_none_of_the_above: bool,
    structured_prompting: bool,
    enable_randomness: bool,
    k_times: int,
    save_every: int,
) -> Dict[str, Any]:
    """
    Runs the streaming multi-process evaluation for `model_id` and writes results/summary
    to the provided absolute file paths. Returns the final summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    _save_now([], out_results_path, out_summary_path)

    # GPU detection (uses ALL visible GPUs)
    try:
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0

    if num_gpus <= 0:
        print("No GPUs found. Running on CPU…")
        devices = ["cpu"]
        shards_list = [items]
    else:
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        shards_list = shard(items, num_gpus)

    # Multiprocessing with a fast Queue (NO Manager)
    ctx = get_context("spawn")
    q = ctx.Queue()

    procs = []
    for i, device in enumerate(devices):
        shard_items = shards_list[i] if i < len(shards_list) else []
        p = ctx.Process(
            target=_worker_stream,
            args=(shard_items, model_id, device, prompt,
                  max_new_tokens, temperature, thinking_budget, include_none_of_the_above,
                  structured_prompting, enable_randomness, k_times, q),
        )
        # NOTE: do NOT daemonize; daemon processes can be killed abruptly and not flush queue.
        p.start()
        procs.append(p)

    all_results: List[Dict[str, Any]] = []
    total = 0
    done_workers = 0
    num_workers = len(procs)

    pbar = tqdm(total=len(items), desc="Total progress", dynamic_ncols=True)

    try:
        while done_workers < num_workers:
            msg_type, payload = q.get()
            if msg_type == "result":
                all_results.append(payload)
                total += 1
                pbar.update(1)

                if save_every and (total % save_every == 0):
                    summary = _save_now(all_results, out_results_path, out_summary_path)
                    print(
                        f"[checkpoint] saved {total}/{len(items)} "
                        f"({summary.get('num_correct', 0)}/{summary.get('num_evaluated', 0)} correct, "
                        f"acc={summary.get('accuracy')})",
                        flush=True,
                    )

            elif msg_type == "error":
                dev = payload.get("device")
                err = payload.get("error")
                tb  = payload.get("traceback", "")
                print(f"\n[worker-error] {dev}: {err}\n", flush=True)
                if tb:
                    print("[traceback]\n" + tb, flush=True)
                for p in procs:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                raise SystemExit(1)

            elif msg_type == "done":
                done_workers += 1
                dev = payload.get("device")
                print(f"[worker-done] {dev}", flush=True)

    finally:
        pbar.close()
        for p in procs:
            try:
                p.join(timeout=5)
            except Exception:
                pass

    final_summary = _save_now(all_results, out_results_path, out_summary_path)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(final_summary, indent=2))
    print(f"\nSaved per-pair results to: {out_results_path}")
    print(f"Saved summary to: {out_summary_path}")

    return final_summary


def run_for_dataset(args, dataset_root: str, dataset_tag: str, multi_dataset: bool) -> Dict[str, Any]:
    items = load_dataset(dataset_root)
    if not items:
        raise SystemExit(f"No pairs found under {dataset_root}")

    dataset_out = Path(args.output_dir) / sanitize_tag(dataset_tag) if multi_dataset else Path(args.output_dir)
    ensure_dir(dataset_out)

    is_dir_model = os.path.isdir(args.model_id)

    if not args.scan_checkpoints:
        if args.checkpoint and is_dir_model:
            base_dir = args.model_id

            cand = os.path.join(base_dir, args.checkpoint)
            if os.path.isdir(cand):
                ckpt_name = args.checkpoint
                ckpt_path = cand
            else:
                ck_re = re.compile(args.checkpoint)
                matches = [d for d in os.listdir(base_dir)
                           if os.path.isdir(os.path.join(base_dir, d)) and ck_re.search(d)]
                if not matches:
                    raise SystemExit(f"No checkpoint under {base_dir} matching: {args.checkpoint}")
                ckpt_name = matches[0]
                ckpt_path = os.path.join(base_dir, ckpt_name)

            base_name = sanitize_tag(Path(base_dir).name)
            root_out = dataset_out / base_name
            ensure_dir(root_out)

            out_results_path = str(root_out / f"{base_name}_{sanitize_tag(ckpt_name)}_results.json")
            out_summary_path = str(root_out / f"{base_name}_{sanitize_tag(ckpt_name)}_summary.json")

            summary = evaluate_one_model(
                items=items,
                model_id=ckpt_path,
                output_dir=str(root_out),
                out_results_path=out_results_path,
                out_summary_path=out_summary_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                thinking_budget=args.thinking_budget,
                include_none_of_the_above=args.include_none_of_the_above,
                structured_prompting=args.structured_prompting,
                enable_randomness=args.enable_randomness,
                k_times=args.k_times,
                save_every=args.save_every,
            )
            return {
                "dataset_tag": dataset_tag,
                "dataset_root": dataset_root,
                "mode": "single_checkpoint",
                "summary_path": out_summary_path,
                "accuracy": summary.get("accuracy"),
                "majority_vote_accuracy": summary.get("majority_vote_accuracy"),
                "oracle_accuracy": summary.get("oracle_accuracy"),
            }

        model_stamp = sanitize_tag(args.model_id.replace("/", "_"))
        out_results_path = str(dataset_out / f"{model_stamp}_results.json")
        out_summary_path = str(dataset_out / f"{model_stamp}_summary.json")

        summary = evaluate_one_model(
            items=items,
            model_id=args.model_id,
            output_dir=str(dataset_out),
            out_results_path=out_results_path,
            out_summary_path=out_summary_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            thinking_budget=args.thinking_budget,
            include_none_of_the_above=args.include_none_of_the_above,
            structured_prompting=args.structured_prompting,
            enable_randomness=args.enable_randomness,
            k_times=args.k_times,
            save_every=args.save_every,
        )
        return {
            "dataset_tag": dataset_tag,
            "dataset_root": dataset_root,
            "mode": "single_model",
            "summary_path": out_summary_path,
            "accuracy": summary.get("accuracy"),
            "majority_vote_accuracy": summary.get("majority_vote_accuracy"),
            "oracle_accuracy": summary.get("oracle_accuracy"),
        }

    base_dir = args.model_id
    if not os.path.isdir(base_dir):
        raise SystemExit(f"--scan-checkpoints set but --model-id is not a directory: {base_dir}")

    ckpt_regex = re.compile(args.ckpt_name_regex)
    ckpts = discover_checkpoints(base_dir, ckpt_regex)
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {base_dir} with regex: {args.ckpt_name_regex}")

    if args.checkpoint:
        pat = re.compile(args.checkpoint)
        ckpts = [c for c in ckpts if (c[1] == args.checkpoint) or pat.search(c[1])]
        if not ckpts:
            raise SystemExit(f"No checkpoints in {base_dir} match --checkpoint={args.checkpoint}")

    print(f"[dataset={dataset_tag}] [ckpt] Found {len(ckpts)} checkpoints: {[name for _, name, _ in ckpts]}")

    base_name = sanitize_tag(Path(base_dir).name)
    root_out = dataset_out / base_name
    ensure_dir(root_out)

    rows = []
    for step, ckpt_name, ckpt_path in ckpts:
        ckpt_tag = sanitize_tag(ckpt_name)
        out_results_path = str(root_out / f"{base_name}_{ckpt_tag}_results.json")
        out_summary_path = str(root_out / f"{base_name}_{ckpt_tag}_summary.json")

        summary_path = Path(out_summary_path)
        if args.reuse_existing_results and summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
            acc = summary.get("accuracy", None)
            print(f"[dataset={dataset_tag}] [ckpt] Reusing {ckpt_name}: accuracy={acc}")
        else:
            print(f"[dataset={dataset_tag}] [ckpt] Evaluating {ckpt_name} @ step={step} …")
            summary = evaluate_one_model(
                items=items,
                model_id=ckpt_path,
                output_dir=str(root_out),
                out_results_path=out_results_path,
                out_summary_path=out_summary_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                thinking_budget=args.thinking_budget,
                include_none_of_the_above=args.include_none_of_the_above,
                structured_prompting=args.structured_prompting,
                enable_randomness=args.enable_randomness,
                k_times=args.k_times,
                save_every=args.save_every,
            )
            acc = summary.get("accuracy", None)

        rows.append((
            step,
            ckpt_name,
            acc,
            summary.get("majority_vote_accuracy"),
            summary.get("oracle_accuracy"),
        ))

    csv_path = root_out / "summary_per_checkpoint.csv"
    with open(csv_path, "w") as f:
        if args.k_times > 1:
            f.write("checkpoint,step,accuracy,majority_vote_accuracy,oracle_accuracy\n")
            for step, name, acc, majority_acc, oracle_acc in rows:
                f.write(
                    f"{name},{step},{'' if acc is None else acc},"
                    f"{'' if majority_acc is None else majority_acc},"
                    f"{'' if oracle_acc is None else oracle_acc}\n"
                )
        else:
            f.write("checkpoint,step,accuracy\n")
            for step, name, acc, _, _ in rows:
                f.write(f"{name},{step},{'' if acc is None else acc}\n")
    print(f"[dataset={dataset_tag}] [ckpt] Wrote rollup CSV: {csv_path}")

    xs = [step for step, _, acc, _, _ in rows if isinstance(acc, (int, float))]
    ys = [acc for _, _, acc, _, _ in rows if isinstance(acc, (int, float))]
    if xs and ys:
        plot_path = try_plot_xy(
            xs, ys,
            title=f"Accuracy vs. Checkpoint • {Path(base_dir).name} • {dataset_tag}",
            out_path=root_out / "accuracy_vs_checkpoint",
            fmt=args.plot_format,
        )
        if plot_path:
            print(f"[dataset={dataset_tag}] [ckpt] Wrote plot: {plot_path}")

    return {
        "dataset_tag": dataset_tag,
        "dataset_root": dataset_root,
        "mode": "checkpoint_sweep",
        "csv_path": str(csv_path),
        "rows": [
            {
                "step": step,
                "checkpoint": name,
                "accuracy": acc,
                "majority_vote_accuracy": majority_acc,
                "oracle_accuracy": oracle_acc,
            }
            for step, name, acc, majority_acc, oracle_acc in rows
        ],
    }

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Qwen2.5/3-VL on annotated pairs (A,B), with streaming progress and periodic checkpoints. Also supports checkpoint sweep."
    )
    ap.add_argument(
        "--dataset-root", default="/nfs_share4/code/om/hypersim/3d_eval_FULL",
        help="Root folder containing pair subfolders + annotations.json"
    )
    ap.add_argument(
        "--model-id", default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF model id (e.g., Qwen3-VL-*) OR a local directory. If --scan-checkpoints is set, this must be a directory containing checkpoints. Supports MODEL_DIR@CHECKPOINT shorthand."
    )
    ap.add_argument(
        "--output-dir", default="/nfs_share4/code/om/hypersim/results_eval_colm/hamed-newprompt",
        help="Directory where to save outputs"
    )
    ap.add_argument(
        "--prompt", default=PROMPT,
        help="Text instruction sent with A and B"
    )
    ap.add_argument(
        "--structured-prompting", action="store_true",
        help="Run structured prompting as separate model queries: describe Image A, describe Image B, list correspondences, compare objects, then emit Final Answer."
    )
    ap.add_argument(
        "--include-none-of-the-above", action="store_true",
        help=f"Add a None of the above forced-choice option, emitted as {NONE_OF_THE_ABOVE_TOKEN}."
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max new tokens for each generation call. In --structured-prompting mode this applies to each structured step query."
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy)"
    )
    ap.add_argument(
        "--enable-randomness", action="store_true",
        help="Enable stochastic decoding: do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.5."
    )
    ap.add_argument(
        "--k-times", type=int, default=1,
        help="Run each item K times. If K > 1, randomness is enabled and summaries include majority-vote and oracle accuracy."
    )
    ap.add_argument(
        "--save-every", type=int, default=100,
        help="Save partial results & summary every N examples (across all workers). Default 1 updates JSON after every result; set 0 to save only at the end."
    )
    ap.add_argument(
        "--thinking-budget", type=int, default=16384,
        help="If >0 and model name contains 'Thinking', limit first-pass thinking tokens to this budget; "
             "if not finished and `</think>` not reached, append the early-stopping text and resume."
    )

    # Checkpoint sweep options
    ap.add_argument(
        "--scan-checkpoints", action="store_true",
        help="If set, treat --model-id as a directory and evaluate all checkpoints found within."
    )
    ap.add_argument(
        "--ckpt-name-regex", type=str, default=r"(?:checkpoint-(\d+))|(?:iter_(\d+))|(?:step-(\d+))",
        help=r"Regex capturing the step number. Examples: 'checkpoint-1000', 'iter_2000', 'step-3000'."
    )
    ap.add_argument(
        "--plot-format", type=str, default="png",
        help="Plot file format for accuracy vs. checkpoint (e.g., png, pdf, svg)."
    )
    ap.add_argument(
        "--reuse-existing-results", action="store_true",
        help="Skip a checkpoint if its summary.json already exists."
    )

    # single-checkpoint selector
    ap.add_argument(
        "--checkpoint", type=str, default=None,
        help="If set and --model-id is a directory, evaluate only this checkpoint subfolder. "
             "Accepts exact name or regex. Also supported via --model-id MODEL_DIR@CHECKPOINT."
    )

    args = ap.parse_args()
    args.k_times = max(1, int(args.k_times))
    if args.k_times > 1:
        args.enable_randomness = True
    if args.structured_prompting:
        print(
            "[config] Structured prompting enabled: each structured step will run as a separate model query, "
            "then the final query will emit Final Answer.",
            flush=True,
        )
    if args.include_none_of_the_above:
        print(
            f"[config] None of the above option enabled "
            f"(model may answer {NONE_OF_THE_ABOVE_TOKEN}).",
            flush=True,
        )
    if args.enable_randomness:
        print(
            "[config] Randomness enabled: do_sample=True, greedy=False, "
            "temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, "
            "presence_penalty=1.5.",
            flush=True,
        )
    if args.k_times > 1:
        print(
            f"[config] k_times={args.k_times}; running each example {args.k_times} times "
            "and reporting majority-vote plus oracle accuracy.",
            flush=True,
        )

    # Parse MODEL_DIR@CHECKPOINT shorthand
    if "@" in args.model_id:
        base, ck = args.model_id.split("@", 1)
        args.model_id = base
        if not args.checkpoint:
            args.checkpoint = ck

    dataset_roots = discover_dataset_roots(args.dataset_root)
    if not dataset_roots:
        raise SystemExit(f"No datasets found under {args.dataset_root}")

    multi_dataset = len(dataset_roots) > 1
    rollup = []
    for dataset_tag, dataset_root in dataset_roots:
        rollup.append(run_for_dataset(args, dataset_root, dataset_tag, multi_dataset))

    if multi_dataset:
        multi_summary_path = Path(args.output_dir) / "multi_dataset_summary.json"
        ensure_dir(multi_summary_path.parent)
        with open(multi_summary_path, "w") as f:
            json.dump(rollup, f, indent=2)
        print(f"[multi] Wrote dataset rollup: {multi_summary_path}")

if __name__ == "__main__":
    main()
