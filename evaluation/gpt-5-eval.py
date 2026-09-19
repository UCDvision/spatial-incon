#!/usr/bin/env python3
"""
Evaluate gpt-5-nano (ChatGPT API) on an annotated pairs dataset (A.jpg labeled, B.jpg augmented).

Parallelized version:
  - Add --concurrency to run multiple requests at once (default: 1).
  - Streams JSONL as each finishes; rewrites pretty JSON and summary after each.
  - Fail-fast semantics preserved: first failure cancels the rest and exits non-zero.
"""

import os
import io
import json
import re
import argparse
import base64
import time
import threading
from typing import List, Dict, Any, Tuple, Any as AnyType
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter

# ---------------------------
# ChatGPT API (Responses)
# ---------------------------
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "The 'openai' package is required. Install with:\n  pip install -U openai\n"
        f"Import error: {e}"
    )

# ---------------------------
# Data loading
# ---------------------------
PROMPT = "Here are two photos of a static scene from different views. However, in between taking the photos, I edited the second image such that, for one object, its positioning is inconsistent with the camera motion between the two frames. Which letter marks the modified object? Please use the labels from the first image. "

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

NONE_OF_THE_ABOVE_TOKEN = "NONE_OF_THE_ABOVE"

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
        # normalize GT to single capital letter or None
        if isinstance(gt, str) and len(gt) >= 1:
            gt = gt.strip().upper()[0]
            if not ("A" <= gt <= "Z"):
                gt = None
        else:
            gt = None
        items.append(PairItem(pair_id, pair_dir, A_path, B_path, gt, rec))
    return items

def discover_dataset_roots(dataset_root: str) -> List[Tuple[str, str]]:
    root = Path(dataset_root)
    if (root / "annotations.json").is_file():
        return [(root.name, str(root))]
    children = sorted(p for p in root.iterdir() if p.is_dir() and (p / "annotations.json").is_file())
    return [(p.name, str(p)) for p in children]

# ---------------------------
# JSON safety helpers
# ---------------------------

def to_jsonable(obj):
    """
    Recursively convert SDK / pydantic objects to JSON-serializable types.
    Falls back to repr(...) for unknown objects.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    try:
        if hasattr(obj, "model_dump_json"):
            return json.loads(obj.model_dump_json())
    except Exception:
        pass
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
    except Exception:
        pass
    try:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            return {k: to_jsonable(v) for k, v in d.items()}
    except Exception:
        pass
    return repr(obj)

# ---------------------------
# OpenAI Responses: helpers
# ---------------------------

def b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def image_data_urls(imgA: Image.Image, imgB: Image.Image) -> Tuple[str, str]:
    return (
        f"data:image/jpeg;base64,{b64_jpeg(imgA)}",
        f"data:image/jpeg;base64,{b64_jpeg(imgB)}",
    )

def build_openai_input_from_text(text: str, a_data_url: str, b_data_url: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": a_data_url},
                {"type": "input_image", "image_url": b_data_url},
                {"type": "input_text", "text": text},
            ],
        }
    ]

def build_openai_input(
    prompt: str,
    imgA: Image.Image,
    imgB: Image.Image,
    include_none_of_the_above: bool = False,
) -> list:
    """
    Build the Responses API multimodal input:
      - Two images (A then B), then the strict instruction text.
    Keep the prompt and strict format exactly as specified.
    """
    if include_none_of_the_above:
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
        strict = (
            f"{prompt}\n\n"
            "Format your reply exactly as:\n"
            "Final Answer: <LETTER>\n"
            "…where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
        )
    a_data_url, b_data_url = image_data_urls(imgA, imgB)
    return build_openai_input_from_text(strict, a_data_url, b_data_url)

def build_structured_step_openai_input(
    base_prompt: str,
    a_data_url: str,
    b_data_url: str,
    step_index: int,
    prior_steps: List[Dict[str, str]],
    include_none_of_the_above: bool = False,
) -> list:
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

    return build_openai_input_from_text(step_prompt, a_data_url, b_data_url)

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
    for ch in generated_text.upper():
        if "A" <= ch <= "Z":
            return ch
    return ""

def call_openai_low_reasoning(
    client: OpenAI,
    *,
    model_id: str,
    messages: list,
    request_timeout: float,
) -> Any:
    # Let the model run until it finishes: do NOT set max_output_tokens or temperature.
    return client.responses.create(
        model=model_id,
        input=messages,
        timeout=request_timeout,
        reasoning={"effort": "low"}
    )

def run_structured_multi_query_openai(
    client: OpenAI,
    *,
    model_id: str,
    base_prompt: str,
    imgA: Image.Image,
    imgB: Image.Image,
    request_timeout: float,
    include_none_of_the_above: bool = False,
) -> Tuple[str, str, List[Dict[str, Any]], Any]:
    a_data_url, b_data_url = image_data_urls(imgA, imgB)
    trace: List[Dict[str, Any]] = []
    final_resp = None
    final_text = ""
    final_reasoning = ""

    for step_index, (step_name, _) in enumerate(STRUCTURED_PROMPT_STEPS):
        step_messages = build_structured_step_openai_input(
            base_prompt=base_prompt,
            a_data_url=a_data_url,
            b_data_url=b_data_url,
            step_index=step_index,
            prior_steps=trace,
            include_none_of_the_above=include_none_of_the_above,
        )
        resp = call_openai_low_reasoning(
            client,
            model_id=model_id,
            messages=step_messages,
            request_timeout=request_timeout,
        )
        output_text, reasoning_text = extract_text_and_reasoning(resp)
        trace.append({
            "step": str(step_index + 1),
            "step_name": step_name,
            "text": output_text,
            "reasoning_text": reasoning_text,
            "response_id": getattr(resp, "id", None),
            "usage": to_jsonable(getattr(resp, "usage", None)),
        })
        final_resp = resp
        final_text = output_text
        final_reasoning = reasoning_text

    return final_text, final_reasoning, trace, final_resp

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

def extract_text_and_reasoning(resp_obj: Any) -> Tuple[str, str]:
    """
    Extract flattened output text and any explicit 'reasoning' parts if present.
    """
    output_text = ""
    reasoning_text = ""
    try:
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            output_text = resp_obj.output_text
        if hasattr(resp_obj, "output") and isinstance(resp_obj.output, list):
            for block in resp_obj.output:
                content = getattr(block, "content", None)
                if isinstance(content, list):
                    for part in content:
                        ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                        if ptype == "reasoning":
                            text_val = getattr(part, "text", None)
                            if not text_val and isinstance(part, dict):
                                text_val = part.get("text") or part.get("reasoning") or ""
                            if text_val:
                                reasoning_text += (text_val + "\n")
    except Exception:
        pass

    # Fallback reconstruction if output_text was empty
    if not output_text and hasattr(resp_obj, "output") and isinstance(resp_obj.output, list):
        try:
            pieces = []
            for block in resp_obj.output:
                content = getattr(block, "content", None)
                if isinstance(content, list):
                    for part in content:
                        ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                        if ptype in ("output_text", "text"):
                            text_val = getattr(part, "text", None)
                            if not text_val and isinstance(part, dict):
                                text_val = part.get("text") or ""
                            if text_val:
                                pieces.append(text_val)
            output_text = "\n".join(pieces).strip()
        except Exception:
            pass
    return output_text or "", reasoning_text.strip()

def response_to_dict(resp_obj: Any) -> Dict[str, Any]:
    """
    Convert the SDK response to a plain dict for JSON logging.
    """
    try:
        return json.loads(resp_obj.model_dump_json())
    except Exception:
        try:
            return resp_obj.model_dump()
        except Exception:
            try:
                return json.loads(json.dumps(resp_obj, default=lambda o: getattr(o, "__dict__", str(o))))
            except Exception:
                return {"_unserializable_response_repr": repr(resp_obj)}

# ---------------------------
# Inference with ChatGPT
# ---------------------------

def run_inference_pair_worker(
    api_key: str,
    it: PairItem,
    model_id: str,
    prompt: str,
    include_none_of_the_above: bool,
    structured_prompting: bool,
    k_times: int,
    request_timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Worker-friendly wrapper: constructs its own OpenAI client to avoid
    sharing state across threads.
    """
    start_ts = iso_now()
    start_time = time.perf_counter()

    try:
        imgA = Image.open(it.A_path).convert("RGB")
        imgB = Image.open(it.B_path).convert("RGB")
    except Exception as e:
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        return {
            "pair_id": it.pair_id,
            "error": f"Image open failed: {repr(e)}",
            "pred_letter": "",
            "gt_letter": it.gt_letter,
            "correct": False,
            "raw_text": "",
            "reasoning_text": "",
            "raw_response": None,
            "a_path": it.A_path,
            "b_path": it.B_path,
            "timing": {
                "start_utc": start_ts,
                "end_utc": iso_now(),
                "latency_ms": latency_ms,
                "latency_s": round(latency_ms / 1000.0, 3),
            },
            "request_meta": None,
        }

    try:
        client = OpenAI(api_key=api_key)
        k_times = max(1, int(k_times))
        output_texts: List[str] = []
        reasoning_texts: List[str] = []
        raw_responses: List[Any] = []
        structured_traces: List[List[Dict[str, Any]]] = []
        preds: List[str] = []

        for _ in range(k_times):
            if structured_prompting:
                output_text, reasoning_text, structured_trace, resp = run_structured_multi_query_openai(
                    client,
                    model_id=model_id,
                    base_prompt=prompt,
                    imgA=imgA,
                    imgB=imgB,
                    request_timeout=request_timeout,
                    include_none_of_the_above=include_none_of_the_above,
                )
                structured_traces.append(structured_trace)
            else:
                messages = build_openai_input(prompt, imgA, imgB, include_none_of_the_above)
                resp = call_openai_low_reasoning(
                    client,
                    model_id=model_id,
                    messages=messages,
                    request_timeout=request_timeout,
                )
                output_text, reasoning_text = extract_text_and_reasoning(resp)

            output_texts.append(output_text)
            reasoning_texts.append(reasoning_text)
            raw_responses.append(resp)
            preds.append(parse_letter_from_generated(
                output_text,
                include_none_of_the_above,
                structured_prompting,
            ))

        pred = _majority_vote(preds)
        correct = (pred == it.gt_letter) if it.gt_letter else False
        oracle_correct = any(p == it.gt_letter for p in preds) if it.gt_letter else False

        output_text_out = output_texts[0] if len(output_texts) == 1 else output_texts
        reasoning_text_out = reasoning_texts[0] if len(reasoning_texts) == 1 else reasoning_texts
        raw_response_out = (
            response_to_dict(raw_responses[0])
            if len(raw_responses) == 1
            else [response_to_dict(resp) for resp in raw_responses]
        )
        last_resp = raw_responses[-1] if raw_responses else None

        usage = (
            to_jsonable(getattr(last_resp, "usage", None))
            if last_resp is not None
            else None
        )
        if len(raw_responses) > 1:
            usage = [to_jsonable(getattr(resp, "usage", None)) for resp in raw_responses]

        req_meta = {
            "response_id": getattr(last_resp, "id", None) if last_resp is not None else None,
            "model": getattr(last_resp, "model", model_id) if last_resp is not None else model_id,
            "system_fingerprint": getattr(last_resp, "system_fingerprint", None) if last_resp is not None else None,
            "usage": usage,
        }

        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)

        result = {
            "pair_id": it.pair_id,
            "pred_letter": pred,
            "gt_letter": it.gt_letter,
            "correct": bool(correct),
            "raw_text": output_text_out,
            "reasoning_text": reasoning_text_out,
            "raw_response": raw_response_out,
            "a_path": it.A_path,
            "b_path": it.B_path,
            "timing": {
                "start_utc": start_ts,
                "end_utc": iso_now(),
                "latency_ms": latency_ms,
                "latency_s": round(latency_ms / 1000.0, 3),
            },
            "request_meta": req_meta,
        }
        if structured_prompting:
            result["structured_queries"] = (
                structured_traces[0] if len(structured_traces) == 1 else structured_traces
            )
        if k_times > 1:
            result.update({
                "k_times": k_times,
                "sample_pred_letters": preds,
                "majority_vote_pred_letter": pred,
                "majority_vote_correct": bool(correct),
                "oracle_correct": bool(oracle_correct),
            })
        return result
    except Exception as e:
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        return {
            "pair_id": it.pair_id,
            "error": f"API call failed: {repr(e)}",
            "pred_letter": "",
            "gt_letter": it.gt_letter,
            "correct": False,
            "raw_text": "",
            "reasoning_text": "",
            "raw_response": None,
            "a_path": it.A_path,
            "b_path": it.B_path,
            "timing": {
                "start_utc": start_ts,
                "end_utc": iso_now(),
                "latency_ms": latency_ms,
                "latency_s": round(latency_ms / 1000.0, 3),
            },
            "request_meta": None,
        }

# ---------------------------
# Accuracy / rolling summary
# ---------------------------

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in results if r.get("gt_letter")]
    if not evaluated:
        return {"num_pairs": len(results), "num_evaluated": 0, "accuracy": None}
    correct = sum(1 for r in evaluated if r.get("correct"))
    acc = correct / max(1, len(evaluated))
    latencies = [r["timing"]["latency_s"] for r in results if r.get("timing") and r["timing"].get("latency_s") is not None]
    latencies_sorted = sorted(latencies) if latencies else []
    timing_summary = {
        "count": len(latencies_sorted),
        "mean_s": (sum(latencies_sorted) / len(latencies_sorted)) if latencies_sorted else None,
        "p50_s": (latencies_sorted[len(latencies_sorted)//2] if latencies_sorted else None),
        "p90_s": (latencies_sorted[int(0.9*(len(latencies_sorted)-1))] if latencies_sorted else None) if latencies_sorted else None,
        "max_s": (max(latencies_sorted) if latencies_sorted else None),
    }
    summary = {
        "num_pairs": len(results),
        "num_evaluated": len(evaluated),
        "num_correct": correct,
        "accuracy": acc,
        "timing": timing_summary
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

def add_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(os.path.dirname(path), base + suffix + ext)


def iter_jsonl_rows(path: str):
    if not os.path.isfile(path):
        return
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] Skipping invalid JSONL line {line_no} in {path}")
                continue
            if isinstance(row, dict):
                yield row


def load_existing_results(jsonl_path: str, json_path: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}

    for row in iter_jsonl_rows(jsonl_path) or []:
        pair_id = row.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            records[pair_id] = row

    if records:
        return records

    if not os.path.isfile(json_path):
        return records

    try:
        with open(json_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return records

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            pair_id = row.get("pair_id")
            if isinstance(pair_id, str) and pair_id:
                records[pair_id] = row
    return records


def is_complete_result(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("error"):
        return False
    raw_text = str(row.get("raw_text") or "").strip()
    reasoning_text = str(row.get("reasoning_text") or "").strip()
    return bool(raw_text or reasoning_text)


def build_ordered_results(
    items: List[PairItem],
    results_by_pair_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [to_jsonable(results_by_pair_id[it.pair_id]) for it in items if it.pair_id in results_by_pair_id]


def evaluate_dataset(
    *,
    dataset_root: str,
    dataset_tag: str,
    model_id: str,
    prompt: str,
    out_results: str,
    out_results_jsonl: str,
    out_summary: str,
    timeout: float,
    fail_fast: bool,
    concurrency: int,
    resume: bool,
    include_none_of_the_above: bool,
    structured_prompting: bool,
    k_times: int,
) -> Dict[str, Any]:
    items = load_dataset(dataset_root)
    if not items:
        raise SystemExit(f"No pairs found under {dataset_root}")

    jsonl_path = out_results_jsonl
    json_path = out_results
    summary_path = out_summary

    for p in [jsonl_path, json_path, summary_path]:
        d = os.path.dirname(os.path.abspath(p))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)

    existing_results = load_existing_results(jsonl_path, json_path) if resume else {}
    kept_results: Dict[str, Dict[str, Any]] = {}
    for it in items:
        row = existing_results.get(it.pair_id)
        if row and is_complete_result(row):
            kept_results[it.pair_id] = row
    pending_items = [it for it in items if it.pair_id not in kept_results]

    print(
        f"[resume:{dataset_tag}] total={len(items)} "
        f"kept_complete={len(kept_results)} pending={len(pending_items)}"
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if pending_items and not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    results_by_pair_id: Dict[str, Dict[str, Any]] = dict(kept_results)

    if resume:
        with open(jsonl_path, "w") as f:
            for it in items:
                row = kept_results.get(it.pair_id)
                if row is not None:
                    f.write(json.dumps(to_jsonable(row)) + "\n")
    else:
        # Ensure old content is cleared for a fresh run.
        open(jsonl_path, "w").close()

    def write_snapshot() -> Dict[str, Any]:
        ordered_results = build_ordered_results(items, results_by_pair_id)
        with open(json_path, "w") as f:
            json.dump(ordered_results, f, indent=2)
        summary = summarize(ordered_results)
        summary["_run_config"] = {
            "model_id": model_id,
            "dataset_root": dataset_root,
            "dataset_tag": dataset_tag,
            "concurrency": concurrency,
            "fail_fast": fail_fast,
            "resume": resume,
            "include_none_of_the_above": include_none_of_the_above,
            "structured_prompting": structured_prompting,
            "k_times": k_times,
        }
        with open(summary_path, "w") as f:
            json.dump(to_jsonable(summary), f, indent=2)
        return summary

    if not pending_items:
        final_summary = write_snapshot()
        print(f"\n=== Evaluation Summary (resume, no pending): {dataset_tag} ===")
        print(json.dumps(final_summary, indent=2))
        print(f"\nStreaming results: {jsonl_path}")
        print(f"Saved per-pair results to: {json_path}")
        print(f"Saved summary to: {summary_path}")
        return final_summary

    lock = threading.Lock()
    stop_requested = False
    exit_code = 0
    jsonl_file = open(jsonl_path, "a", buffering=1)

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex, tqdm(total=len(pending_items), desc=f"{model_id}:{dataset_tag}", smoothing=0.05) as pbar:
            futures = {
                ex.submit(
                    run_inference_pair_worker,
                    api_key,
                    it,
                    model_id,
                    prompt,
                    include_none_of_the_above,
                    structured_prompting,
                    k_times,
                    timeout
                ): it
                for it in pending_items
            }

            for fut in as_completed(futures):
                it = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        "pair_id": it.pair_id,
                        "error": f"Worker exception: {repr(e)}",
                        "pred_letter": "",
                        "gt_letter": it.gt_letter,
                        "correct": False,
                        "raw_text": "",
                        "reasoning_text": "",
                        "raw_response": None,
                        "a_path": it.A_path,
                        "b_path": it.B_path,
                        "timing": None,
                        "request_meta": None,
                    }

                with lock:
                    if stop_requested:
                        continue

                    results_by_pair_id[it.pair_id] = r
                    jsonl_file.write(json.dumps(to_jsonable(r)) + "\n")
                    jsonl_file.flush()

                    summary = write_snapshot()

                    pbar.update(1)

                    if fail_fast:
                        failed_api = bool(r.get("error"))
                        rt = (r.get("raw_text") or "").strip()
                        rz = (r.get("reasoning_text") or "").strip()
                        empty_both = (rt == "" and rz == "")

                        if failed_api or empty_both:
                            stop_requested = True
                            exit_code = 1 if failed_api else 2
                            for other in futures:
                                if other is not fut:
                                    other.cancel()
                            break

        final_summary = write_snapshot()

        if not stop_requested:
            print(f"\n=== Evaluation Summary: {dataset_tag} ===")
            print(json.dumps(final_summary, indent=2))
            print(f"\nStreaming results: {jsonl_path}")
            print(f"Saved per-pair results to: {json_path}")
            print(f"Saved summary to: {summary_path}")
            return final_summary

        print(f"\n=== Early Stop (fail-fast): {dataset_tag} ===")
        print(json.dumps(final_summary, indent=2))
        print(f"\nLast written files:\n  {jsonl_path}\n  {json_path}\n  {summary_path}")
        raise SystemExit(exit_code)

    finally:
        try:
            jsonl_file.close()
        except Exception:
            pass

# ---------------------------
# CLI / realtime writer
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate gpt-5-nano (ChatGPT API) on annotated pairs (A,B).")
    ap.add_argument("--dataset-root", default="/nfs_share5/code/om/dl3dv-10k/inconsistencies-50k",
                    help="Root folder containing pair subfolders + annotations.json")
    ap.add_argument("--model-id", default="gpt-5",
                    help="ChatGPT model id (default: gpt-5-nano)")
    ap.add_argument("--prompt", default=PROMPT, help="Text instruction sent with A and B")
    ap.add_argument("--structured-prompting", action="store_true",
                    help="Run structured prompting as separate model queries: describe Image A, describe Image B, list correspondences, compare objects, then emit Final Answer.")
    ap.add_argument("--include-none-of-the-above", action="store_true",
                    help=f"Add a None of the above forced-choice option, emitted as {NONE_OF_THE_ABOVE_TOKEN}.")
    ap.add_argument("--k-times", type=int, default=1,
                    help="Run each item K times. If K > 1, summaries include majority-vote and oracle accuracy.")
    ap.add_argument("--out-results", default="results_gpt5-low-expandfrac.json",
                    help="Pretty JSON; rewritten after each pair")
    ap.add_argument("--out-results-jsonl", default="results_gpt5-low-expandfrac.jsonl",
                    help="Streaming JSONL; appended in real time")
    ap.add_argument("--out-summary", default="sum_gpt5-low-expandfrac.json",
                    help="Accuracy summary; rewritten after each pair")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="Per-request timeout in seconds")
    ap.add_argument("--fail-fast", action="store_true", default=False,
                    help="Stop immediately on API error or empty output/reasoning")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="Number of concurrent requests (default: 1)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Start fresh instead of reusing complete rows from existing outputs.")
    args = ap.parse_args()
    args.k_times = max(1, int(args.k_times))

    if args.structured_prompting:
        print(
            "[config] Structured prompting enabled: each structured step will run as a separate low-reasoning GPT query, "
            "then the final query will emit Final Answer.",
            flush=True,
        )
    if args.k_times > 1:
        print(
            f"[config] k_times={args.k_times}; running each example {args.k_times} times "
            "and reporting majority-vote plus oracle accuracy.",
            flush=True,
        )
    if args.include_none_of_the_above:
        print(
            f"[config] None of the above option enabled "
            f"(model may answer {NONE_OF_THE_ABOVE_TOKEN}).",
            flush=True,
        )

    dataset_roots = discover_dataset_roots(args.dataset_root)
    if not dataset_roots:
        raise SystemExit(f"No datasets found under {args.dataset_root}")

    multi = len(dataset_roots) > 1
    rollup = []
    for dataset_tag, dataset_root in dataset_roots:
        suffix = f"-{dataset_tag}" if multi else ""
        summary = evaluate_dataset(
            dataset_root=dataset_root,
            dataset_tag=dataset_tag,
            model_id=args.model_id,
            prompt=args.prompt,
            out_results=add_suffix(args.out_results, suffix),
            out_results_jsonl=add_suffix(args.out_results_jsonl, suffix),
            out_summary=add_suffix(args.out_summary, suffix),
            timeout=args.timeout,
            fail_fast=args.fail_fast,
            concurrency=args.concurrency,
            resume=not args.no_resume,
            include_none_of_the_above=args.include_none_of_the_above,
            structured_prompting=args.structured_prompting,
            k_times=args.k_times,
        )
        rollup.append({
            "dataset_tag": dataset_tag,
            "dataset_root": dataset_root,
            "summary_path": add_suffix(args.out_summary, suffix),
            "accuracy": summary.get("accuracy"),
            "num_evaluated": summary.get("num_evaluated"),
            "num_correct": summary.get("num_correct"),
        })

    if multi:
        multi_summary_path = add_suffix(args.out_summary, "_multi")
        with open(multi_summary_path, "w") as f:
            json.dump(rollup, f, indent=2)
        print(f"\nSaved multi-dataset summary to: {multi_summary_path}")

if __name__ == "__main__":
    main()
