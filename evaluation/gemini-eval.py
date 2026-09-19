#!/usr/bin/env python3
"""
Evaluate Gemini (Gemini API) on an annotated pairs dataset (A.jpg labeled, B.jpg augmented),
with a global rate limit of 100 requests per minute and infinite retry on API errors.
"""

import os
import io
import json
import re
import argparse
import time
import threading
from typing import List, Dict, Any, Tuple, Any as AnyType
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from pathlib import Path

# ---------------------------
# Rate limiter
# ---------------------------
class RateLimiter:
    """Thread-safe sliding-window rate limiter."""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = deque()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            # drop old timestamps
            while self.calls and now - self.calls[0] >= self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                to_wait = self.period - (now - self.calls[0])
                time.sleep(to_wait)
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= self.period:
                    self.calls.popleft()
            self.calls.append(now)

# will be initialized in main()
limiter: RateLimiter

# ---------------------------
# Gemini API (google-genai)
# ---------------------------
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise SystemExit(
        "The 'google-genai' package is required. Install with:\n  pip install -U google-genai\n"
        f"Import error: {e}"
    )

# ---------------------------
# Data loading
# ---------------------------
PROMPT = (
    "Here are two photos of a static scene from different views. However, in between taking the photos, "
    "I edited the second image such that, for one object, its positioning is inconsistent with the camera "
    "motion between the two frames. Which letter marks the modified object? Please use the labels from the first image."
)

@dataclass
class PairItem:
    pair_id: str
    pair_dir: str
    A_path: str
    B_path: str
    gt_letter: str
    meta: Dict[str, Any]

def load_dataset(dataset_root: str) -> List[PairItem]:
    ann_path = os.path.join(dataset_root, "annotations.json")
    with open(ann_path, "r") as f:
        anns = json.load(f)
    items = []
    for rec in anns:
        pair_id = rec["pair_id"]
        dir_ = os.path.join(dataset_root, pair_id)
        A = os.path.join(dir_, "A.jpg")
        B = os.path.join(dir_, "B.jpg")
        if not (os.path.isfile(A) and os.path.isfile(B)):
            continue
        gt = rec.get("moved_object_letter")
        if isinstance(gt, str) and gt.strip():
            gt = gt.strip().upper()[0]
            if not ("A" <= gt <= "Z"):
                gt = None
        else:
            gt = None
        items.append(PairItem(pair_id, dir_, A, B, gt, rec))
    return items

def discover_dataset_roots(dataset_root: str) -> List[Tuple[str, str]]:
    root = Path(dataset_root)
    if (root / "annotations.json").is_file():
        return [(root.name, str(root))]
    children = sorted(p for p in root.iterdir() if p.is_dir() and (p / "annotations.json").is_file())
    return [(p.name, str(p)) for p in children]

# ---------------------------
# Helpers
# ---------------------------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def img_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def build_gemini_contents(prompt, imgA, imgB):
    strict = (
        f"{prompt}\n\n"
        "Format your reply exactly as:\n"
        "Final Answer: <LETTER>\n"
        "…where <LETTER> is ONE capital letter A to Z only. Do not write anything else."
    )
    return [
        types.Part.from_bytes(data=img_to_bytes(imgA), mime_type="image/jpeg"),
        types.Part.from_bytes(data=img_to_bytes(imgB), mime_type="image/jpeg"),
        types.Part.from_text(text=strict),
    ]

FINAL_RE = re.compile(r"Final Answer:\s*([A-Z])")

def parse_letter_from_generated(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    m = FINAL_RE.search(txt)
    if m:
        return m.group(1)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if len(last) == 1 and "A" <= last <= "Z":
            return last
        for ch in last:
            if "A" <= ch <= "Z":
                return ch
    for ch in txt:
        if "A" <= ch <= "Z":
            return ch
    return ""

def extract_text_and_reasoning(resp) -> Tuple[str, str]:
    out, reasoning = "", ""
    try:
        out = getattr(resp, "text", "") or ""
        cands = getattr(resp, "candidates", None)
        if isinstance(cands, list) and cands:
            parts = getattr(cands[0].content, "parts", None)
            if isinstance(parts, list):
                for p in parts:
                    if getattr(p, "thought", False):
                        t = getattr(p, "text", "")
                        reasoning += t + "\n"
    except:
        pass
    return out.strip(), reasoning.strip()

def response_to_dict(resp):
    try:
        return json.loads(resp.model_dump_json())
    except:
        try:
            return resp.model_dump()
        except:
            return repr(resp)

def thinking_level_to_budget(level: str) -> int:
    lvl = level.lower().strip()
    return {"low": 8192, "medium": 16384, "high": 32768}.get(lvl, 8192)

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evald = [r for r in results if r.get("gt_letter")]
    correct = sum(1 for r in evald if r.get("correct"))
    acc = correct / len(evald) if evald else None
    lats = [r["timing"]["latency_s"] for r in results if r.get("timing")]
    lats_sorted = sorted(lats)
    timing = {
        "count": len(lats_sorted),
        "mean_s": sum(lats_sorted)/len(lats_sorted) if lats_sorted else None,
        "p50_s": lats_sorted[len(lats_sorted)//2] if lats_sorted else None,
        "p90_s": lats_sorted[int(0.9*(len(lats_sorted)-1))] if lats_sorted else None,
        "max_s": max(lats_sorted) if lats_sorted else None,
    }
    return {
        "num_pairs": len(results),
        "num_evaluated": len(evald),
        "num_correct": correct,
        "accuracy": acc,
        "timing": timing,
    }

def add_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(os.path.dirname(path), base + suffix + ext)


def evaluate_dataset(
    *,
    dataset_root: str,
    dataset_tag: str,
    model_id: str,
    prompt: str,
    out_results_jsonl: str,
    out_results: str,
    out_summary: str,
    timeout: float,
    fail_fast: bool,
    concurrency: int,
    thinking_budget: int,
) -> Dict[str, Any]:
    items = load_dataset(dataset_root)
    if not items:
        raise SystemExit(f"No pairs under {dataset_root}")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    jsonl_path = out_results_jsonl
    json_path = out_results
    summary_path = out_summary

    for p in (jsonl_path, json_path, summary_path):
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    stop = False
    exit_code = 0

    jsonl_file = open(jsonl_path, "a", buffering=1)

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex, \
             tqdm(total=len(items), desc=f"{model_id}:{dataset_tag}") as pbar:

            futures = {
                ex.submit(
                    run_inference_pair_worker,
                    api_key,
                    it,
                    model_id,
                    prompt,
                    thinking_budget,
                    timeout
                ): it
                for it in items
            }

            for fut in as_completed(futures):
                it = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        "pair_id": it.pair_id,
                        "error": f"Worker crash: {e!r}",
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
                    if stop:
                        continue

                    results.append(r)
                    jsonl_file.write(json.dumps(r) + "\n")

                    with open(json_path, "w") as f:
                        json.dump(results, f, indent=2)

                    summary = summarize(results)
                    summary["_run_config"] = {
                        "model_id": model_id,
                        "thinking_budget": thinking_budget,
                        "dataset_root": dataset_root,
                        "dataset_tag": dataset_tag,
                        "concurrency": concurrency,
                        "fail_fast": fail_fast,
                    }
                    with open(summary_path, "w") as f:
                        json.dump(summary, f, indent=2)

                    pbar.update(1)

                    if fail_fast:
                        err = bool(r.get("error"))
                        both_empty = not r.get("raw_text") and not r.get("reasoning_text")
                        if err or both_empty:
                            stop = True
                            exit_code = 1 if err else 2
                            for other in futures:
                                if other is not fut:
                                    other.cancel()
                            break

        final = summarize(results)
        final["_run_config"] = {
            "model_id": model_id,
            "thinking_budget": thinking_budget,
            "dataset_root": dataset_root,
            "dataset_tag": dataset_tag,
            "concurrency": concurrency,
            "fail_fast": fail_fast,
        }
        if not stop:
            print(f"\n=== Evaluation Summary: {dataset_tag} ===")
            print(json.dumps(final, indent=2))
            print(f"\nResults JSONL: {jsonl_path}")
            print(f"Details JSON: {json_path}")
            print(f"Summary JSON: {summary_path}")
            return final

        print(f"\n=== Early Stop: {dataset_tag} ===")
        print(json.dumps(final, indent=2))
        raise SystemExit(exit_code)

    finally:
        jsonl_file.close()

# ---------------------------
# Worker
# ---------------------------
def run_inference_pair_worker(
    api_key: str,
    it: PairItem,
    model_id: str,
    prompt: str,
    thinking_budget: int,
    request_timeout: float,
) -> Dict[str, Any]:

    start_ts = iso_now()
    start_perf = time.perf_counter()

    # load images
    try:
        imgA = Image.open(it.A_path).convert("RGB")
        imgB = Image.open(it.B_path).convert("RGB")
    except Exception as e:
        return {
            "pair_id": it.pair_id,
            "error": f"Image load failed: {e!r}",
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
                "latency_s": round((time.perf_counter() - start_perf), 3),
            },
            "request_meta": None,
        }

    contents = build_gemini_contents(prompt, imgA, imgB)

    # loop until success
    while True:
        limiter.acquire()
        try:
            client = genai.Client(api_key=api_key)
            cfg = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True,
                ),
                temperature=0.0,
            )
            resp = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=cfg,
            )
            break
        except Exception as e:
            # print error and retry after 60s
            print(f"[{it.pair_id}] API error: {e!r}. Retrying in 60s...", flush=True)
            time.sleep(60)

    # parse response
    out_txt, reasoning = extract_text_and_reasoning(resp)
    pred = parse_letter_from_generated(out_txt)
    correct = (pred == it.gt_letter) if it.gt_letter else False

    end_perf = time.perf_counter()
    return {
        "pair_id": it.pair_id,
        "pred_letter": pred,
        "gt_letter": it.gt_letter,
        "correct": correct,
        "raw_text": out_txt,
        "reasoning_text": reasoning,
        "raw_response": response_to_dict(resp),
        "a_path": it.A_path,
        "b_path": it.B_path,
        "timing": {
            "start_utc": start_ts,
            "end_utc": iso_now(),
            "latency_s": round((end_perf - start_perf), 3),
        },
        "request_meta": {
            "model": model_id,
            "usage": getattr(resp, "usage", None),
        },
    }

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate Gemini on annotated pairs (A,B).")
    ap.add_argument("--dataset-root", default="/nfs_share5/code/om/dl3dv-10k/inconsistencies-50k")
    ap.add_argument("--model-id", default="gemini-2.5-pro")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out-results-jsonl", default="results_gemini-expandfrac.jsonl")
    ap.add_argument("--out-results", default="results_gemini-expandfrac.json")
    ap.add_argument("--out-summary", default="summary_gemini-expandfrac.json")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--fail-fast", action="store_true", default=False)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--thinking", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--thinking-budget-tokens", type=int, default=None)
    args = ap.parse_args()

    # initialize rate limiter: 100 calls / 60s
    global limiter
    limiter = RateLimiter(max_calls=100, period=60.0)

    if args.thinking_budget_tokens is not None:
        thinking_budget = max(0, args.thinking_budget_tokens)
        tb_name = f"{thinking_budget}tok"
    else:
        thinking_budget = thinking_level_to_budget(args.thinking)
        tb_name = args.thinking

    dataset_roots = discover_dataset_roots(args.dataset_root)
    if not dataset_roots:
        raise SystemExit(f"No datasets under {args.dataset_root}")

    multi = len(dataset_roots) > 1
    model_suffix = f"-{args.model_id.replace('/', '_')}-{tb_name}"
    rollup = []
    for dataset_tag, dataset_root in dataset_roots:
        dataset_suffix = f"-{dataset_tag}" if multi else ""
        summary = evaluate_dataset(
            dataset_root=dataset_root,
            dataset_tag=dataset_tag,
            model_id=args.model_id,
            prompt=args.prompt,
            out_results_jsonl=add_suffix(args.out_results_jsonl, dataset_suffix + model_suffix),
            out_results=add_suffix(args.out_results, dataset_suffix + model_suffix),
            out_summary=add_suffix(args.out_summary, dataset_suffix + model_suffix),
            timeout=args.timeout,
            fail_fast=args.fail_fast,
            concurrency=args.concurrency,
            thinking_budget=thinking_budget,
        )
        rollup.append({
            "dataset_tag": dataset_tag,
            "dataset_root": dataset_root,
            "accuracy": summary.get("accuracy"),
            "num_evaluated": summary.get("num_evaluated"),
            "num_correct": summary.get("num_correct"),
        })

    if multi:
        multi_summary_path = add_suffix(args.out_summary, model_suffix + "-multi")
        with open(multi_summary_path, "w") as f:
            json.dump(rollup, f, indent=2)
        print(f"\nSaved multi-dataset summary to: {multi_summary_path}")

if __name__ == "__main__":
    main()
