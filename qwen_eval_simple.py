#!/usr/bin/env python3
"""
Qwen evaluation script for the Spatial Inconsistency benchmark.

"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROMPT = (
    "Here are two photos of a static scene from different views. "
    "However, between the two photos, one object was edited so that its position "
    "is inconsistent with the camera motion. Which letter marks the modified "
    "object? Use the labels from the first image only. If an object is visible in "
    "only one image, ignore it.\n\n"
    "Reply with exactly:\n"
    "Final Answer: <LETTER>"
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = SCRIPT_DIR / "public_eval_annotations.json"
FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*([A-Z])")
RANDOM_GENERATION_SETTINGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
}


@dataclass
class PairItem:
    pair_id: str
    a_path: Path
    b_path: Path
    gt_letter: str
    metadata: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen VL model from one combined Spatial Inconsistency annotation file."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Combined annotation JSON with image paths relative to --pairs-root.",
    )
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=None,
        help="Path to the folder containing the pair directories referenced by the annotation file.",
    )
    parser.add_argument(
        "--list-groupings",
        action="store_true",
        help="Print the grouping dimensions and bucket order found in the annotation file, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print dataset and grouping counts without loading a model.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", "cuda", or "cuda:0".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "qwen_eval_simple",
        help="Directory where predictions and summaries will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of examples.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--enable-randomness",
        action="store_true",
        help=(
            "Use sampled decoding with temperature=0.7, top_p=0.8, top_k=20, "
            "repetition_penalty=1.0, and presence_penalty=1.5 when supported."
        ),
    )
    parser.add_argument(
        "--k-times",
        type=int,
        default=1,
        help=(
            "Run each example this many times. Values greater than 1 automatically "
            "use randomized decoding and report majority-vote plus oracle accuracy."
        ),
    )
    args = parser.parse_args()
    if args.k_times < 1:
        parser.error("--k-times must be >= 1")
    return args


def resolve_device(device_arg: str) -> str:
    import torch

    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        return "cuda:0"
    return device_arg


def normalize_gt_letter(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().upper()
    if not value:
        return ""
    return value[0] if "A" <= value[0] <= "Z" else ""


def load_annotation_payload(annotation_path: Path) -> Dict[str, object]:
    payload = json.loads(annotation_path.read_text())
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"Unexpected annotation format in {annotation_path}")
    return payload


def resolve_record_path(annotation_path: Path, pairs_root: Path | None, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if pairs_root is not None:
        return (pairs_root / path).resolve()
    return (annotation_path.parent / path).resolve()


def load_items(
    annotation_path: Path,
    pairs_root: Path | None,
    limit: int | None = None,
) -> List[PairItem]:
    payload = load_annotation_payload(annotation_path)
    items: List[PairItem] = []
    for record in payload["records"]:
        items.append(
            PairItem(
                pair_id=record["pair_id"],
                a_path=resolve_record_path(annotation_path, pairs_root, record["a_path"]),
                b_path=resolve_record_path(annotation_path, pairs_root, record["b_path"]),
                gt_letter=normalize_gt_letter(record.get("moved_object_letter")),
                metadata=record,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def load_model_and_processor(model_id: str, device: str):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype = torch.float32
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def parse_prediction(text: str) -> str:
    if not isinstance(text, str):
        return ""

    match = FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1)

    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        if len(line) == 1 and "A" <= line <= "Z":
            return line

    for char in text:
        if "A" <= char <= "Z":
            return char
    return ""


def build_messages(prompt: str, image_a, image_b) -> List[Dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_a},
                {"type": "image", "image": image_b},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generation_kwargs_for(
    model,
    processor,
    max_new_tokens: int,
    temperature: float,
    enable_randomness: bool,
) -> Dict[str, object]:
    generation_kwargs: Dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if enable_randomness:
        generation_kwargs.update(RANDOM_GENERATION_SETTINGS)
        if not hasattr(model.generation_config, "presence_penalty"):
            generation_kwargs.pop("presence_penalty", None)
    else:
        generation_kwargs["do_sample"] = temperature > 0.0
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature
    return generation_kwargs


def generate_answer(
    model,
    processor,
    image_a,
    image_b,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    enable_randomness: bool,
) -> str:
    import torch

    messages = build_messages(prompt, image_a, image_b)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image_a, image_b], return_tensors="pt")
    inputs = inputs.to(model.device)
    input_length = int(inputs["input_ids"].shape[1])

    generation_kwargs = generation_kwargs_for(
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_randomness=enable_randomness,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0, input_length:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True)


def majority_vote(pred_letters: Sequence[str]) -> str:
    valid_letters = [letter for letter in pred_letters if isinstance(letter, str) and letter]
    if not valid_letters:
        return ""
    counts = Counter(valid_letters)
    best_count = max(counts.values())
    for letter in valid_letters:
        if counts[letter] == best_count:
            return letter
    return ""


def summarize_results(
    results: Sequence[Dict[str, object]],
    correct_key: str = "correct",
) -> Dict[str, object]:
    evaluated = [row for row in results if row.get("gt_letter")]
    num_correct = sum(1 for row in evaluated if row.get(correct_key))
    accuracy = (num_correct / len(evaluated)) if evaluated else None
    return {
        "num_pairs": len(results),
        "num_evaluated": len(evaluated),
        "num_correct": num_correct,
        "accuracy": accuracy,
    }


def chance_for_rows(rows: Sequence[Dict[str, object]]) -> float | None:
    values = []
    for row in rows:
        num_labels = row.get("num_labels")
        if isinstance(num_labels, int) and num_labels > 0:
            values.append(1.0 / num_labels)
    return (sum(values) / len(values)) if values else None


def bucket_for_row(row: Dict[str, object], grouping_key: str, grouping_spec: Dict[str, object]) -> str | None:
    source_field = grouping_spec.get("source_field")
    value_to_bucket = grouping_spec.get("value_to_bucket", {})
    if source_field is not None:
        value = row.get(source_field)
        if value is None:
            return None
        if value_to_bucket:
            return value_to_bucket.get(str(value))
        return value

    if grouping_key in row:
        return row.get(grouping_key)
    return None


def build_breakdowns(
    payload: Dict[str, object],
    results: Sequence[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    grouping_specs = payload.get("groupings", {})
    breakdowns: Dict[str, List[Dict[str, object]]] = {}

    for grouping_key, grouping_spec in grouping_specs.items():
        order = grouping_spec.get("order", [])
        labels = grouping_spec.get("labels", {})
        rows_for_grouping = []
        for bucket_key in order:
            bucket_rows = [
                row for row in results
                if bucket_for_row(row, grouping_key, grouping_spec) == bucket_key
            ]
            summary = summarize_results(bucket_rows)
            summary.update(
                {
                    "bucket_key": bucket_key,
                    "bucket_label": labels.get(bucket_key, bucket_key),
                    "random_chance": chance_for_rows(bucket_rows),
                }
            )
            rows_for_grouping.append(summary)
        breakdowns[grouping_key] = rows_for_grouping

    return breakdowns


def write_breakdowns_csv(out_path: Path, breakdowns: Dict[str, List[Dict[str, object]]]) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "grouping",
                "bucket_key",
                "bucket_label",
                "accuracy",
                "num_correct",
                "num_evaluated",
                "random_chance",
            ]
        )
        for grouping_key, rows in breakdowns.items():
            for row in rows:
                writer.writerow(
                    [
                        grouping_key,
                        row["bucket_key"],
                        row["bucket_label"],
                        "" if row["accuracy"] is None else f"{row['accuracy']:.6f}",
                        row["num_correct"],
                        row["num_evaluated"],
                        "" if row["random_chance"] is None else f"{row['random_chance']:.6f}",
                    ]
                )


def evaluate_all(
    items: Sequence[PairItem],
    annotation_path: Path,
    model,
    processor,
    output_dir: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    enable_randomness: bool,
    k_times: int,
) -> List[Dict[str, object]]:
    from PIL import Image
    from tqdm import tqdm

    results: List[Dict[str, object]] = []
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w") as jsonl_file:
        for item in tqdm(items, desc="eval"):
            attempts: List[Dict[str, object]] = []
            try:
                image_a = Image.open(item.a_path).convert("RGB")
                image_b = Image.open(item.b_path).convert("RGB")
                for run_index in range(k_times):
                    try:
                        raw_text = generate_answer(
                            model=model,
                            processor=processor,
                            image_a=image_a,
                            image_b=image_b,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            enable_randomness=enable_randomness,
                        )
                        pred_letter = parse_prediction(raw_text)
                        error = ""
                    except Exception as exc:
                        raw_text = ""
                        pred_letter = ""
                        error = repr(exc)
                    attempts.append(
                        {
                            "run_index": run_index,
                            "pred_letter": pred_letter,
                            "correct": pred_letter == item.gt_letter if item.gt_letter else False,
                            "raw_text": raw_text,
                            "error": error,
                        }
                    )
            except Exception as exc:
                attempts.append(
                    {
                        "run_index": 0,
                        "pred_letter": "",
                        "correct": False,
                        "raw_text": "",
                        "error": repr(exc),
                    }
                )

            pred_letters = [str(attempt["pred_letter"]) for attempt in attempts]
            pred_letter = majority_vote(pred_letters)
            oracle_correct = any(attempt.get("correct") for attempt in attempts)
            first_attempt = attempts[0] if attempts else {}
            errors = [str(attempt["error"]) for attempt in attempts if attempt.get("error")]
            # A per-attempt error is already captured above.  In particular, do
            # not reference the exception variable from an ``except`` block
            # here: it is undefined when every attempt succeeds.
            error = "; ".join(errors)

            row = {
                "pair_id": item.pair_id,
                "pred_letter": pred_letter,
                "gt_letter": item.gt_letter,
                "correct": pred_letter == item.gt_letter if item.gt_letter else False,
                "oracle_correct": oracle_correct,
                "raw_text": first_attempt.get("raw_text", ""),
                "error": error,
                "a_path": str(item.a_path),
                "b_path": str(item.b_path),
            }
            if k_times > 1:
                row["k_times"] = k_times
                row["sample_pred_letters"] = pred_letters
                row["attempts"] = attempts
            for key, value in item.metadata.items():
                if key not in row:
                    row[key] = value
            results.append(row)
            jsonl_file.write(json.dumps(row) + "\n")
    return results


def print_groupings(payload: Dict[str, object]) -> None:
    groupings = payload.get("groupings", {})
    for grouping_key, grouping_spec in groupings.items():
        order = grouping_spec.get("order", [])
        labels = grouping_spec.get("labels", {})
        bucket_text = ", ".join(f"{bucket} ({labels.get(bucket, bucket)})" for bucket in order)
        print(f"{grouping_key}: {bucket_text}")


def print_dry_run(payload: Dict[str, object]) -> None:
    records = payload["records"]
    print(f"num_records\t{len(records)}")
    for grouping_key, grouping_spec in payload.get("groupings", {}).items():
        order = grouping_spec.get("order", [])
        counts = {bucket: 0 for bucket in order}
        for record in records:
            bucket = bucket_for_row(record, grouping_key, grouping_spec)
            if bucket in counts:
                counts[bucket] += 1
        for bucket in order:
            print(f"{grouping_key}\t{bucket}\t{counts[bucket]}")


def main() -> None:
    args = parse_args()
    payload = load_annotation_payload(args.annotations)

    if args.list_groupings:
        print_groupings(payload)
        return

    if args.dry_run:
        print_dry_run(payload)
        return

    if args.pairs_root is None:
        raise SystemExit(
            "--pairs-root is required for evaluation and should point to the folder "
            "containing the pair directories referenced by the annotation file."
        )
    if not args.pairs_root.is_dir():
        raise SystemExit(f"--pairs-root does not exist or is not a directory: {args.pairs_root}")

    items = load_items(args.annotations, pairs_root=args.pairs_root, limit=args.limit)
    if not items:
        raise SystemExit("No items were found in the annotation file.")

    device = resolve_device(args.device)
    enable_randomness = args.enable_randomness or args.k_times > 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    print(f"Using device: {device}")
    if args.k_times > 1 and not args.enable_randomness:
        print("--k-times > 1 requested; enabling randomized decoding for repeated runs.")
    model, processor = load_model_and_processor(args.model_id, device)

    results = evaluate_all(
        items=items,
        annotation_path=args.annotations,
        model=model,
        processor=processor,
        output_dir=args.output_dir,
        prompt=PROMPT,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_randomness=enable_randomness,
        k_times=args.k_times,
    )

    overall = summarize_results(results)
    generation_kwargs = generation_kwargs_for(
        model=model,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_randomness=enable_randomness,
    )
    overall.update(
        {
            "annotation_path": str(args.annotations),
            "pairs_root": str(args.pairs_root.resolve()),
            "model_id": getattr(model.config, "_name_or_path", None) or "",
            "enable_randomness": enable_randomness,
            "k_times": args.k_times,
            "generation_settings": generation_kwargs,
            "requested_random_generation_settings": RANDOM_GENERATION_SETTINGS
            if enable_randomness
            else None,
            "results_path": str(args.output_dir / "results.json"),
            "results_jsonl_path": str(args.output_dir / "results.jsonl"),
            "summary_path": str(args.output_dir / "summary.json"),
            "breakdowns_path": str(args.output_dir / "breakdowns.json"),
            "breakdowns_csv_path": str(args.output_dir / "breakdowns.csv"),
            "random_chance": chance_for_rows(results),
        }
    )
    if args.k_times > 1:
        oracle = summarize_results(results, correct_key="oracle_correct")
        overall.update(
            {
                "majority_vote_num_correct": overall["num_correct"],
                "majority_vote_accuracy": overall["accuracy"],
                "oracle_num_correct": oracle["num_correct"],
                "oracle_accuracy": oracle["accuracy"],
            }
        )
    breakdowns = build_breakdowns(payload, results)

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(overall, indent=2))
    (args.output_dir / "breakdowns.json").write_text(json.dumps(breakdowns, indent=2))
    write_breakdowns_csv(args.output_dir / "breakdowns.csv", breakdowns)

    acc = overall["accuracy"]
    print(
        f"overall majority vote: {overall['num_correct']}/{overall['num_evaluated']} correct "
        f"(accuracy={acc:.4f})" if isinstance(acc, float) else "overall: no scored examples"
    )
    if args.k_times > 1:
        oracle_acc = overall["oracle_accuracy"]
        print(
            f"overall oracle: {overall['oracle_num_correct']}/{overall['num_evaluated']} correct "
            f"(accuracy={oracle_acc:.4f})"
            if isinstance(oracle_acc, float)
            else "overall oracle: no scored examples"
        )
    print(f"Wrote results to {args.output_dir / 'results.json'}")
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")
    print(f"Wrote breakdowns to {args.output_dir / 'breakdowns.json'}")
    print(f"Wrote breakdowns CSV to {args.output_dir / 'breakdowns.csv'}")


if __name__ == "__main__":
    main()
