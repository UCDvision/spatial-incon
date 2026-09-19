#!/usr/bin/env python3
"""Compute paper-style accuracy breakdowns from released prediction files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_rows(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows

def correct(row: dict, gt: str) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    return str(row.get("pred_letter", "")).strip().upper() == gt

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--annotations", type=Path, default=ROOT / "public_eval_annotations.json")
    parser.add_argument("--output", type=Path, default=Path("aggregate_results.csv"))
    args = parser.parse_args()

    payload = json.loads(args.annotations.read_text())
    metadata = {row["pair_id"]: row for row in payload["records"]}
    out = []
    for result_path in args.results:
        rows = [row for row in load_rows(result_path) if row.get("pair_id") in metadata]
        for key, spec in {"overall": {"order": ["all"]}, **payload["groupings"]}.items():
            for bucket in spec["order"]:
                selected = rows if key == "overall" else [
                    row for row in rows
                    if metadata[row["pair_id"]].get(spec.get("source_field", key)) == bucket
                    or spec.get("value_to_bucket", {}).get(str(metadata[row["pair_id"]].get(spec.get("source_field", key)))) == bucket
                ]
                n = len(selected)
                n_correct = sum(correct(row, metadata[row["pair_id"]]["moved_object_letter"]) for row in selected)
                out.append([result_path.name, key, bucket, n, n_correct, n_correct / n if n else ""])
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["result_file", "grouping", "bucket", "num_evaluated", "num_correct", "accuracy"])
        writer.writerows(out)
    print(f"wrote {len(out)} rows to {args.output}")

if __name__ == "__main__":
    main()
