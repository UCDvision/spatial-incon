#!/usr/bin/env python3
"""Validate the portable Spatial Inconsistency release layout."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "benchmark"

def fail(message: str) -> None:
    raise SystemExit(f"FAILED: {message}")

def main() -> None:
    annotations = json.loads((BENCHMARK / "annotations.json").read_text())
    public_annotations = json.loads((ROOT / "public_eval_annotations.json").read_text())
    if len(annotations) != 615 or len(public_annotations.get("records", [])) != 615:
        fail("expected 615 benchmark and public-annotation records")
    source_ids = {row["pair_id"] for row in annotations}
    public_ids = {row["pair_id"] for row in public_annotations["records"]}
    if source_ids != public_ids:
        fail("benchmark and public annotation pair IDs differ")
    for row in public_annotations["records"]:
        answer = row.get("moved_object_letter", "")
        if not isinstance(answer, str) or len(answer) != 1 or not answer.isalpha():
            fail(f"invalid answer for {row.get('pair_id')}")
        for field in ("a_path", "b_path"):
            path = BENCHMARK / row[field]
            if not path.is_file():
                fail(f"missing {field} for {row['pair_id']}: {path}")
    for key, spec in public_annotations.get("groupings", {}).items():
        order = set(spec.get("order", []))
        source_field = spec.get("source_field", key)
        mapping = spec.get("value_to_bucket", {})
        values = {mapping.get(str(row.get(source_field)), row.get(source_field)) for row in public_annotations["records"]}
        unknown = values - order
        if unknown:
            fail(f"unknown {key} buckets: {sorted(unknown)}")
    split_root = ROOT / "data" / "splits"
    if split_root.is_dir():
        meta = json.loads((split_root / "split_meta.json").read_text())
        if set(meta["train_scenes"]) & set(meta["test_scenes"]):
            fail("train/test scene overlap")
    print("OK: 615 benchmark pairs, portable annotations, groupings, and splits validated")

if __name__ == "__main__":
    main()
