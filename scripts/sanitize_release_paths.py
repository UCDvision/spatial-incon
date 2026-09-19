#!/usr/bin/env python3
"""Remove machine-local paths from JSON and JSONL release artifacts."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PREFIX = "/nfs_share4/code/om/hypersim/"
DROP_KEYS = {"pair_folder", "A_src", "B_src", "a_path", "b_path"}

def transform(value, drop_paths: bool = False):
    if isinstance(value, dict):
        return {
            key: transform(item, drop_paths)
            for key, item in value.items()
            if not (drop_paths and key in DROP_KEYS)
        }
    if isinstance(value, list):
        return [transform(item, drop_paths) for item in value]
    if isinstance(value, str) and value.startswith(LOCAL_PREFIX):
        return value.replace(LOCAL_PREFIX, "<workspace>/", 1)
    return value

def rewrite_json(path: Path, drop_paths: bool = False) -> bool:
    try:
        original = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    updated = transform(original, drop_paths)
    if updated == original:
        return False
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(updated, indent=2) + "\n")
    tmp.replace(path)
    return True

def rewrite_jsonl(path: Path) -> bool:
    changed = False
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        updated = transform(item, True)
        changed |= updated != item
        lines.append(json.dumps(updated))
    if changed:
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(path)
    return changed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, help="Optional JSON/JSONL files to sanitize")
    args = parser.parse_args()
    rewritten = 0
    paths = args.paths or [*ROOT.rglob("*.json"), *ROOT.rglob("*.jsonl")]
    for path in paths:
        if path.suffix not in {".json", ".jsonl"}:
            continue
        drop = path.is_relative_to(ROOT / "results") or path.is_relative_to(ROOT / "metadata")
        rewritten += rewrite_jsonl(path) if path.suffix == ".jsonl" else rewrite_json(path, drop)
    print(f"sanitized {rewritten} artifacts")

if __name__ == "__main__":
    main()
