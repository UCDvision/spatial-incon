#!/usr/bin/env python3
"""
Compile the main benchmark into one combined annotation file for the public eval
script, enriched with the grouping metadata used in the paper figures.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = DEFAULT_REPO_ROOT / "data" / "benchmark"
DEFAULT_METADATA = DEFAULT_REPO_ROOT / "metadata" / "pairs_with_wacky_roll.json"
DEFAULT_SCENE_LABELS = DEFAULT_REPO_ROOT / "metadata" / "scenes_labeled.json"
DEFAULT_OBJECT_CLASS_CACHE = DEFAULT_REPO_ROOT / "metadata" / "object_class_by_pair.json"
DEFAULT_OUTPUT = DEFAULT_REPO_ROOT / "public_eval_annotations.json"

LABEL_BUCKET_SIZES = [6, 5, 5, 6]
MISC_CLASS = "misc."
MISC_SCENE = "Misc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the public evaluation annotations for the main benchmark."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--scene-labels", type=Path, default=DEFAULT_SCENE_LABELS)
    parser.add_argument("--object-class-cache", type=Path, default=DEFAULT_OBJECT_CLASS_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_default_path(value: Path, default_value: Path, repo_root: Path) -> Path:
    if value != default_value:
        return value
    return default_value


def relative_from_output(output_path: Path, target_path: Path) -> str:
    return os.path.relpath(str(target_path.resolve()), start=str(output_path.resolve().parent))


def build_label_bucket_by_pair(num_labels_by_pair: Dict[str, int]) -> tuple[Dict[str, str], List[str]]:
    distinct_values = sorted(set(num_labels_by_pair.values()))
    if sum(LABEL_BUCKET_SIZES) != len(distinct_values):
        raise ValueError(
            f"Expected {sum(LABEL_BUCKET_SIZES)} distinct num_labels values, found {len(distinct_values)}"
        )

    bucket_by_num_labels: Dict[int, str] = {}
    bucket_labels: List[str] = []
    start = 0
    for bucket_size in LABEL_BUCKET_SIZES:
        values = distinct_values[start : start + bucket_size]
        bucket_label = f"{values[0]}-{values[-1]}"
        bucket_labels.append(bucket_label)
        for value in values:
            bucket_by_num_labels[value] = bucket_label
        start += bucket_size

    return (
        {pair_id: bucket_by_num_labels[value] for pair_id, value in num_labels_by_pair.items()},
        bucket_labels,
    )


def build_scene_class_by_pair(scene_labels_path: Path) -> tuple[Dict[str, str], List[str]]:
    data = json.loads(scene_labels_path.read_text())
    scene_class_by_pair: Dict[str, str] = {}
    for item in data:
        pair_id = item.get("pair_id")
        scene_category = item.get("scene_category")
        if not pair_id:
            continue
        if not isinstance(scene_category, str) or not scene_category.strip():
            scene_category = MISC_SCENE
        scene_class_by_pair[pair_id] = (
            MISC_SCENE if scene_category.strip().lower() in {"misc", "misc."} else scene_category.strip()
        )

    counts = Counter(scene_class_by_pair.values())
    order = sorted(
        counts,
        key=lambda category: (
            category == MISC_SCENE,
            -counts[category],
            category,
        ),
    )
    return scene_class_by_pair, order


def build_object_class_by_pair(object_class_cache_path: Path) -> tuple[Dict[str, str], List[str]]:
    raw_class_by_pair = json.loads(object_class_cache_path.read_text())
    counts = Counter(raw_class_by_pair.values())

    remapped: Dict[str, str] = {}
    for pair_id, class_name in raw_class_by_pair.items():
        if counts[class_name] < 10 or "other" in class_name:
            remapped[pair_id] = MISC_CLASS
        else:
            remapped[pair_id] = class_name

    remapped_counts = Counter(remapped.values())
    order = sorted(
        remapped_counts,
        key=lambda name: (
            name == MISC_CLASS,
            -remapped_counts[name],
            name,
        ),
    )
    return remapped, order


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = resolve_default_path(args.dataset_root, DEFAULT_DATASET_ROOT, repo_root)
    metadata_path = resolve_default_path(args.metadata, DEFAULT_METADATA, repo_root)
    scene_labels_path = resolve_default_path(args.scene_labels, DEFAULT_SCENE_LABELS, repo_root)
    object_class_cache_path = resolve_default_path(
        args.object_class_cache,
        DEFAULT_OBJECT_CLASS_CACHE,
        repo_root,
    )
    output_path = resolve_default_path(args.output, DEFAULT_OUTPUT, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_records = json.loads(metadata_path.read_text())
    scene_class_by_pair, scene_order = build_scene_class_by_pair(scene_labels_path)
    object_class_by_pair, object_order = build_object_class_by_pair(object_class_cache_path)

    num_labels_by_pair = {
        record["pair_id"]: int(record.get("num_labels", 0))
        for record in metadata_records
    }
    num_labels_bucket_by_pair, num_labels_order = build_label_bucket_by_pair(num_labels_by_pair)
    num_labels_value_to_bucket = {
        str(num_labels): bucket_label
        for pair_id, num_labels in num_labels_by_pair.items()
        for bucket_label in [num_labels_bucket_by_pair[pair_id]]
    }

    records: List[Dict[str, object]] = []
    for record in metadata_records:
        pair_id = record["pair_id"]
        pair_dir = dataset_root / pair_id
        a_path = pair_dir / "A.jpg"
        b_path = pair_dir / "B.jpg"
        if not a_path.is_file() or not b_path.is_file():
            continue

        depth_bucket = (record.get("depth_label") or "unknown").strip().lower()
        lighting_bucket = (record.get("pair_label") or "unknown").strip().lower()
        plausibility_bucket = "pi" if bool(record.get("wacky_roll")) else "pp"
        num_labels = int(record.get("num_labels", 0))

        records.append(
            {
                "pair_id": pair_id,
                "scene_id": record.get("scene_id"),
                "mode": record.get("mode"),
                "A": record.get("A"),
                "B": record.get("B"),
                "C": record.get("C"),
                "moved_object_id": record.get("moved_object_id"),
                "moved_object_letter": record.get("moved_object_letter"),
                "num_labels": num_labels,
                "depth_bucket": depth_bucket,
                "lighting_bucket": lighting_bucket,
                "plausibility_bucket": plausibility_bucket,
                "scene_class": scene_class_by_pair.get(pair_id, MISC_SCENE),
                "object_class": object_class_by_pair.get(pair_id, MISC_CLASS),
                # Evaluation paths are deliberately relative to --pairs-root,
                # so a release can place the pair archive anywhere.
                "a_path": f"{pair_id}/A.jpg",
                "b_path": f"{pair_id}/B.jpg",
            }
        )

    payload = {
        "version": 2,
        "description": (
            "Combined public evaluation annotations for the main Spatial Inconsistency benchmark, "
            "including grouping metadata and image paths relative to the pair root."
        ),
        "records": records,
        "groupings": {
            "depth_bucket": {
                "display_name": "Object Depth",
                "order": ["low", "medium", "high"],
                "labels": {"low": "close", "medium": "medium", "high": "far"},
            },
            "lighting_bucket": {
                "display_name": "Scene Lighting",
                "order": ["low", "medium", "high"],
                "labels": {"low": "dark", "medium": "medium", "high": "bright"},
            },
            "plausibility_bucket": {
                "display_name": "Physical Plausibility",
                "order": ["pp", "pi"],
                "labels": {"pp": "plausible", "pi": "implausible"},
            },
            "num_labels": {
                "display_name": "Number of Labels",
                "order": num_labels_order,
                "labels": {bucket: bucket for bucket in num_labels_order},
                "source_field": "num_labels",
                "value_to_bucket": num_labels_value_to_bucket,
            },
            "object_class": {
                "display_name": "Object Class",
                "order": object_order,
                "labels": {bucket: bucket for bucket in object_order},
            },
            "scene_class": {
                "display_name": "Scene Class",
                "order": scene_order,
                "labels": {bucket: bucket for bucket in scene_order},
            },
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(records)} records to {output_path}")
    for grouping_key, grouping_spec in payload["groupings"].items():
        print(f"{grouping_key}: {len(grouping_spec['order'])} buckets")


if __name__ == "__main__":
    main()
