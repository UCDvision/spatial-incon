#!/usr/bin/env python3
"""
Scene-disjoint splitter for JSON A and JSON B.

- JSON A: list of objects with keys including 'scene_id' and 'pair_id'
- JSON B: list of objects with keys including 'id' (scene_id is derived as the prefix up to first "__")

Guarantees:
- Single scene split built from union of scenes in A and B
- No scene appears in both train and test
- Original ordering preserved within each file
- Seed fixed to 42 for reproducibility
- Fails hard on duplicates or malformed entries

Outputs:
- A_train.json, A_test.json, B_train.json, B_test.json
- split_meta.json (summary of scenes and counts)
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Any, Set, Tuple

SEED = 42  # fixed as requested

def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be a list/array")
    return data

def get_scene_from_A(item: Dict[str, Any], src: str) -> str:
    if "scene_id" not in item:
        raise ValueError(f"{src}: entry missing 'scene_id'")
    scene = item["scene_id"]
    if not isinstance(scene, str) or not scene:
        raise ValueError(f"{src}: 'scene_id' must be a non-empty string")
    # Consistency check: pair_id should start with "<scene_id>__"
    pid = item.get("pair_id")
    if pid is None or not isinstance(pid, str) or not pid:
        raise ValueError(f"{src}: entry missing or invalid 'pair_id'")
    if not pid.startswith(scene + "__"):
        raise ValueError(f"{src}: pair_id '{pid}' not consistent with scene_id '{scene}'")
    return scene

def get_scene_from_B(item: Dict[str, Any], src: str) -> str:
    if "id" not in item:
        raise ValueError(f"{src}: entry missing 'id'")
    _id = item["id"]
    if not isinstance(_id, str) or "__" not in _id:
        raise ValueError(f"{src}: 'id' must be a string containing '__' (got: {repr(_id)})")
    scene = _id.split("__", 1)[0]
    # Basic sanity check (allow general patterns but ensure non-empty)
    if not scene or not isinstance(scene, str):
        raise ValueError(f"{src}: derived scene_id is empty from id '{_id}'")
    return scene

def detect_duplicates(items: List[Dict[str, Any]], key: str, label: str, src_path: str) -> None:
    vals = []
    for i, it in enumerate(items):
        if key not in it or not isinstance(it[key], str) or not it[key]:
            raise ValueError(f"{src_path}: entry {i} missing/invalid '{key}'")
        vals.append(it[key])
    counts = Counter(vals)
    dups = [k for k, c in counts.items() if c > 1]
    if dups:
        raise ValueError(f"{src_path}: duplicate {label} values found: {dups[:5]}{' ...' if len(dups) > 5 else ''}")

def build_scene_sets(A: List[Dict[str, Any]], B: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str], Set[str]]:
    scenes_A = set()
    scenes_B = set()
    for i, it in enumerate(A):
        scenes_A.add(get_scene_from_A(it, f"JSON A idx {i}"))
    for j, it in enumerate(B):
        scenes_B.add(get_scene_from_B(it, f"JSON B idx {j}"))
    return scenes_A, scenes_B, scenes_A | scenes_B

def pick_train_scenes(all_scenes: List[str], train_ratio: float) -> Set[str]:
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1 (exclusive), got {train_ratio}")
    n = len(all_scenes)
    if n < 2:
        raise ValueError("Need at least 2 distinct scenes to create disjoint train/test splits.")
    # Reproducible shuffle using fixed seed
    import random
    rng = random.Random(SEED)
    shuffled = all_scenes[:]
    rng.shuffle(shuffled)
    # Round to nearest, but ensure at least 1 and at most n-1
    k = int(round(train_ratio * n))
    k = max(1, min(n - 1, k))
    return set(shuffled[:k])

def filter_by_scenes_preserve_order(items: List[Dict[str, Any]], scenes: Set[str], is_A: bool) -> List[Dict[str, Any]]:
    out = []
    for idx, it in enumerate(items):
        scene = get_scene_from_A(it, f"JSON A idx {idx}") if is_A else get_scene_from_B(it, f"JSON B idx {idx}")
        if scene in scenes:
            out.append(it)
    return out

def main():
    parser = argparse.ArgumentParser(description="Split JSON A & B into scene-disjoint train/test partitions.")
    parser.add_argument("--json-a", default="/nfs_share4/code/om/hypersim/3d_eval_FULL/annotations.json", help="Path to full JSON A (array).")
    parser.add_argument("--json-b", default="/nfs_share4/code/om/hypersim/3d_eval_auto/instructions.json", help="Path to full JSON B (array).")
    parser.add_argument("--out-dir", default="split_eval", help="Output directory for split files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio in (0,1), seed fixed to 42.")
    args = parser.parse_args()

    a_path = args.json_a
    b_path = args.json_b
    out_dir = args.out_dir
    train_ratio = args.train_ratio

    os.makedirs(out_dir, exist_ok=True)

    A = load_json_array(a_path)
    B = load_json_array(b_path)

    # Fail hard on duplicates
    detect_duplicates(A, key="pair_id", label="pair_id (A)", src_path=a_path)
    detect_duplicates(B, key="id", label="id (B)", src_path=b_path)

    # Build scene universe
    scenes_A, scenes_B, all_scenes = build_scene_sets(A, B)
    if not all_scenes:
        raise ValueError("No scenes found across A and B.")

    all_scenes_sorted = sorted(all_scenes)
    train_scenes = pick_train_scenes(all_scenes_sorted, train_ratio)
    test_scenes = set(all_scenes_sorted) - train_scenes

    # Filter while preserving original order
    A_train = filter_by_scenes_preserve_order(A, train_scenes, is_A=True)
    A_test  = filter_by_scenes_preserve_order(A, test_scenes,  is_A=True)
    B_train = filter_by_scenes_preserve_order(B, train_scenes, is_A=False)
    B_test  = filter_by_scenes_preserve_order(B, test_scenes,  is_A=False)

    # Small integrity checks
    # Ensure no overlap of scenes
    if train_scenes & test_scenes:
        raise AssertionError("Train/test scene sets overlap. This should be impossible.")
    # Ensure coverage of scenes present in files
    seen_A_train = {get_scene_from_A(it, "A_train") for it in A_train}
    seen_A_test  = {get_scene_from_A(it, "A_test") for it in A_test}
    seen_B_train = {get_scene_from_B(it, "B_train") for it in B_train}
    seen_B_test  = {get_scene_from_B(it, "B_test") for it in B_test}
    if (seen_A_train | seen_A_test) - scenes_A:
        raise AssertionError("Unexpected scenes appeared in A split.")
    if (seen_B_train | seen_B_test) - scenes_B:
        raise AssertionError("Unexpected scenes appeared in B split.")
    # Ensure every scene from each file appears entirely in exactly one split
    if (seen_A_train & seen_A_test) or (seen_B_train & seen_B_test):
        raise AssertionError("A scene from A or B appears in both train and test.")

    # Write outputs
    def dump(obj, name):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path

    pA_tr = dump(A_train, "A_train.json")
    pA_te = dump(A_test,  "A_test.json")
    pB_tr = dump(B_train, "B_train.json")
    pB_te = dump(B_test,  "B_test.json")

    meta = {
        "seed": SEED,
        "train_ratio_requested": train_ratio,
        "num_scenes_total": len(all_scenes_sorted),
        "num_scenes_train": len(train_scenes),
        "num_scenes_test": len(test_scenes),
        "train_scenes": sorted(train_scenes),
        "test_scenes": sorted(test_scenes),
        "counts": {
            "A": {"train": len(A_train), "test": len(A_test), "total": len(A)},
            "B": {"train": len(B_train), "test": len(B_test), "total": len(B)},
        },
        "inputs": {"json_a": a_path, "json_b": b_path},
        "outputs": {
            "A_train": pA_tr,
            "A_test":  pA_te,
            "B_train": pB_tr,
            "B_test":  pB_te,
        },
    }
    dump(meta, "split_meta.json")

    # Console summary
    print(json.dumps({
        "seed": SEED,
        "scenes_total": len(all_scenes_sorted),
        "scenes_train": len(train_scenes),
        "scenes_test": len(test_scenes),
        "A_counts": {"train": len(A_train), "test": len(A_test), "total": len(A)},
        "B_counts": {"train": len(B_train), "test": len(B_test), "total": len(B)},
        "out_dir": out_dir,
    }, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail hard with clear message
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
