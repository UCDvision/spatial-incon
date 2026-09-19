#!/usr/bin/env python3
"""
Build self-paste ablations for an existing eval dataset.

For each requested K, this script creates a dataset root with the same structure
as the input dataset:

  <out-root>/annotations.json
  <out-root>/<pair_id>/A.jpg
  <out-root>/<pair_id>/B.jpg

`A.jpg` is copied unchanged from the source dataset. `B.jpg` starts from the
existing augmented image and adds self-pastes for K additional objects visible
in the B-frame instance map, excluding the main moved/inconsistent object.

Selection policy:
  - Candidate objects are all instance ids visible in B except -1 and the
    original `moved_object_id`.
  - One stable random ordering is generated per pair from `--seed`.
  - For K > 0, the first K objects from that ordering are used.
  - For K = -1, all available candidate objects are used.
  - If fewer than K candidates exist, all available candidates are used.

The output annotations preserve the original annotation fields and add one extra
field:

  "self_paste": {
    "mode": "same_spot_b_to_b",
    "requested_k": ...,
    "applied_k": ...,
    "object_ids": [...],
    "candidate_object_count": ...,
    "seed": ...
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "3d_eval_FULL"
DEFAULT_HYPERSIM_ROOT = REPO_ROOT
DEFAULT_OUT_PARENT = REPO_ROOT / "3d_eval_FULL_selfpaste"
EXPAND_FRAC = 0.05
DEFAULT_CPU_WORKERS = min(8, max(1, os.cpu_count() or 1))

_LAMA_CACHE: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create self-paste ablations for an eval dataset.",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Input dataset root containing annotations.json and pair folders.",
    )
    ap.add_argument(
        "--hypersim-root",
        type=Path,
        default=DEFAULT_HYPERSIM_ROOT,
        help="Root containing instance/<scene>/images/... HDF5 files.",
    )
    ap.add_argument(
        "--out-parent",
        type=Path,
        default=DEFAULT_OUT_PARENT,
        help="Parent directory under which one dataset root per K is created.",
    )
    ap.add_argument(
        "--k-values",
        nargs="+",
        required=True,
        help="One or more K values. Accepts space-separated ints and/or comma-separated lists, e.g. 1 3 5 -1 or 1,3,5,-1.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used to derive one stable object ordering per pair.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inpainting device: auto, cpu, cuda:0, ...",
    )
    ap.add_argument(
        "--inpaint-engine",
        type=str,
        choices=["lama", "opencv"],
        default="lama",
        help="Inpaint backend. 'lama' matches the original pipeline; 'opencv' is a lightweight fallback.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pairs to process, for smoke tests.",
    )
    ap.add_argument(
        "--pair-id",
        action="append",
        default=None,
        help="Optional specific pair_id to process. Can be passed multiple times.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output parent.",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Worker count. Use 0 for auto: all GPUs in GPU mode, or a small CPU pool otherwise.",
    )
    ap.add_argument(
        "--output-layout",
        type=str,
        choices=["nested", "siblings"],
        default="nested",
        help="Output layout: nested writes <out-parent>/k_1, siblings writes <prefix>_k1 style roots.",
    )
    ap.add_argument(
        "--sibling-parent",
        type=Path,
        default=None,
        help="Parent directory for sibling output roots. Defaults to the input dataset's parent.",
    )
    ap.add_argument(
        "--sibling-prefix",
        type=str,
        default=None,
        help="Prefix for sibling output roots. Defaults to the input dataset folder name.",
    )
    return ap.parse_args()


def parse_k_values(raw_values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    seen = set()
    for raw in raw_values:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if not piece:
                continue
            k = int(piece)
            if k == 0 or k < -1:
                raise ValueError(f"Invalid K value: {k}. Use positive integers or -1.")
            if k not in seen:
                parsed.append(k)
                seen.add(k)
    if not parsed:
        raise ValueError("No valid K values were provided.")
    return parsed


def available_cuda_devices() -> List[str]:
    try:
        if not torch.cuda.is_available():
            return []
        count = int(torch.cuda.device_count())
    except Exception:
        return []
    return [f"cuda:{i}" for i in range(max(0, count))]


def resolve_execution(
    device_arg: str,
    max_workers_arg: int,
    inpaint_engine: str,
) -> Tuple[str, List[str], int]:
    gpu_devices = available_cuda_devices() if inpaint_engine == "lama" else []

    if device_arg == "auto":
        if gpu_devices:
            gpu_count = len(gpu_devices) if max_workers_arg <= 0 else min(len(gpu_devices), max_workers_arg)
            return "gpu", gpu_devices[:gpu_count], gpu_count
        cpu_workers = DEFAULT_CPU_WORKERS if max_workers_arg <= 0 else max(1, max_workers_arg)
        return "cpu", ["cpu"], cpu_workers

    if device_arg in {"cuda", "cuda:all", "all-gpus"}:
        if not gpu_devices:
            raise ValueError("CUDA was requested but no GPUs are available.")
        gpu_count = len(gpu_devices) if max_workers_arg <= 0 else min(len(gpu_devices), max_workers_arg)
        return "gpu", gpu_devices[:gpu_count], gpu_count

    if str(device_arg).startswith("cuda"):
        if max_workers_arg not in (0, 1):
            raise ValueError(
                "A specific CUDA device can only be used with --max-workers 1. "
                "Use --device auto or --device cuda:all to use multiple GPUs."
            )
        return "single", [device_arg], 1

    cpu_workers = DEFAULT_CPU_WORKERS if max_workers_arg <= 0 else max(1, max_workers_arg)
    return "cpu", [device_arg], cpu_workers


def normalize_cam_id(cam: str) -> str:
    s = str(cam).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        raise ValueError(f"Cannot normalize camera id: {cam}")
    return digits.zfill(2)


def resolve_inst_path(hypersim_root: Path, scene_id: str, cam: str, frame: str) -> Path:
    cam_digits = normalize_cam_id(cam)
    return (
        hypersim_root
        / "instance"
        / scene_id
        / "images"
        / f"scene_cam_{cam_digits}_geometry_hdf5"
        / f"frame.{frame}.semantic_instance.hdf5"
    )


def read_instance_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "dataset" in f:
            arr = f["dataset"][:]
        else:
            arr = None
            for key in f.keys():
                try:
                    arr = f[key][:]
                    break
                except Exception:
                    continue
            if arr is None:
                raise KeyError(f"No dataset found in {path}")
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        arr = np.rint(arr)
    return arr.astype(np.int64, copy=False)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    frac: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    dx = int(round(frac * w))
    dy = int(round(frac * h))
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(width, x + w + dx)
    y1 = min(height, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


def stable_pair_rng(seed: int, pair_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def get_lama(device: str):
    if device not in _LAMA_CACHE:
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:
            raise ImportError(
                "simple_lama_inpainting is required for this script."
            ) from exc
        if str(device).startswith("cuda"):
            try:
                torch.cuda.set_device(device)
            except Exception as exc:
                raise RuntimeError(f"Failed to select CUDA device {device}") from exc
        _LAMA_CACHE[device] = SimpleLama(device=device)
    return _LAMA_CACHE[device]


def inpaint_with_lama(image_rgb: Image.Image, mask_l: Image.Image, device: str) -> Image.Image:
    lama = get_lama(device)
    return lama(image_rgb, mask_l)


def inpaint_with_opencv(image_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    mask_np = np.array(mask_l)
    inpainted_bgr = cv2.inpaint(image_bgr, mask_np, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)


def inpaint_image(image_rgb: Image.Image, mask_l: Image.Image, device: str, engine: str) -> Image.Image:
    if engine == "lama":
        return inpaint_with_lama(image_rgb, mask_l, device)
    if engine == "opencv":
        return inpaint_with_opencv(image_rgb, mask_l)
    raise ValueError(f"Unsupported inpaint engine: {engine}")


def choose_object_ids(
    inst_map: np.ndarray,
    moved_object_id: int,
    pair_id: str,
    seed: int,
    k: int,
) -> Tuple[List[int], int]:
    candidate_ids = sorted(
        int(oid)
        for oid in np.unique(inst_map)
        if int(oid) != -1 and int(oid) != int(moved_object_id)
    )
    rng = stable_pair_rng(seed, pair_id)
    rng.shuffle(candidate_ids)
    if k == -1:
        selected = candidate_ids
    else:
        selected = candidate_ids[: min(k, len(candidate_ids))]
    return selected, len(candidate_ids)


def apply_self_paste(
    base_image: Image.Image,
    inst_map: np.ndarray,
    object_ids: Sequence[int],
    device: str,
    inpaint_engine: str,
) -> Image.Image:
    if not object_ids:
        return base_image.copy()

    width, height = base_image.size
    base_np = np.array(base_image)
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    occluder_mask = np.zeros((height, width), dtype=np.uint8)
    donor_plans = []

    for object_id in object_ids:
        mask = (inst_map == int(object_id)).astype(np.uint8)
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue

        x, y, w, h = bbox
        donor_rgb_crop = base_image.crop((x, y, x + w, y + h))
        alpha_crop = Image.fromarray((mask[y : y + h, x : x + w] * 255).astype(np.uint8), mode="L")
        donor_plans.append((x, y, donor_rgb_crop.convert("RGBA"), alpha_crop))

        ix, iy, iw, ih = expand_bbox(bbox, EXPAND_FRAC, width, height)
        inpaint_mask[iy : iy + ih, ix : ix + iw] = 255

        inst_crop = inst_map[y : y + h, x : x + w]
        occ_crop = ((inst_crop != int(object_id)) & (inst_crop != -1)).astype(np.uint8) * 255
        occluder_mask[y : y + h, x : x + w] = np.maximum(
            occluder_mask[y : y + h, x : x + w],
            occ_crop,
        )

    if not donor_plans:
        return base_image.copy()

    inpainted = inpaint_image(
        base_image,
        Image.fromarray(inpaint_mask, mode="L"),
        device,
        inpaint_engine,
    )
    result = inpainted.copy()

    for x, y, donor_rgba, alpha_crop in donor_plans:
        result.paste(donor_rgba, (x, y), mask=alpha_crop)

    if np.any(occluder_mask):
        result.paste(Image.fromarray(base_np), (0, 0), mask=Image.fromarray(occluder_mask, mode="L"))

    return result


def k_dir_name(k: int) -> str:
    return "k_all" if k == -1 else f"k_{k}"


def k_suffix(k: int) -> str:
    return "kall" if k == -1 else f"k{k}"


def load_annotations(dataset_root: Path) -> List[dict]:
    ann_path = dataset_root / "annotations.json"
    with open(ann_path, "r") as f:
        return json.load(f)


def filter_annotations(
    annotations: Sequence[dict],
    pair_ids: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[dict]:
    filtered = list(annotations)
    if pair_ids:
        wanted = set(pair_ids)
        filtered = [rec for rec in filtered if rec["pair_id"] in wanted]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def resize_instance_if_needed(inst_map: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if inst_map.shape == (height, width):
        return inst_map
    resized = cv2.resize(inst_map.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int64, copy=False)


def prepare_output_dirs(
    dataset_root: Path,
    out_parent: Path,
    ks: Sequence[int],
    overwrite: bool,
    output_layout: str,
    sibling_parent: Optional[Path],
    sibling_prefix: Optional[str],
) -> Dict[int, Path]:
    out_roots: Dict[int, Path] = {}

    if output_layout == "nested":
        if out_parent.exists() and overwrite:
            shutil.rmtree(out_parent)
        out_parent.mkdir(parents=True, exist_ok=True)
        for k in ks:
            out_root = out_parent / k_dir_name(k)
            if out_root.exists() and any(out_root.iterdir()) and not overwrite:
                raise FileExistsError(
                    f"Output directory already exists and is non-empty: {out_root}. "
                    "Use --overwrite to replace it."
                )
            out_root.mkdir(parents=True, exist_ok=True)
            out_roots[k] = out_root
        return out_roots

    parent = sibling_parent or dataset_root.parent
    prefix = sibling_prefix or dataset_root.name
    parent.mkdir(parents=True, exist_ok=True)
    for k in ks:
        out_root = parent / f"{prefix}_{k_suffix(k)}"
        if out_root.exists() and overwrite:
            shutil.rmtree(out_root)
        if out_root.exists() and any(out_root.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {out_root}. "
                "Use --overwrite to replace it."
            )
        out_root.mkdir(parents=True, exist_ok=True)
        out_roots[k] = out_root
    return out_roots


def process_one_pair(
    rec: dict,
    dataset_root: Path,
    hypersim_root: Path,
    out_roots: Dict[int, Path],
    ks: Sequence[int],
    seed: int,
    device: str,
    inpaint_engine: str,
) -> Dict[int, dict]:
    pair_id = rec["pair_id"]
    pair_in_dir = dataset_root / pair_id
    a_in_path = pair_in_dir / "A.jpg"
    b_in_path = pair_in_dir / "B.jpg"

    if not a_in_path.is_file() or not b_in_path.is_file():
        raise FileNotFoundError(f"Missing A.jpg or B.jpg for pair: {pair_id}")

    base_b = Image.open(b_in_path).convert("RGB")
    inst_path = resolve_inst_path(
        hypersim_root,
        rec["scene_id"],
        rec["B"]["cam"],
        rec["B"]["frame"],
    )
    inst_map = read_instance_h5(inst_path)
    inst_map = resize_instance_if_needed(inst_map, base_b.size)

    image_cache: Dict[Tuple[int, ...], Image.Image] = {}
    out_records: Dict[int, dict] = {}

    for k in ks:
        selected_ids, candidate_count = choose_object_ids(
            inst_map=inst_map,
            moved_object_id=rec["moved_object_id"],
            pair_id=pair_id,
            seed=seed,
            k=k,
        )

        key = tuple(selected_ids)
        if key not in image_cache:
            image_cache[key] = apply_self_paste(
                base_b,
                inst_map,
                selected_ids,
                device,
                inpaint_engine,
            )

        out_root = out_roots[k]
        pair_out_dir = out_root / pair_id
        pair_out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(a_in_path, pair_out_dir / "A.jpg")
        image_cache[key].save(pair_out_dir / "B.jpg", quality=95)

        out_rec = deepcopy(rec)
        out_rec["self_paste"] = {
            "mode": "same_spot_b_to_b",
            "inpaint_engine": inpaint_engine,
            "requested_k": int(k),
            "applied_k": int(len(selected_ids)),
            "object_ids": [int(oid) for oid in selected_ids],
            "candidate_object_count": int(candidate_count),
            "seed": int(seed),
        }
        out_records[k] = out_rec

    return out_records


def process_pair_batch(
    records: Sequence[dict],
    dataset_root: Path,
    hypersim_root: Path,
    out_roots: Dict[int, Path],
    ks: Sequence[int],
    seed: int,
    device: str,
    inpaint_engine: str,
) -> Dict[int, List[dict]]:
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    batch_out: Dict[int, List[dict]] = {k: [] for k in ks}
    for rec in records:
        pair_out = process_one_pair(
            rec=rec,
            dataset_root=dataset_root,
            hypersim_root=hypersim_root,
            out_roots=out_roots,
            ks=ks,
            seed=seed,
            device=device,
            inpaint_engine=inpaint_engine,
        )
        for k in ks:
            batch_out[k].append(pair_out[k])
    return batch_out


def main() -> None:
    args = parse_args()
    ks = parse_k_values(args.k_values)
    execution_mode, devices, worker_count = resolve_execution(
        args.device,
        int(args.max_workers),
        args.inpaint_engine,
    )

    annotations = load_annotations(args.dataset_root)
    annotations = filter_annotations(annotations, args.pair_id, args.limit)
    if not annotations:
        raise ValueError("No pairs matched the requested filters.")

    out_roots = prepare_output_dirs(
        dataset_root=args.dataset_root,
        out_parent=args.out_parent,
        ks=ks,
        overwrite=args.overwrite,
        output_layout=args.output_layout,
        sibling_parent=args.sibling_parent,
        sibling_prefix=args.sibling_prefix,
    )
    out_annotations: Dict[int, List[dict]] = {k: [] for k in ks}

    if execution_mode == "gpu":
        print(f"Using {len(devices)} GPU workers: {', '.join(devices)}")
        shards = [annotations[idx::len(devices)] for idx in range(len(devices))]
        futures = {}
        spawn_ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=spawn_ctx) as ex:
            for device, shard in zip(devices, shards):
                if not shard:
                    continue
                fut = ex.submit(
                    process_pair_batch,
                    shard,
                    args.dataset_root,
                    args.hypersim_root,
                    out_roots,
                    ks,
                    args.seed,
                    device,
                    args.inpaint_engine,
                )
                futures[fut] = device

            for fut in tqdm(as_completed(futures), total=len(futures), desc="GPU shards"):
                device = futures[fut]
                try:
                    batch_out = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed processing GPU shard on {device}") from exc
                for k in ks:
                    out_annotations[k].extend(batch_out[k])
    elif worker_count == 1:
        device = devices[0]
        print(f"Using single-worker mode on {device}")
        for rec in tqdm(annotations, desc="Pairs"):
            pair_out = process_one_pair(
                rec=rec,
                dataset_root=args.dataset_root,
                hypersim_root=args.hypersim_root,
                out_roots=out_roots,
                ks=ks,
                seed=args.seed,
                device=device,
                inpaint_engine=args.inpaint_engine,
            )
            for k in ks:
                out_annotations[k].append(pair_out[k])
    else:
        device = devices[0]
        print(f"Using {worker_count} CPU workers on device={device}")
        futures = {}
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            for rec in annotations:
                fut = ex.submit(
                    process_one_pair,
                    rec,
                    args.dataset_root,
                    args.hypersim_root,
                    out_roots,
                    ks,
                    args.seed,
                    device,
                    args.inpaint_engine,
                )
                futures[fut] = rec["pair_id"]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Pairs"):
                pair_id = futures[fut]
                try:
                    pair_out = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed processing pair {pair_id}") from exc
                for k in ks:
                    out_annotations[k].append(pair_out[k])

    for k, out_root in out_roots.items():
        with open(out_root / "annotations.json", "w") as f:
            ordered = sorted(out_annotations[k], key=lambda rec: rec["pair_id"])
            json.dump(ordered, f, indent=2)

    if args.output_layout == "nested":
        print(f"Wrote self-paste ablations to: {args.out_parent}")
    else:
        print("Wrote self-paste ablations to sibling dataset roots:")
    for k in ks:
        print(f"  {k_suffix(k)} -> {out_roots[k]}")


if __name__ == "__main__":
    main()
