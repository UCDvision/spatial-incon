#!/usr/bin/env python3
"""
Build EXPAND_FRAC ablations for an existing eval dataset.

For each requested expand fraction, this script recreates the original moved-
object inconsistency from source artifacts while changing only the initial
inpaint expansion around the moved object. It writes one dataset root per
fraction with the same structure as the input dataset:

  <out-root>/annotations.json
  <out-root>/<pair_id>/A.jpg
  <out-root>/<pair_id>/B.jpg

Assumptions:
  - The input dataset points back to the original generator artifacts via
    `sources.pair_folder`.
  - The original donor object, paste size, and paste position should remain
    fixed. Only the inpaint expansion around the moved object changes.
"""

from __future__ import annotations

import argparse
import json
import os
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
from PIL import Image, ImageDraw
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "3d_eval_FULL"
DEFAULT_HYPERSIM_ROOT = REPO_ROOT
DEFAULT_OUT_PARENT = REPO_ROOT / "3d_eval_expandfrac"
DEFAULT_CPU_WORKERS = min(8, max(1, os.cpu_count() or 1))

_LAMA_CACHE: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create EXPAND_FRAC ablations for an eval dataset.")
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--hypersim-root", type=Path, default=DEFAULT_HYPERSIM_ROOT)
    ap.add_argument("--out-parent", type=Path, default=DEFAULT_OUT_PARENT)
    ap.add_argument(
        "--expand-fracs",
        nargs="+",
        required=True,
        help="One or more expansion fractions. Accepts space-separated floats and/or comma-separated lists.",
    )
    ap.add_argument("--device", type=str, default="auto", help="Inpainting device: auto, cpu, cuda:0, ...")
    ap.add_argument(
        "--inpaint-engine",
        type=str,
        choices=["lama", "opencv"],
        default="lama",
        help="Inpaint backend. 'lama' matches the original pipeline; 'opencv' is a lightweight fallback.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pair-id", action="append", default=None)
    ap.add_argument("--overwrite", action="store_true")
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
        help="Output layout: nested writes <out-parent>/frac_0p05, siblings writes <prefix>_frac_0p05 roots.",
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


def parse_expand_fracs(raw_values: Sequence[str]) -> List[float]:
    parsed: List[float] = []
    seen = set()
    for raw in raw_values:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if not piece:
                continue
            frac = float(piece)
            if frac < 0.0:
                raise ValueError(f"Invalid EXPAND_FRAC: {frac}. Must be >= 0.")
            key = round(frac, 8)
            if key not in seen:
                parsed.append(frac)
                seen.add(key)
    if not parsed:
        raise ValueError("No valid expand fractions were provided.")
    return parsed


def available_cuda_devices() -> List[str]:
    try:
        if not torch.cuda.is_available():
            return []
        count = int(torch.cuda.device_count())
    except Exception:
        return []
    return [f"cuda:{i}" for i in range(max(0, count))]


def resolve_execution(device_arg: str, max_workers_arg: int, inpaint_engine: str) -> Tuple[str, List[str], int]:
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
    if s.startswith("scene_cam_") and s.endswith("_final_preview"):
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        raise ValueError(f"Cannot normalize camera id: {cam}")
    return f"scene_cam_{digits.zfill(2)}_final_preview"


def resolve_rgb_path(hypersim_root: Path, scene_id: str, frame_id: str, cam: str) -> Path:
    cam_dir = normalize_cam_id(cam)
    base = hypersim_root / "rgb" / scene_id / "images" / cam_dir
    jpg = base / f"frame.{frame_id}.color.jpg"
    if jpg.is_file():
        return jpg
    png = base / f"frame.{frame_id}.color.png"
    if png.is_file():
        return png
    raise FileNotFoundError(f"RGB not found for {scene_id} {cam_dir} frame {frame_id}")


def resolve_inst_path(hypersim_root: Path, scene_id: str, frame_id: str, cam: str) -> Path:
    cam_dir = normalize_cam_id(cam).replace("final_preview", "geometry_hdf5")
    path = hypersim_root / "instance" / scene_id / "images" / cam_dir / f"frame.{frame_id}.semantic_instance.hdf5"
    if path.is_file():
        return path
    raise FileNotFoundError(f"Instance map not found for {scene_id} {cam_dir} frame {frame_id}")


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


def resize_instance_if_needed(inst_map: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if inst_map.shape == (height, width):
        return inst_map
    resized = cv2.resize(inst_map.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int64, copy=False)


def expand_bbox(bbox: Tuple[int, int, int, int], frac: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    dx = int(round(frac * w))
    dy = int(round(frac * h))
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(width, x + w + dx)
    y1 = min(height, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


def rect_union(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(0, min(ax, bx))
    y0 = max(0, min(ay, by))
    x1 = min(width, max(ax + aw, bx + bw))
    y1 = min(height, max(ay + ah, by + bh))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def crop_instance(img: Image.Image, inst_map: np.ndarray, object_id: int) -> Tuple[Image.Image, Image.Image]:
    mask = (inst_map == object_id).astype(np.uint8)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        raise ValueError(f"Empty donor instance for object {object_id}")
    x, y, w, h = bbox
    rgb_crop = img.crop((x, y, x + w, y + h))
    alpha = Image.fromarray((mask[y : y + h, x : x + w] * 255).astype(np.uint8), mode="L")
    return rgb_crop, alpha


def get_lama(device: str):
    if device not in _LAMA_CACHE:
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:
            raise ImportError("simple_lama_inpainting is required for this script.") from exc
        if str(device).startswith("cuda"):
            try:
                torch.cuda.set_device(device)
            except Exception as exc:
                raise RuntimeError(f"Failed to select CUDA device {device}") from exc
        _LAMA_CACHE[device] = SimpleLama(device=device)
    return _LAMA_CACHE[device]


def inpaint_with_lama(image_rgb: Image.Image, mask_l: Image.Image, device: str) -> Image.Image:
    return get_lama(device)(image_rgb, mask_l)


def inpaint_with_opencv(image_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    mask_np = np.array(mask_l)
    inpainted_bgr = cv2.inpaint(image_bgr, mask_np, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))


def inpaint_image(image_rgb: Image.Image, mask_l: Image.Image, device: str, engine: str) -> Image.Image:
    if engine == "lama":
        return inpaint_with_lama(image_rgb, mask_l, device)
    if engine == "opencv":
        return inpaint_with_opencv(image_rgb, mask_l)
    raise ValueError(f"Unsupported inpaint engine: {engine}")


def load_annotations(dataset_root: Path) -> List[dict]:
    with open(dataset_root / "annotations.json", "r") as f:
        return json.load(f)


def filter_annotations(annotations: Sequence[dict], pair_ids: Optional[Sequence[str]], limit: Optional[int]) -> List[dict]:
    filtered = list(annotations)
    if pair_ids:
        wanted = set(pair_ids)
        filtered = [rec for rec in filtered if rec["pair_id"] in wanted]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def frac_tag(frac: float) -> str:
    s = f"{frac:.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def frac_dir_name(frac: float) -> str:
    return f"frac_{frac_tag(frac)}"


def prepare_output_dirs(
    dataset_root: Path,
    out_parent: Path,
    fracs: Sequence[float],
    overwrite: bool,
    output_layout: str,
    sibling_parent: Optional[Path],
    sibling_prefix: Optional[str],
) -> Dict[float, Path]:
    out_roots: Dict[float, Path] = {}
    if output_layout == "nested":
        if out_parent.exists() and overwrite:
            shutil.rmtree(out_parent)
        out_parent.mkdir(parents=True, exist_ok=True)
        for frac in fracs:
            out_root = out_parent / frac_dir_name(frac)
            if out_root.exists() and any(out_root.iterdir()) and not overwrite:
                raise FileExistsError(f"Output directory already exists and is non-empty: {out_root}")
            out_root.mkdir(parents=True, exist_ok=True)
            out_roots[frac] = out_root
        return out_roots

    parent = sibling_parent or dataset_root.parent
    prefix = sibling_prefix or dataset_root.name
    parent.mkdir(parents=True, exist_ok=True)
    for frac in fracs:
        out_root = parent / f"{prefix}_{frac_dir_name(frac)}"
        if out_root.exists() and overwrite:
            shutil.rmtree(out_root)
        if out_root.exists() and any(out_root.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {out_root}")
        out_root.mkdir(parents=True, exist_ok=True)
        out_roots[frac] = out_root
    return out_roots


def regenerate_pair_with_expand_frac(pair_folder: Path, meta: dict, expand_frac: float, hypersim_root: Path, device: str, inpaint_engine: str) -> Image.Image:
    b_original = Image.open(pair_folder / "B_original.jpg").convert("RGB")
    width, height = b_original.size

    scene_id = meta["scene_id"]
    cam_c = meta["cameras"]["C"]
    frame_c = meta["frames"]["C_padded"]
    object_id = int(meta["object_id"])

    rgb_c = Image.open(resolve_rgb_path(hypersim_root, scene_id, frame_c, cam_c)).convert("RGB")
    inst_c = read_instance_h5(resolve_inst_path(hypersim_root, scene_id, frame_c, cam_c))
    inst_c = resize_instance_if_needed(inst_c, rgb_c.size)
    donor_rgb_crop, donor_alpha = crop_instance(rgb_c, inst_c, object_id)

    new_w, new_h = [int(v) for v in meta["paste"]["size"]]
    paste_x, paste_y = [int(v) for v in meta["paste"]["paste_xy"]]
    donor_resized = donor_rgb_crop.resize((new_w, new_h), Image.BICUBIC)
    alpha_resized = donor_alpha.resize((new_w, new_h), Image.NEAREST)

    x, y, w, h = [int(v) for v in meta["bboxes"]["object_bbox"]]
    initial_bbox = expand_bbox((x, y, w, h), expand_frac, width, height)
    paste_rect = (paste_x, paste_y, new_w, new_h)
    inpaint_bbox = rect_union(initial_bbox, paste_rect, width, height)
    ix, iy, iw, ih = inpaint_bbox

    inpaint_rect_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(inpaint_rect_mask)
    draw.rectangle([ix, iy, ix + iw, iy + ih], fill=255)

    inpainted = inpaint_image(b_original, inpaint_rect_mask, device, inpaint_engine)
    result = inpainted.copy()
    result.paste(donor_resized.convert("RGBA"), (paste_x, paste_y), mask=alpha_resized)

    if bool(meta.get("occluder_restore", {}).get("enabled", False)):
        cam_b = meta["cameras"]["B"]
        frame_b = meta["frames"]["B_padded"]
        inst_b = read_instance_h5(resolve_inst_path(hypersim_root, scene_id, frame_b, cam_b))
        inst_b = resize_instance_if_needed(inst_b, b_original.size)
        occluder_mask = np.zeros((height, width), dtype=np.uint8)
        bbox_slice = (slice(y, y + h), slice(x, x + w))
        inst_crop = inst_b[bbox_slice]
        occ_crop = ((inst_crop != object_id) & (inst_crop != -1)).astype(np.uint8) * 255
        occluder_mask[bbox_slice] = occ_crop
        result.paste(b_original, (0, 0), mask=Image.fromarray(occluder_mask, mode="L"))

    return result


def process_one_pair(
    rec: dict,
    dataset_root: Path,
    hypersim_root: Path,
    out_roots: Dict[float, Path],
    expand_fracs: Sequence[float],
    device: str,
    inpaint_engine: str,
) -> Dict[float, dict]:
    pair_id = rec["pair_id"]
    pair_in_dir = dataset_root / pair_id
    a_in_path = pair_in_dir / "A.jpg"
    if not a_in_path.is_file():
        raise FileNotFoundError(f"Missing A.jpg for pair: {pair_id}")

    pair_folder = Path(rec["sources"]["pair_folder"])
    meta_path = pair_folder / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing source metadata for pair: {pair_id}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    out_records: Dict[float, dict] = {}
    image_cache: Dict[float, Image.Image] = {}

    for expand_frac in expand_fracs:
        if expand_frac not in image_cache:
            image_cache[expand_frac] = regenerate_pair_with_expand_frac(
                pair_folder=pair_folder,
                meta=meta,
                expand_frac=expand_frac,
                hypersim_root=hypersim_root,
                device=device,
                inpaint_engine=inpaint_engine,
            )

        out_root = out_roots[expand_frac]
        pair_out_dir = out_root / pair_id
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a_in_path, pair_out_dir / "A.jpg")
        image_cache[expand_frac].save(pair_out_dir / "B.jpg", quality=95)

        out_rec = deepcopy(rec)
        out_rec["expand_frac_ablation"] = {
            "expand_frac": float(expand_frac),
            "inpaint_engine": inpaint_engine,
            "mode": "main_inconsistency_only",
        }
        out_records[expand_frac] = out_rec

    return out_records


def process_pair_batch(
    records: Sequence[dict],
    dataset_root: Path,
    hypersim_root: Path,
    out_roots: Dict[float, Path],
    expand_fracs: Sequence[float],
    device: str,
    inpaint_engine: str,
) -> Dict[float, List[dict]]:
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    batch_out: Dict[float, List[dict]] = {frac: [] for frac in expand_fracs}
    for rec in records:
        pair_out = process_one_pair(
            rec=rec,
            dataset_root=dataset_root,
            hypersim_root=hypersim_root,
            out_roots=out_roots,
            expand_fracs=expand_fracs,
            device=device,
            inpaint_engine=inpaint_engine,
        )
        for frac in expand_fracs:
            batch_out[frac].append(pair_out[frac])
    return batch_out


def main() -> None:
    args = parse_args()
    expand_fracs = parse_expand_fracs(args.expand_fracs)
    execution_mode, devices, worker_count = resolve_execution(args.device, int(args.max_workers), args.inpaint_engine)

    annotations = load_annotations(args.dataset_root)
    annotations = filter_annotations(annotations, args.pair_id, args.limit)
    if not annotations:
        raise ValueError("No pairs matched the requested filters.")

    out_roots = prepare_output_dirs(
        dataset_root=args.dataset_root,
        out_parent=args.out_parent,
        fracs=expand_fracs,
        overwrite=args.overwrite,
        output_layout=args.output_layout,
        sibling_parent=args.sibling_parent,
        sibling_prefix=args.sibling_prefix,
    )
    out_annotations: Dict[float, List[dict]] = {frac: [] for frac in expand_fracs}

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
                    expand_fracs,
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
                for frac in expand_fracs:
                    out_annotations[frac].extend(batch_out[frac])
    elif worker_count == 1:
        device = devices[0]
        print(f"Using single-worker mode on {device}")
        for rec in tqdm(annotations, desc="Pairs"):
            pair_out = process_one_pair(
                rec=rec,
                dataset_root=args.dataset_root,
                hypersim_root=args.hypersim_root,
                out_roots=out_roots,
                expand_fracs=expand_fracs,
                device=device,
                inpaint_engine=args.inpaint_engine,
            )
            for frac in expand_fracs:
                out_annotations[frac].append(pair_out[frac])
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
                    expand_fracs,
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
                for frac in expand_fracs:
                    out_annotations[frac].append(pair_out[frac])

    for frac, out_root in out_roots.items():
        with open(out_root / "annotations.json", "w") as f:
            ordered = sorted(out_annotations[frac], key=lambda rec: rec["pair_id"])
            json.dump(ordered, f, indent=2)

    if args.output_layout == "nested":
        print(f"Wrote expand-frac ablations to: {args.out_parent}")
    else:
        print("Wrote expand-frac ablations to sibling dataset roots:")
    for frac in expand_fracs:
        print(f"  {frac_dir_name(frac)} -> {out_roots[frac]}")


if __name__ == "__main__":
    main()
