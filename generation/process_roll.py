#!/usr/bin/env python3
"""
Compute B↔C roll component for each pair and flag 'wacky_roll' if it exceeds a threshold.

Inputs:
  - Pairs JSON (same schema you already use)
  - HDF5 camera orientations per your paths:
      /nfs_share4/code/om/hypersim/detail/[scene id]/_detail/cam_[id, two digits]/camera_keyframe_orientations.hdf5

Outputs:
  - JSON with added fields:
      roll_component_deg, roll_deg_threshold, wacky_roll, roll_status
"""

import os
import json
import math
import argparse
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from tqdm import tqdm

# -----------------------
# Paths
# -----------------------

def ori_path(scene_id: str, cam: str) -> str:
    cam2 = cam if len(cam) == 2 else str(cam).zfill(2)
    return f"/nfs_share4/code/om/hypersim/detail/{scene_id}/_detail/cam_{cam2}/camera_keyframe_orientations.hdf5"

# -----------------------
# HDF5 helpers
# -----------------------

def _first_dataset(f: h5py.File):
    # Try common names first
    for k in ["orientations", "quaternions", "rotations", "R", "data", "dataset", "array"]:
        if k in f and isinstance(f[k], h5py.Dataset):
            return f[k][()]
    # Fallback: DFS first dataset
    def dfs(g):
        for _, obj in g.items():
            if isinstance(obj, h5py.Dataset):
                return obj[()]
            if isinstance(obj, h5py.Group):
                out = dfs(obj)
                if out is not None:
                    return out
        return None
    return dfs(f)

def read_orientations_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        arr = _first_dataset(f)
        if arr is None:
            raise RuntimeError(f"No dataset found in {path}")
    return np.asarray(arr)

# -----------------------
# Rotation utilities
# -----------------------

def quat_to_R(q: np.ndarray, layout: str = "wxyz") -> np.ndarray:
    """Quaternion -> 3x3 rotation. layout in {'wxyz','xyzw'}."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must have 4 elements")
    if layout == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
    ], dtype=np.float64)
    return R

def item_to_R(item: np.ndarray, quat_layout: str) -> np.ndarray:
    """Accepts 4 (quat), 9/3x3/16/4x4; returns 3x3."""
    a = np.asarray(item)
    if a.ndim == 1 and a.size == 4:
        return quat_to_R(a, layout=quat_layout)
    if a.ndim == 1 and a.size == 9:
        return a.reshape(3,3).astype(np.float64)
    if a.shape == (3,3):
        return a.astype(np.float64)
    if a.ndim == 1 and a.size == 16:
        return a.reshape(4,4)[:3,:3].astype(np.float64)
    if a.shape == (4,4):
        return a[:3,:3].astype(np.float64)
    raise ValueError(f"Unsupported orientation item shape: {a.shape}")

def frame_index(frame_str: str) -> int:
    # '000123' -> 123
    return int(frame_str)

def load_R_c2w(scene_id: str, cam: str, frame: str, quat_layout: str) -> np.ndarray:
    """Load camera->world rotation for a frame."""
    path = ori_path(scene_id, cam)
    arr = read_orientations_array(path)
    idx = frame_index(frame)
    try:
        item = arr[idx]
    except Exception as e:
        raise IndexError(f"Index {idx} out of range for {path} (shape {arr.shape})") from e
    return item_to_R(item, quat_layout=quat_layout)

def z_axis_roll_from_relative(R_rel_body: np.ndarray) -> float:
    """
    Return roll-about-+Z (radians) from body-frame relative rotation R_rel_body.
    This is the z-axis angle: atan2(R[1,0], R[0,0]).
    """
    return math.atan2(R_rel_body[1,0], R_rel_body[0,0])

# -----------------------
# Pair processing
# -----------------------

def compute_roll_for_pair(
    entry: dict,
    roll_deg_threshold: float,
    quat_layout: str,
) -> Tuple[Optional[float], float, Optional[bool], str]:
    """
    Returns: (roll_component_deg, roll_deg_threshold, wacky_roll, roll_status)
    """
    try:
        scene = entry["scene_id"]
        B = entry["B"]
        C = entry["C"]
        camB = str(B["cam"])
        camC = str(C["cam"])
        if camB != camC:
            # Different cameras; still valid, use their own orientations
            pass

        RB = load_R_c2w(scene, camB, B["frame"], quat_layout)
        RC = load_R_c2w(scene, camC, C["frame"], quat_layout)

        # Relative rotation in B's camera/body frame: R_rel_body rotates B->C expressed in B
        R_rel_body = RB.T @ RC

        roll_rad = z_axis_roll_from_relative(R_rel_body)  # around +Z (camera forward)
        roll_deg = abs(math.degrees(roll_rad))            # magnitude only
        wacky = roll_deg > roll_deg_threshold

        return roll_deg, roll_deg_threshold, wacky, "ok"

    except FileNotFoundError:
        return None, roll_deg_threshold, None, "missing_orientations"
    except IndexError:
        return None, roll_deg_threshold, None, "frame_index_oob"
    except ValueError as e:
        return None, roll_deg_threshold, None, f"bad_item:{e}"
    except Exception as e:
        return None, roll_deg_threshold, None, f"error:{type(e).__name__}"

# -----------------------
# Main
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Flag pairs with wacky roll between B and C.")
    ap.add_argument("--pairs-json-in",  default="/nfs_share4/code/om/hypersim/pairs_with_lighting_and_depth.json",
                    help="Input pairs JSON")
    ap.add_argument("--pairs-json-out", default="pairs_with_wacky_roll.json",
                    help="Output JSON with roll fields")
    ap.add_argument("--roll-deg-threshold", type=float, default=5.0,
                    help="Threshold in degrees for 'wacky_roll' (default: 15)")
    ap.add_argument("--quat-layout", choices=["wxyz","xyzw"], default="wxyz",
                    help="Quaternion layout if orientations are quats (default: wxyz)")
    ap.add_argument("--max-workers", type=int, default=32,
                    help="Process pool size")
    return ap.parse_args()

def main():
    args = parse_args()

    with open(args.pairs_json_in, "r") as f:
        pairs = json.load(f)

    results = [None] * len(pairs)

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(
                compute_roll_for_pair,
                e,
                args.roll_deg_threshold,
                args.quat_layout,
            ): i for i, e in enumerate(pairs)
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Computing B↔C roll"):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = (None, args.roll_deg_threshold, None, f"error:{type(e).__name__}")

    augmented = []
    for e, (roll_deg, thr, wacky, status) in zip(pairs, results):
        new_e = dict(e)
        new_e["roll_component_deg"] = roll_deg
        new_e["roll_deg_threshold"] = thr
        new_e["wacky_roll"] = wacky
        new_e["roll_status"] = status
        augmented.append(new_e)

    with open(args.pairs_json_out, "w") as f:
        json.dump(augmented, f, indent=2)

    ok = sum(1 for (_, _, w, s) in results if s == "ok" and w is not None)
    flagged = sum(1 for (_, _, w, s) in results if s == "ok" and w)
    print(f"Wrote {args.pairs_json_out}")
    print(f"Processed {ok} pairs successfully; flagged wacky_roll: {flagged}")

if __name__ == "__main__":
    main()
