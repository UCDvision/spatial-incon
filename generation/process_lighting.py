#!/usr/bin/env python3
import json, os, math, h5py, numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------- CONFIG ----------
PAIRS_JSON_IN  = "/nfs_share4/code/om/hypersim/3d_eval_FULL/annotations.json"
PAIRS_JSON_OUT = "pairs_with_lighting.json"
MAX_WORKERS    = min(8, os.cpu_count() or 4)  # tune if your I/O is fast/slow
TRIM_LOW_PCT   = 1.0
TRIM_HIGH_PCT  = 99.0
# ---------------------------

def diffuse_path(scene_id: str, cam: str, frame: str) -> str:
    # /nfs_share4/code/om/hypersim/diffuse-illum/{scene id}/images/scene_cam_{cam id}_final_hdf5/frame.{frame id}.diffuse_illumination.hdf5
    return f"/nfs_share4/code/om/hypersim/diffuse-illum/{scene_id}/images/scene_cam_{cam}_final_hdf5/frame.{frame}.diffuse_illumination.hdf5"

def _first_dataset(f: h5py.File):
    # Try common keys first
    for k in ["dataset", "data", "diffuse_illumination", "image", "img"]:
        if k in f and isinstance(f[k], h5py.Dataset):
            return f[k][()]
    # Fallback: DFS for first dataset
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

def read_luminance_h5(path: str) -> np.ndarray:
    """Load luminance array from HDF5 (float16 expected). Return float32."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        arr = _first_dataset(f)
        if arr is None:
            raise RuntimeError(f"No dataset in {path}")
    # Cast to float32 before stats to avoid precision issues
    return np.asarray(arr, dtype=np.float32)

def robust_mean(vals: np.ndarray, lo_p=TRIM_LOW_PCT, hi_p=TRIM_HIGH_PCT) -> float:
    flat = vals.ravel().astype(np.float32)
    if flat.size == 0:
        return float("nan")
    lo, hi = np.percentile(flat, [lo_p, hi_p])
    trimmed = flat[(flat >= lo) & (flat <= hi)]
    return float(trimmed.mean() if trimmed.size else flat.mean())

def view_score(scene_id: str, cam: str, frame: str) -> float:
    path = diffuse_path(scene_id, cam, frame)
    Y = read_luminance_h5(path)   # already luminance
    return robust_mean(Y)

def compute_pair_AB(entry: dict) -> tuple[float, float, float]:
    """
    Worker function: compute (A_luminance, B_luminance, pair_lightness) for one entry.
    Safe to run in a separate process (independent file reads).
    """
    scene = entry["scene_id"]
    A = entry["A"]; B = entry["B"]

    A_l = B_l = float("nan")
    try:
        A_l = view_score(scene, A["cam"], A["frame"])
    except FileNotFoundError:
        pass
    try:
        B_l = view_score(scene, B["cam"], B["frame"])
    except FileNotFoundError:
        pass

    vals = [v for v in (A_l, B_l) if not math.isnan(v)]
    pair_mean = float(np.mean(vals)) if vals else float("nan")
    return A_l, B_l, pair_mean

def tertiles(values):
    clean = np.array([v for v in values if not math.isnan(v)], dtype=np.float32)
    if clean.size == 0:
        return float("nan"), float("nan")
    t1, t2 = np.percentile(clean, [33.3333, 66.6667])
    return float(t1), float(t2)

def label(score, t1, t2):
    if math.isnan(score): return "unknown"
    if score < t1: return "low"
    if score < t2: return "medium"
    return "high"

def main():
    with open(PAIRS_JSON_IN, "r") as f:
        pairs = json.load(f)

    # Parallel map with progress bar
    results = [None] * len(pairs)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(compute_pair_AB, e): i for i, e in enumerate(pairs)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring pairs (A & B)"):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                # In case of unexpected read/parse errors, keep NaNs so we can continue.
                results[i] = (float("nan"), float("nan"), float("nan"))

    pair_means = [pm for (_, _, pm) in results]
    t1, t2 = tertiles(pair_means)

    # Build augmented JSON mirroring input + new fields
    augmented = []
    for e, (A_l, B_l, pm) in zip(pairs, results):
        new_e = deepcopy(e)
        new_e["A_luminance"]   = A_l
        new_e["B_luminance"]   = B_l
        new_e["pair_lightness"] = pm
        new_e["pair_label"]     = label(pm, t1, t2)
        augmented.append(new_e)

    with open(PAIRS_JSON_OUT, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"Wrote {PAIRS_JSON_OUT}")
    print(f"Tertile thresholds → low < {t1:.6g}, medium < {t2:.6g}, high ≥ {t2:.6g}")

if __name__ == "__main__":
    main()
