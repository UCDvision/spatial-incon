#!/usr/bin/env python3
"""
Video-style VQAScore evaluation on (A,B_orig) vs (A,B_mod) using t2v_metrics,
with running accuracy printed after each completed pair.

Input JSON (from your captioning step) entries:
{
  "id": "pair_id",
  "caption": "video-style caption...",
  "original_A": "/path/to/A_original.jpg",
  "original_B": "/path/to/B_original.jpg",
  "modified_B": "/path/to/B_final_inpainted.jpg"
}
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio
import t2v_metrics

# ---------------- Stats helpers ----------------

def pearson_corr(x: List[float], y: List[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        return float("nan")
    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    num = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    den = np.sqrt(np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2))
    if den == 0:
        return float("nan")
    return float(num / den)

def kendall_tau_b(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])

            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue

            if dx == dy:
                concordant += 1
            else:
                discordant += 1

    denom = np.sqrt(
        (concordant + discordant + ties_x) *
        (concordant + discordant + ties_y)
    )
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom

# ---------------- Video helpers ----------------
def build_video_dataset(
    records: List[Dict[str, Any]],
    videos_dir: str,
    fps: float,
    skip_missing: bool = True,
    overwrite_videos: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Returns:
      dataset: list of {"images": [video_path], "texts": [caption]}
      metas:   list of (pair_id, kind) aligned with dataset, kind in {"orig","mod"}
    """
    dataset: List[Dict[str, Any]] = []
    metas: List[Tuple[str, str]] = []

    os.makedirs(videos_dir, exist_ok=True)

    for rec in records:
        pair_id = rec.get("id") or rec.get("pair_id")
        caption = rec.get("caption")
        A = rec.get("original_A")
        B_orig = rec.get("original_B")
        B_mod = rec.get("modified_B")

        if not pair_id or not caption or not A or not B_orig or not B_mod:
            if skip_missing:
                continue
            raise ValueError(f"Missing fields for record: {rec}")

        missing_paths = [p for p in [A, B_orig, B_mod] if not os.path.isfile(p)]
        if missing_paths:
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"Missing image files for {pair_id}: {missing_paths}"
            )

        orig_video = os.path.join(videos_dir, f"{pair_id}__orig.mp4")
        mod_video  = os.path.join(videos_dir, f"{pair_id}__mod.mp4")

        try:
            make_two_frame_video([A, B_orig], orig_video, fps=fps, overwrite=overwrite_videos)
            make_two_frame_video([A, B_mod],  mod_video,  fps=fps, overwrite=overwrite_videos)
        except Exception as e:
            if skip_missing:
                continue
            raise RuntimeError(f"Failed to build videos for {pair_id}: {e}")

        # Important: orig then mod, so each pair's entries are consecutive
        dataset.append({"images": [orig_video], "texts": [caption]})
        metas.append((pair_id, "orig"))

        dataset.append({"images": [mod_video], "texts": [caption]})
        metas.append((pair_id, "mod"))

    return dataset, metas

def make_two_frame_video(
    frame_paths: List[str],
    out_path: str,
    fps: float = 8.0,
    overwrite: bool = False,
) -> str:
    if len(frame_paths) != 2:
        raise ValueError("Expected exactly 2 frame paths.")

    if (not overwrite) and os.path.isfile(out_path):
        return out_path

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    imgs = []
    base_w = base_h = None
    for p in frame_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing frame: {p}")
        img = Image.open(p).convert("RGB")
        if base_w is None:
            base_w, base_h = img.size
        elif img.size != (base_w, base_h):
            img = img.resize((base_w, base_h), Image.BICUBIC)
        imgs.append(np.array(img))

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
        for im in imgs:
            writer.append_data(im)

    return out_path

def make_video_from_pair(
    frame_paths: List[str],
    out_path: str,
    fps: float = 8.0,
    duration_sec: float = 2.0, # Make a 2-second video
    overwrite: bool = False,
) -> str:
    if len(frame_paths) != 2:
        raise ValueError("Expected exactly 2 frame paths.")
    
    if (not overwrite) and os.path.isfile(out_path):
        return out_path
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    img_a = Image.open(frame_paths[0]).convert("RGB")
    img_b = Image.open(frame_paths[1]).convert("RGB").resize(img_a.size, Image.BICUBIC)
    
    arr_a = np.array(img_a)
    arr_b = np.array(img_b)

    total_frames = int(fps * duration_sec)
    half_frames = total_frames // 2

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
        # Write first frame for the first half
        for _ in range(half_frames):
            writer.append_data(arr_a)
        # Write second frame for the second half
        for _ in range(total_frames - half_frames):
            writer.append_data(arr_b)
            
    return out_path
# ---------------- Progressive JSON helper ----------------

def write_progress_json(  # NEW
    path: str,
    orig_scores: Dict[str, float],
    mod_scores: Dict[str, float],
) -> None:
    """
    Write all currently completed pairs (with finite scores) to JSON.

    Format:
    [
      {
        "id": "pair_id",
        "score_orig": float,
        "score_mod": float,
        "delta": float
      },
      ...
    ]
    """
    if path is None:
        return

    completed_ids = sorted(set(orig_scores.keys()) & set(mod_scores.keys()))
    data = []
    for pid in completed_ids:
        so = orig_scores[pid]
        sm = mod_scores[pid]
        if not (np.isfinite(so) and np.isfinite(sm)):
            continue
        data.append(
            {
                "id": pid,
                "score_orig": float(so),
                "score_mod": float(sm),
                "delta": float(so - sm),
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- VQAScore + running metrics ----------------

def run_vqascore_with_progress(
    model_name: str,
    dataset: List[Dict[str, Any]],
    metas: List[Tuple[str, str]],
    batch_size: int,
    device: str,
    fps: float = None,
    num_frames: int = None,
    question_template: str = None,
    answer_template: str = None,
    progress_json: str = None,  # NEW
):
    """
    Stream through dataset in batches, compute VQAScore, and:
      - store per-pair orig/mod scores
      - after each pair is completed:
          * print running pairwise accuracy
          * (optionally) update progress_json with all completed pairs

    Returns:
      orig_scores, mod_scores dicts.
    """
    if len(dataset) != len(metas):
        raise ValueError("dataset and metas length mismatch.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running VQAScore evaluation.")
    score_fn = t2v_metrics.VQAScore(model=model_name, api_key=api_key)

    kw = {}
    if fps is not None:
        kw["fps"] = fps
    if num_frames is not None:
        kw["num_frames"] = num_frames
    if question_template is not None:
        kw["question_template"] = question_template
    if answer_template is not None:
        kw["answer_template"] = answer_template

    orig_scores: Dict[str, float] = {}
    mod_scores: Dict[str, float] = {}

    # Running stats
    total_pairs = 0
    correct_pairs = 0

    N = len(dataset)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = dataset[start:end]
        metas_batch = metas[start:end]

        scores_batch = score_fn.batch_forward(dataset=batch, batch_size=len(batch), **kw)
        scores_batch = np.array(scores_batch.cpu())

        for j, (pair_id, kind) in enumerate(metas_batch):
            s = float(scores_batch[j].mean())
            if kind == "orig":
                orig_scores[pair_id] = s
            elif kind == "mod":
                mod_scores[pair_id] = s
            else:
                raise ValueError(f"Unknown kind: {kind}")

            # When both scores for this pair exist, update running accuracy
            if pair_id in orig_scores and pair_id in mod_scores:
                so = orig_scores[pair_id]
                sm = mod_scores[pair_id]
                if np.isfinite(so) and np.isfinite(sm):
                    total_pairs += 1
                    if so > sm:
                        correct_pairs += 1
                    running_acc = correct_pairs / total_pairs
                    print(
                        f"[progress] pairs_done={total_pairs} "
                        f"running_pairwise_acc={running_acc:.4f}",
                        flush=True,
                    )

                    # NEW: update progressive JSON whenever a pair completes
                    if progress_json is not None:
                        write_progress_json(progress_json, orig_scores, mod_scores)

    return orig_scores, mod_scores

# ---------------- Final metrics ----------------

def compute_metrics(
    orig_scores: Dict[str, float],
    mod_scores: Dict[str, float],
) -> Dict[str, float]:
    pair_ids = sorted(set(orig_scores.keys()) & set(mod_scores.keys()))
    if not pair_ids:
        raise ValueError("No overlapping pairs between orig_scores and mod_scores.")

    wins = 0
    total = 0
    cand_scores: List[float] = []
    cand_labels: List[int] = []

    for pid in pair_ids:
        s_orig = orig_scores[pid]
        s_mod = mod_scores[pid]
        if not (np.isfinite(s_orig) and np.isfinite(s_mod)):
            continue

        total += 1
        if s_orig > s_mod:
            wins += 1

        cand_scores.append(s_orig); cand_labels.append(1)
        cand_scores.append(s_mod);  cand_labels.append(0)

    pairwise_acc = wins / total if total > 0 else float("nan")
    pearson = pearson_corr(cand_scores, cand_labels)
    kendall = kendall_tau_b(cand_scores, cand_labels)

    return {
        "num_pairs": total,
        "pairwise_accuracy": pairwise_acc,
        "pearson": pearson,
        "kendall_tau_b": kendall,
    }

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Video VQAScore on stitched (A,B_orig) vs (A,B_mod), with running accuracy."
    )
    ap.add_argument("--captions-json", default="/nfs_share4/code/om/hypersim/vqascore_pairs.json") 
    ap.add_argument("--videos-dir", default="/nfs_share4/code/om/hypersim/video_eval")
    ap.add_argument("--model", default="gpt-4o", help="Video-capable VQAScore model name.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--num-frames", type=int, default=2)
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--overwrite-videos", action="store_true")
    ap.add_argument("--question-template", default=None)
    ap.add_argument("--answer-template", default=None)
    ap.add_argument(
        "--save-per-pair",
        default=None,
        help="Optional JSON to save final per-pair scores.",
    )
    ap.add_argument(  # NEW
        "--progress-json",
        default="/nfs_share4/code/om/hypersim/vqascore_pairs_progress.json",
        help="Optional JSON file to update progressively as pairs complete.",
    )
    args = ap.parse_args()

    # Load records
    with open(args.captions_json, "r") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {args.captions_json}")

    # Build dataset
    dataset, metas = build_video_dataset(
        records=records,
        videos_dir=args.videos_dir,
        fps=args.fps,
        skip_missing=args.skip_missing,
        overwrite_videos=args.overwrite_videos,
    )

    if not dataset:
        raise ValueError("No valid samples after video construction.")

    print(f"Constructed {len(dataset)} video candidates ({len(dataset)//2} pairs).")

    # Run VQAScore with online progress
    print(f"Running VQAScore model='{args.model}' on device={args.device} ...")
    orig_scores, mod_scores = run_vqascore_with_progress(
        model_name=args.model,
        dataset=dataset,
        metas=metas,
        batch_size=args.batch_size,
        device=args.device,
        fps=args.fps,
        num_frames=args.num_frames,
        question_template=args.question_template,
        answer_template=args.answer_template,
        progress_json=args.progress_json,  # NEW
    )

    # Optional final save
    if args.save_per_pair is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_per_pair)) or ".", exist_ok=True)
        out = []
        for pid in sorted(set(orig_scores.keys()) & set(mod_scores.keys())):
            out.append(
                {
                    "id": pid,
                    "score_orig": orig_scores[pid],
                    "score_mod": mod_scores[pid],
                    "delta": orig_scores[pid] - mod_scores[pid],
                }
            )
        with open(args.save_per_pair, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved per-pair scores to {args.save_per_pair}")

    # Final metrics
    metrics = compute_metrics(orig_scores, mod_scores)

    print("\n=== Video VQAScore Pairwise Evaluation ===")
    print(f"#Pairs used:           {metrics['num_pairs']}")
    print(f"Pairwise Accuracy:     {metrics['pairwise_accuracy']:.4f}")
    print(f"Pearson r (score,y):   {metrics['pearson']:.4f}")
    print(f"Kendall tau-b:         {metrics['kendall_tau_b']:.4f}")

if __name__ == "__main__":
    main()
