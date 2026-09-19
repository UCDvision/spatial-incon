# Reproducibility guide

Run structural validation first:

```bash
python scripts/verify_release.py
```

This verifies that all 615 annotations resolve to image pairs, every answer is valid, all public grouping buckets are populated, and released splits are scene-disjoint.

For inference, use the smoke-test command in [README.md](README.md). A full Qwen run requires a CUDA-capable machine and model-download access. The script writes per-example JSONL/JSON results, an overall summary, and grouped CSV/JSON breakdowns.

The primary evaluation is deterministic with `--temperature 0`. Record the model revision, Transformers version, device, and output directory with any new result. API models require valid provider credentials and can change over time; bundled predictions are paper-run artifacts.

`generation/build_selfpaste_ablations.py` creates the additional-object self-paste controls. `generation/build_expandfrac_ablations.py` creates inpainting-box expansion controls. Both require original Hypersim HDF5 assets, a LaMa installation/checkpoint, and substantial compute.
