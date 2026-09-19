# Results artifacts

`results/main/` holds primary paper-run result files and summaries copied from `results_eval_full/`. `results/ablations_qwen/` holds Qwen control and ablation outputs. Top-level `results/*.json` and `results/*.jsonl` contain recorded GPT-5, Gemini, control, and DL3DV outputs available in this workspace.

Use original filenames as provenance: they encode model, reasoning level, and control variant. Do not compare files with different example counts without checking their companion summary. Some checkpoint directories contain partial runs; they are retained as experimental artifacts and are not designated as official paper checkpoints.

The Qwen evaluator can recompute overall and grouped metrics from a new run. For released prediction files, use `scripts/aggregate_results.py` to generate a normalized accuracy CSV, for example:

```bash
python scripts/aggregate_results.py results/main/results_gpt5-low.json --output gpt5_low.csv
```

The result files are organized by model and experiment variant.
