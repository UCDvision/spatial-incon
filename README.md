# Multimodal Language Models Cannot Spot Spatial Inconsistencies

[Paper](https://arxiv.org/abs/2604.00799) · Accepted to COLM 2026 · [Curated benchmark](https://huggingface.co/datasets/reachomk/spatial-inconsistencies-hypersim-curated-615) · [Results](https://huggingface.co/datasets/reachomk/spatial-inconsistencies-results)

This repository contains the code and metadata for evaluating multimodal language models on spatial inconsistency detection. Given two views of the same static scene, the task is to identify the labeled object whose position is inconsistent with camera motion.

The benchmark contains 615 manually inspected Hypersim image pairs. Large artifacts are hosted on Hugging Face; this GitHub repository contains the code, annotations, metadata, and reproducibility utilities.

## Installation

```bash
git clone https://github.com/UCDvision/spatial-incon.git
cd spatial-incon
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
```

## Dataset

Download the curated 615-pair benchmark and scene-disjoint splits from Hugging Face:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='reachomk/spatial-inconsistencies-hypersim-curated-615', repo_type='dataset', local_dir='data/benchmark', ignore_patterns='splits/*')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='reachomk/spatial-inconsistencies-hypersim-curated-615', repo_type='dataset', local_dir='data', allow_patterns='splits/*')"
```

These commands create `data/benchmark/` with the pair directories and annotations, and `data/splits/` with the released train/test splits. The benchmark pairs have `A.jpg` and `B.jpg`; `public_eval_annotations.json` provides labels and evaluation groupings.

The automatic Hypersim and DL3DV datasets are maintained as separate private Hugging Face repositories.

## Evaluation

First validate the downloaded release artifacts:

```bash
python scripts/verify_release.py
```

Run a small Qwen3-VL smoke test:

```bash
python qwen_eval_simple.py \
  --annotations public_eval_annotations.json \
  --pairs-root data/benchmark \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --limit 5 \
  --output-dir outputs/smoke_test
```

For a full evaluation, omit `--limit`. The evaluator writes per-example predictions, an aggregate summary, and grouped accuracy reports. Use `--dry-run` to inspect the benchmark without loading a model.

## Released results

Recorded paper predictions and summaries are available in the private [results dataset](https://huggingface.co/datasets/reachomk/spatial-inconsistencies-results). To aggregate a prediction file after downloading it:

```bash
python scripts/aggregate_results.py results/main/results_gpt5-low.json --output gpt5_low.csv
```

The result filenames encode the model, reasoning setting, and control variant. See [RESULTS.md](RESULTS.md) for the artifact layout.

## Repository structure

```text
qwen_eval_simple.py       Qwen3-VL evaluation entry point
generation/               benchmark annotation and control-generation utilities
evaluation/               evaluation scripts for additional VLMs
scripts/                  release validation and result aggregation
metadata/                 released grouping and provenance metadata
public_eval_annotations.json  labels and evaluation groupings
requirements-*.txt        evaluation and generation environments
```

## Reproducibility

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes the validation procedure, deterministic evaluation settings, output formats, and requirements for regenerating controls.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{khangaonkar2026spatial,
  title     = {Multimodal Language Models Cannot Spot Spatial Inconsistencies},
  author    = {Khangaonkar, Om and Rad, Hadi J. and Pirsiavash, Hamed},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
