# Joint fine-grained detection of personal data and harmful content (Italian and English)

Code for the paper *Joint Fine-Grained Detection of Personal Data and Harmful Content in Italian and English*
(M. Mongiovì, L. Bulla; submitted to IEEE Access).

A single XLM-RoBERTa large encoder is fine-tuned with two heads:
- **PII tagging**: token-level BIO tagging over 20 types of personal identifiers
  ([ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy));
- **harm classification**: text-level `unsafe` flag plus 23 Aegis 2.0 harm categories, multi-label
  ([nvidia/Nemotron-Safety-Guard-Dataset-v3](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3)).

Only the English and Italian portions of both datasets are used. The dataset revisions are pinned in `configs/`.

## Repository layout
| Path | Content |
|---|---|
| `train.py` | Training (joint or single-task via `tasks:`), checkpointing/resume, final test evaluation |
| `sensitive/data.py` | Data loading, label construction, tokenization, length-bucketed batching |
| `sensitive/model.py` | Shared encoder with the PII head and the harm head |
| `eval/evaluate.py` | Official metrics (entity-level F1, per-category F1, threshold calibration); CLI to evaluate a checkpoint |
| `analysis/` | Inference-only analyses: subsets and confusions, bootstrap intervals, joint vs single-task comparison |
| `configs/` | Configurations of all runs in the paper (joint seeds 42/43/44, PII-only, harm-only) |
| `scripts/` | Sequential launcher used for the baseline runs (one tmux session per run) |
| `paper/` | Figure generation (`make_figures.py`) and the architecture figure (TikZ) |

Code comments are in Italian.

## Setup
Tested with Python 3.11, PyTorch 2.9.1 (CUDA 12.8) and Transformers 5.20.1 on one NVIDIA RTX A6000 (48 GB);
peak memory is about 25 GB. See `requirements.txt` for the full list.
```bash
pip install -r requirements.txt
```
Datasets and the encoder are downloaded from the Hugging Face Hub on first use.

## Reproducing the paper
```bash
# joint model (Tables 3-5, Figures 2-4); about 40 min on an RTX A6000
python train.py --config configs/sensitive_xlmr_large.yaml --run_dir runs/joint-seed42
# additional seeds and single-task baselines (Table 6)
python train.py --config configs/sensitive_xlmr_large_seed43.yaml --run_dir runs/joint-seed43
python train.py --config configs/sensitive_xlmr_large_seed44.yaml --run_dir runs/joint-seed44
python train.py --config configs/sensitive_xlmr_large_pii_only.yaml --run_dir runs/pii-only
python train.py --config configs/sensitive_xlmr_large_safety_only.yaml --run_dir runs/safety-only

# analyses (inference only)
python analysis/analyze_best.py --run_dir runs/joint-seed42
python analysis/bootstrap_ci.py --run_dir runs/joint-seed42
python analysis/compare_runs.py --joint runs/joint-seed42 runs/joint-seed43 runs/joint-seed44 \
    --pii_only runs/pii-only --safety_only runs/safety-only --out runs/comparison_test.json
python paper/make_figures.py --run_dir runs/joint-seed42 --out paper/figures
```
Each run directory contains `train.log`, `metrics.jsonl`, TensorBoard logs (`tb/`), `test_results.json`,
`checkpoints/last/` (full state for resuming) and `checkpoints/best/` (weights, tokenizer, labels, validation thresholds).
Training resumes automatically from `checkpoints/last/` if it exists.

To evaluate a saved checkpoint:
```bash
python eval/evaluate.py --ckpt runs/joint-seed42/checkpoints/best --split test
```

## Data licenses
Both datasets are released under CC-BY-4.0. The PII corpus was generated with Llama 3.x models and is subject to the
Llama community license terms ("Built with Llama"). The harm dataset contains offensive and unsafe content.
