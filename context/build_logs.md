# Build Log

## 2025-10-03 (Codex)
- Initialized build log to track implementation blockers and debugging notes.
- Noted missing `experiments/` directory referenced in plans; will create during skeleton setup.
- Confirmed key instructions from `AGENTS.md` and `research/nicholas.md` to align implementation with agreed pipeline.


## 2025-10-03 (Codex)
- Created core directories (`experiments/`, `src/`, `configs/`, `results/`) to mirror agreed architecture.
- Added `configs/base.yaml` with canonical hyperparameters (DistilBERT, max_len=512, batch=16, etc.).
- Stubbed initial modules (`src/config.py`, `src/data.py`, `src/dataset.py`) and experiment entry script for wiring in upcoming steps.

## 2025-10-03 (Codex)
- Encountered missing Python dependencies (`numpy`) when importing `pandas` via `python3`; need to set up project environment or document dependency installation steps.

## 2025-10-03 (Human teammate)
- Provisioned project virtual environment `.venv` with pandas, numpy, transformers, torch, matplotlib installed.
- Confirmed scripts should be run with `.venv/bin/python` (available via `python` shim inside environment).

## 2025-10-04 (Codex)
- Completed comprehensive dataset analysis (2,999 samples, 2.17x imbalance, median 272 tokens, 98.2% fit in 512).
- Validated pipeline decisions: max_length=512, padding='longest', batch_size=16, stratified split.
- Archived detailed analysis in `decisions/dataset_analysis/` directory.
- Updated `research/nicholas.md` with inline technical clarifications (stratification, padding strategies, activations).
- Updated `AGENTS.md` with minimal critical stats (4 key metrics only).

## 2025-10-04 (Codex)
- Installed additional dependencies: `datasets` and `scikit-learn` packages.
- Implemented complete training script `experiments/distilbert_text_only/train.py`:
  - CLI interface with `--config` and `--dry-run` flags
  - Full pipeline: config loading → data prep → tokenization → model training
  - Trainer API with early stopping, stratified split, compute_metrics (accuracy + macro-F1)
  - MPS (Mac GPU) support with automatic fallback to CPU
  - Dry-run smoke test mode for validation without training
- Updated `experiments/distilbert_text_only/README.md` with comprehensive documentation:
  - Quick start guide with prerequisites
  - Dry-run and full training instructions
  - Configuration details and implementation notes
  - Troubleshooting section
- Fixed implementation issues:
  - Hardcoded train data path (data/train.csv) since not in config
  - Removed extra columns from tokenized dataset before collation
- **Successfully validated with dry-run test**: Forward pass works, logits shape correct (4, 6), loss computed (1.7894)
- Training script ready for full fine-tuning run

## 2025-10-04 (Codex)
- Verified teammate's training pipeline by re-running dry-run via `.venv` interpreter; confirmed forward pass and logging behave as expected.
- Noticed `TrainingArguments` uses `eval_strategy` parameter (non-existent); must be `evaluation_strategy` to enable validation + early stopping.

## 2025-10-04 (Codex)
- Installed `accelerate>=0.26.0` dependency required by Trainer API.
- Confirmed `eval_strategy` parameter is correct for transformers 4.56.2 (newer API).
- First real training run (1 epoch, quick_test.yaml):
  - Final train loss: 1.3387
  - Final validation accuracy: 0.8100 (beats weak baseline 30%, approaching strong baseline 43%)
  - Final validation F1-macro: 0.7913 (critical for 2.17x imbalance)
- Model and metrics saved to artifacts/ for baseline reference.
- Created `.gitignore` to exclude large model files (~1GB) from git:
  - Ignores: checkpoints, model weights (.safetensors, .bin, .pt)
  - Keeps: metrics.txt, configs, documentation
  - Prevents GitHub 100MB file limit errors

## 2025-10-04
- **End-to-end verification completed**:
  - ✅ CLI pipeline (train.py:205-280) wiring validated: config → data → tokenization → Trainer
  - ✅ Confirmed `eval_strategy` is correct parameter name for transformers 4.56.2 (via inspect.signature)
  - ✅ Documentation accuracy: warmup=6%, stratified split, artifact handling all correct
  - ✅ Baseline artifacts consistent: 81% acc / 0.79 F1 reproduced in quick_test.yaml
  - ✅ .gitignore working correctly: heavyweight checkpoints excluded, metrics preserved
- Cleaned quick-test artifacts (checkpoint-150, final_model) to prepare for 5-epoch production run
- **Ready for full training**: artifacts/ cleared, baseline documented, all systems validated
