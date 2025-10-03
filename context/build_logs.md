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
