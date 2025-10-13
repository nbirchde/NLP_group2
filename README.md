# Chef Classification (Group 2)

DistilBERT-based classifier that assigns each recipe in the provided dataset to one of six chefs.  
All code runs on Python 3.10+ and was tested on macOS with Apple Silicon (MPS) acceleration.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets accelerate scikit-learn pandas numpy matplotlib seaborn
```

The training and test CSV files (`data/train.csv`, `data/test-no-labels.csv`) must be in place before running the scripts.

## 2. Train the model

```bash
python experiments/distilbert_text_only/train.py --config configs/chill_mode.yaml
```

Outputs:
- Fine-tuned model: `experiments/distilbert_text_only/artifacts/final_model/`
- Metrics summary: `experiments/distilbert_text_only/artifacts/final_metrics.txt`
- Training log: `experiments/distilbert_text_only/chill_training.log`

## 3. Generate test predictions

```bash
python experiments/distilbert_text_only/predict.py \
  --model-path experiments/distilbert_text_only/artifacts/final_model \
  --test-path data/test-no-labels.csv \
  --output results.txt
```

The `results.txt` file contains one chef ID per test recipe and is ready for submission.

## 4. Repository map

```
configs/                 YAML configs used by train.py
experiments/distilbert_text_only/train.py   main training script
experiments/distilbert_text_only/predict.py inference script
src/                    data loading, tokenisation, and model helpers
results/figures/        generated plots for the report
Project-Template/       LaTeX sources for the two-page paper
```
