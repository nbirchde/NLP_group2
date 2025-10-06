# DistilBERT Text-Only Experiment

Primary fine-tuning run using `distilbert-base-uncased` on concatenated recipe text fields. Configuration sourced from `configs/base.yaml`.

Artifacts (checkpoints, logs, metrics) are saved under `experiments/distilbert_text_only/artifacts/`.

---

## 🚀 Quick Start

### Prerequisites

Ensure your virtual environment is activated and dependencies are installed:

```bash
# Activate virtual environment (from project root)
source .venv/bin/activate

# Install missing dependencies if needed
pip install datasets scikit-learn
```

### Dry Run (Smoke Test)

Test the pipeline without full training:

```bash
cd experiments/distilbert_text_only
python train.py --config ../../configs/base.yaml --dry-run
```

This will:
- Load config and data
- Tokenize a small batch
- Run a single forward pass
- Validate model output shapes
- Exit without training

**Expected output**: ✓ marks for each stage, final "Dry run completed successfully!"

### Full Training

Run complete fine-tuning:

```bash
cd experiments/distilbert_text_only
python train.py --config ../../configs/base.yaml
```

**Duration**: ~30 minutes total on M1/M2 Mac in chill-mode (10 epochs, batch size 8)

**Outputs**:
- `artifacts/checkpoint-*`: Model checkpoints per epoch
- `artifacts/final_model/`: Best model from training
- `artifacts/final_metrics.txt`: Final accuracy and macro-F1 with bootstrap 95% CIs
- `artifacts/logs/`: Training logs for tensorboard

### Generate Test Predictions

After training completes, generate predictions for the test set:

```bash
cd experiments/distilbert_text_only
python predict.py \
  --model-path artifacts/final_model \
  --test-path ../../data/test-no-labels.csv \
  --output ../../results.txt
```

**Outputs**:
- `results.txt`: One chef_id prediction per line (submission file)
- Prediction distribution printed to console

**Options**:
- `--batch-size`: Inference batch size (default: 32, increase for faster inference)
- `--config`: Path to config YAML (default: ../../configs/base.yaml)

---

## 📊 Configuration

All hyperparameters are defined in `configs/base.yaml`:

- **Model**: `distilbert-base-uncased` (66M parameters)
- **Max Length**: 512 tokens (98.2% of samples fit)
- **Padding**: `longest` (dynamic per batch, saves memory)
- **Batch Size**: 16 (train), 32 (eval) — chill-mode uses 8/16
- **Learning Rate**: 2e-5 with AdamW
- **Epochs**: 10 (early stopping patience 2 on macro-F1)
- **Evaluation**: every 100 steps with checkpointing on the same cadence
- **Metrics**: Accuracy (primary), Macro-F1 (secondary) + bootstrap confidence intervals
- **Split**: 80/20 stratified (critical for 2.17x class imbalance)

---

## 🎯 Implementation Details

### Data Pipeline

1. **Load CSV**: `load_recipes_csv()` from `src/data.py`
   - Parses list-like columns (tags, steps, ingredients)
   - Normalizes column names

2. **Prepare Dataset**: `prepare_dataset()` from `src/dataset.py`
   - Concatenates text fields: `recipe_name → ingredients → tags → description → steps`
   - Removes duplicate rows based on concatenated text before splitting (prevents leakage)
   - Encodes chef labels (6 classes)
   - Stratified 80/20 split (2.17x imbalance)

3. **Tokenize**: `tokenize_dataset()` from `src/tokenization.py`
   - DistilBERT tokenizer with `max_length=512`
   - Dynamic padding per batch (`padding='longest'`)
   - Truncation from the end (protects recipe name/ingredients)

### Model Architecture

- **Base**: DistilBERT-base-uncased (66M params, 6 layers, 768 hidden)
- **Head**: Single linear layer (768 → 6 classes)
- **Activation**: GELU substituted in the classification head (replaces default ReLU)
- **Loss**: CrossEntropyLoss (built into `AutoModelForSequenceClassification`)

### Training Strategy

- **Optimizer**: AdamW with weight_decay=0.01
- **Scheduler**: Linear warmup (6% of steps) + linear decay
- **Early Stopping**: Patience=2 evaluations on `f1_macro`
- **Checkpointing**: Save best model at evaluation cadence (every 100 steps)
- **Uncertainty**: Bootstrap resampling (1,000 draws) for accuracy and macro-F1 confidence intervals
- **Device**: Automatic MPS (Mac GPU) or CPU detection

---
## 📝 Notes

- **Class Imbalance**: Post-dedup distribution → 4470 (801), 5060 (534), 3288 (451), 8688 (432), 1533 (402), 6357 (365)
  → Stratified splitting + macro-F1 monitoring keeps minority chefs above ~80% recall

- **Token Lengths**: Median = 234 tokens, 95th percentile = 418, max = 512
  → 98% of samples fit without truncation at max\_length=512

- **Field Order**: Token share ≈ steps 46%, tags 26%, description 16%, ingredients 10%, name 3%
  → Prepending name/ingredients prevents them from being trimmed when truncation kicks in

- **Memory**: Chill-mode batch size 8 keeps M1/M2 GPUs comfortable; base config (16) remains available for fuller runs

---

## 🐛 Troubleshooting

**ImportError: No module named 'datasets'**
```bash
source .venv/bin/activate
pip install datasets scikit-learn
```

**MPS not available warning**
→ Normal on Intel Macs, will use CPU (slower but works)

**Out of memory**
→ Reduce `train_batch_size` in `configs/base.yaml` (try 8 or 4)

**Poor validation accuracy**
→ Check macro-F1 score (accuracy can be misleading with class imbalance)
→ Try more epochs or lower learning rate

---

## ✅ Implementation Status

- [x] Data preprocessing/module wiring
- [x] Training script with CLI interface
- [x] Dry-run smoke test mode
- [x] Evaluation with accuracy + macro-F1
- [x] Prediction export script (`predict.py`)
