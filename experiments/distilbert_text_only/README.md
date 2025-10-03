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

**Duration**: ~15-30 minutes per epoch on M1/M2 Mac (depends on CPU/GPU)

**Outputs**:
- `artifacts/checkpoint-*`: Model checkpoints per epoch
- `artifacts/final_model/`: Best model from training
- `artifacts/final_metrics.txt`: Final accuracy and F1-macro scores
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
- **Batch Size**: 16 (train), 32 (eval)
- **Learning Rate**: 2e-5 with AdamW
- **Epochs**: 5 (with early stopping patience=2)
- **Metrics**: Accuracy (primary), Macro-F1 (secondary)
- **Split**: 80/20 stratified (critical for 2.17x class imbalance)

---

## 🎯 Implementation Details

### Data Pipeline

1. **Load CSV**: `load_recipes_csv()` from `src/data.py`
   - Parses list-like columns (tags, steps, ingredients)
   - Normalizes column names

2. **Prepare Dataset**: `prepare_dataset()` from `src/dataset.py`
   - Concatenates text fields: `recipe_name → ingredients → tags → description → steps`
   - Encodes chef labels (6 classes)
   - Stratified 80/20 split (2.17x imbalance)

3. **Tokenize**: `tokenize_dataset()` from `src/tokenization.py`
   - DistilBERT tokenizer with `max_length=512`
   - Dynamic padding per batch (`padding='longest'`)
   - Truncation from the end (protects recipe name/ingredients)

### Model Architecture

- **Base**: DistilBERT-base-uncased (66M params, 6 layers, 768 hidden)
- **Head**: Single linear layer (768 → 6 classes)
- **Activation**: GELU (built into DistilBERT)
- **Loss**: CrossEntropyLoss (built into `AutoModelForSequenceClassification`)

### Training Strategy

- **Optimizer**: AdamW with weight_decay=0.01
- **Scheduler**: Linear warmup (6% of steps) + linear decay
- **Early Stopping**: Patience=2 epochs on `f1_macro`
- **Checkpointing**: Save best model based on validation F1-macro
- **Device**: Automatic MPS (Mac GPU) or CPU detection

---

## 📈 Final Results (5 Epochs - Chill Mode)

- **Train loss:** 0.4042
- **Validation accuracy:** 0.9017 (90.17%)
- **Validation F1-macro:** 0.8967

**Performance vs Baselines:**
- Weak baseline (TF-IDF on description): 30.0% → **+60.17%**
- Strong baseline (TF-IDF on all fields): 43.0% → **+47.17%**

Artifacts saved in `artifacts/final_model/` for reproducibility.

---

## 📝 Notes

- **Class Imbalance**: 2.17x ratio (chef 4470: 806 samples, chef 6357: 372 samples)
  → Use stratified split and track macro-F1 to catch minority class issues

- **Token Lengths**: Median=272, p95=431, max=627
  → 98.2% fit in 512 tokens, only 1.8% get truncated

- **Field Order**: `steps=42%` of tokens → truncate last, protect `recipe_name` and `ingredients`

- **Memory**: Batch size 16 safe for M1/M2 Macs (median 272 tokens * 16 = 4,352 tokens/batch)

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
- [ ] Prediction export for test set (future work)
