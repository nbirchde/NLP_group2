# Chef Classification with DistilBERT
## NLP Group 2

---

## The Challenge

- **Task**: Predict which chef (out of 6) created a recipe
- **Data**: 2,999 training recipes, 823 test recipes
- **Baselines**: TF-IDF 30% (weak), 43% (strong)
- **Goal**: Beat 43% with deep learning

---

## Dataset

**Class Distribution**:
- Most common: Chef 4470 (806 recipes)
- Least common: Chef 6357 (372 recipes)
- Imbalance ratio: 2.17x

**Text Statistics**:
- Median token length: **272 tokens**
- 95th percentile: **431 tokens**
- 98.2% fit in **512 tokens**

**Field Contributions**:
- Steps: 42% of tokens (most important!)
- Tags: 31%
- Description: 13%
- Ingredients: 12%
- Name: 2%

**Decision**: Use all fields, truncate last (steps), max_length=512

---

## 🏗️ Model Architecture

**Choice**: DistilBERT-base-uncased

**Why DistilBERT?**
- ✅ 66M parameters (40% smaller than BERT)
- ✅ 60% faster inference
- ✅ Retains 97% of BERT performance
- ✅ Fits on consumer hardware (M1/M2 Macs)

**Architecture**:
```
Input: Recipe Text (all fields concatenated)
         ↓
DistilBERT Encoder (6 layers, 768 hidden dim)
         ↓
[CLS] Token Representation (768-dim)
         ↓
Linear Classification Head (768 → 6)
         ↓
Softmax → Chef Prediction
```

**Key Design Decisions**:
- Single linear head (no hidden layer)
- CrossEntropyLoss handles softmax
- GELU activations inside DistilBERT

---

## ⚙️ Training Configuration

**Hyperparameters**:
- Learning rate: `2e-5` (AdamW)
- Batch size: `8/16` (train/eval)
- Max length: `512` tokens
- Padding: `longest` (dynamic)
- Epochs: `5` with early stopping (patience=2)
- Warmup: `10%` of steps

**Data Split**:
- Train: 2,399 samples (80%)
- Validation: 600 samples (20%)
- **Stratified** to preserve class balance

**Hardware**:
- Device: MPS (Apple Silicon GPU)
- Training time: ~20-25 minutes
- Thermal management: `nice -n 15` priority

---

## 📊 Results

### Final Performance

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | **90.17%** |
| **Macro-F1 Score** | **89.67%** |
| Train Loss | 0.4042 |

### Baseline Comparison

| Method | Accuracy | Improvement |
|--------|----------|-------------|
| Weak Baseline | 30.0% | - |
| Strong Baseline | 43.0% | +13.0 pp |
| **DistilBERT (Ours)** | **90.17%** | **+60.17 pp** |

### Learning Progress

| Epoch | Val Acc | Val F1 | Status |
|-------|---------|--------|--------|
| 1 | 81.0% | 79.1% | ✅ Quick validation |
| 2 | 84.2% | 82.5% | ↗️ Steady improvement |
| 3 | 87.5% | 86.2% | ↗️ |
| 4 | 89.0% | 88.1% | ↗️ |
| 5 | **90.17%** | **89.67%** | 🎯 Final model |

---

## 🔬 Analysis & Discussion

**What Worked Well**:
- ✅ **Balanced performance**: F1-macro (89.67%) ≈ Accuracy (90.17%)
  - Model handles class imbalance effectively
- ✅ **No overfitting**: Train loss 0.40 reasonable, validation stable
- ✅ **Efficient transfer learning**: Pre-trained DistilBERT adapts well
- ✅ **All fields matter**: 47pp improvement over description-only

**Key Insights**:
- **Steps field is critical** (42% of tokens)
  - Cooking instructions contain chef-specific patterns
- **Stratification essential** with 2.17x imbalance
- **Dynamic padding** saved 40% memory vs max_length padding
- **Batch size 8→16** minimal impact on final accuracy

**Limitations**:
- ⚠️ Dataset may have errors (recipes from same source?)
- ⚠️ No per-class metrics reported (some chefs easier?)
- ⚠️ Single model (no ensemble or cross-validation)
- ⚠️ English-only, limited to 6 chefs

---

## 💡 Challenges & Solutions

**Challenge 1: Mac Overheating** 🔥
- Problem: Full-speed training made computer unusable
- Solution: "Chill Mode" config
  - Batch size 16 → 8 (50% GPU usage)
  - `nice -n 15` priority (system responsive)
  - Result: Same accuracy, usable system

**Challenge 2: Class Imbalance**
- Problem: 2.17x difference in class sizes
- Solution: Stratified splitting
  - Preserves class distribution in train/val
  - Macro-F1 metric weighs all classes equally
  - Result: Balanced performance (89.67% F1)

**Challenge 3: Memory Constraints**
- Problem: Long recipes (max 627 tokens)
- Solution: Smart preprocessing
  - Dynamic padding (`longest` per batch)
  - max_length=512 covers 98.2%
  - Saved 40% memory
  - Result: Fits on 16GB unified memory

---

## 🎯 Deliverables

**1. Trained Model** ✅
- DistilBERT fine-tuned on 2,999 recipes
- 90.17% validation accuracy
- Saved at `experiments/distilbert_text_only/artifacts/final_model/`

**2. Code & Documentation** ✅
- Training script: `train.py` with CLI
- Prediction script: `predict.py` for test set
- Config system: YAML-based hyperparameters
- README: Complete usage guide

**3. Paper** ✅
- 2-page report (LaTeX)
- Results, analysis, limitations
- Critical discussion of approach

**4. Test Predictions** ⏳
- Format: `results.txt` with chef IDs
- Ready to generate with `predict.py`

---

## 🚀 Next Steps

**Immediate** (for submission):
1. Generate test set predictions → `results.txt`
2. Finalize LaTeX paper (compile, proofread)
3. Submit code + paper + predictions

**Future Improvements**:
- 🔮 Ensemble multiple runs (boost accuracy)
- 🔮 Per-class analysis & confusion matrix
- 🔮 Hyperparameter tuning (lr, batch, epochs)
- 🔮 Try larger models (BERT-base, RoBERTa)
- 🔮 Data augmentation (paraphrase recipes?)

---

## 📚 References

- Sanh et al. (2019). "DistilBERT, a distilled version of BERT"
- Hugging Face Transformers library
- PyTorch & Accelerate for Apple Silicon

---

## 🙏 Thank You!

**Team**: NLP Group 2

**Key Achievement**: **90.17% accuracy** (+60.17pp over weak baseline)

**Code**: `github.com/nbirchde/NLP_group2`

**Questions?**

---

## 📎 Appendix: Technical Details

**Environment**:
- Python 3.13.4
- transformers 4.56.2
- torch 2.8.0 (MPS backend)
- accelerate 1.10.1

**Training Logs**:
- `experiments/distilbert_text_only/chill_training.log`
- Final metrics: `artifacts/final_metrics.txt`

**Reproducibility**:
```bash
# Quick test (1 epoch)
python experiments/distilbert_text_only/train.py --config configs/quick_test.yaml

# Full training (5 epochs, chill mode)
./train_chill.sh

# Generate predictions
python experiments/distilbert_text_only/predict.py \
  --model-path experiments/distilbert_text_only/artifacts/final_model \
  --test-path data/test-no-labels.csv \
  --output results.txt
```

**Dataset Stats**:
- Total samples: 2,999
- Train: 2,399 (80%)
- Validation: 600 (20%)
- Test: 750 (unlabeled)

**Model Size**:
- Parameters: 66M
- Disk space: ~256MB (safetensors)
- Memory footprint: ~2GB (inference)
