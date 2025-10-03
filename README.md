# 🤖 NLP Group 2 - Chef Classification Project

**Goal**: Build a model that predicts which chef created a recipe based on recipe features.

**Team**: 4 people | **Deadline**: October 15, 2025 | **Language**: Python 3

---

## 🎯 Final Results

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | **90.17%** |
| **Macro-F1 Score** | **89.67%** |
| Train Loss | 0.4042 |

**Baseline Comparisons**:
- Weak baseline (TF-IDF on description): 30.0% → **+60.17 pp improvement** ✅
- Strong baseline (TF-IDF on all fields): 43.0% → **+47.17 pp improvement** ✅

**Model**: DistilBERT-base-uncased fine-tuned for 5 epochs

---

## 📊 The Challenge

- **Dataset**: Recipe data with features (name, tags, steps, description, ingredients, etc.)
- **Task**: Multi-class classification (6 chefs)
- **Metric**: Accuracy
- **Training**: `data/train.csv` (2,999 recipes with labels)
- **Test**: `data/test-no-labels.csv` (predict these!)

---

## 🚀 Quick Start

### Training

```bash
# 1. Install dependencies
pip install transformers datasets torch accelerate scikit-learn

# 2. Quick test (1 epoch, ~5 minutes)
python experiments/distilbert_text_only/train.py --config configs/quick_test.yaml

# 3. Full training (5 epochs, ~20-25 minutes, Mac-friendly)
./train_chill.sh

# Check results
cat experiments/distilbert_text_only/artifacts/final_metrics.txt
```

### Generate Test Predictions

```bash
python experiments/distilbert_text_only/predict.py \
  --model-path experiments/distilbert_text_only/artifacts/final_model \
  --test-path data/test-no-labels.csv \
  --output results.txt
```

---

## 🧊 Thermal-Friendly Training (Optional)

- `configs/chill_mode.yaml`: lower batch size (8/16) for gentler runs
- `train_chill.sh`: wraps the chill config with `nice` priority and logs to `experiments/distilbert_text_only/chill_training.log`

Manual invocation:

```bash
source .venv/bin/activate
nice -n 15 python experiments/distilbert_text_only/train.py --config configs/chill_mode.yaml
```

---

## 📈 Visualizations & Presentation

**Generate Publication-Quality Figures**:
```bash
# Install visualization dependencies
python scripts/install_viz_deps.py

# Generate all figures
python scripts/generate_visualizations.py
```

**Output** (`results/figures/`):
- `baseline_comparison.png` - Bar chart comparing our model to baselines
- `training_curves.png` - Loss, accuracy, F1 evolution across epochs
- `dataset_overview.png` - Class distribution, token stats, field contributions
- `metrics_summary.png` - Comprehensive results dashboard

**Presentation**:
- Markdown slides: `presentation/PRESENTATION.md`
- Includes all key results, architecture, challenges, and discussion

---

## 📂 Project Structure

```
├── data/                    # Training & test datasets
├── experiments/             # Implementation & testing
│   └── distilbert_text_only/
│       ├── train.py        # Training script
│       ├── predict.py      # Test set inference
│       └── artifacts/      # Trained models & checkpoints
├── configs/                # Training configurations
│   ├── base.yaml          # Full-speed training
│   ├── chill_mode.yaml    # Mac-friendly settings
│   └── quick_test.yaml    # 1-epoch validation
├── scripts/                # Utility scripts
│   └── generate_visualizations.py
├── presentation/           # Slide deck
├── decisions/             # Team decisions & analysis
├── research/              # Research phase notes
├── results/               # Figures & predictions
└── Project-Template/      # LaTeX paper
```

---

## 📝 Deliverables

1. **Code**: Training & prediction scripts ✅
2. **Model**: Fine-tuned DistilBERT (90.17% accuracy) ✅
3. **Results**: `results.txt` with predicted chef_ids for test set ⏳
4. **Paper**: 2-page report with results & analysis ⏳
5. **Visualizations**: Publication-quality figures ✅
6. **Presentation**: Slide deck with all findings ✅

---

## 💡 Key Features

- **Chill Mode Training** 🌡️: Mac-friendly config that keeps your computer cool
- **Dry-Run Testing** 🔍: Validate setup before full training
- **Modular Config System** ⚙️: YAML-based hyperparameter management
- **Complete Documentation** 📚: README, experiment guide, build log
- **Professional Visualizations** 📊: Publication-ready figures
- **Git-Safe** 🔒: Large files excluded from version control

---

## 📚 Documentation & Assets

- **Training Guide**: `experiments/distilbert_text_only/README.md`
- **Dataset Analysis**: `decisions/dataset_analysis/DATASET_ANALYSIS.md`
- **Research Notes**: `research/` (team rationale and experiments)
- **Build Log**: `context/build_logs.md`
- **Slide Deck**: `presentation/PRESENTATION.md`
- **Paper Template**: `Project-Template/template.tex`

---

## 🏆 Team

NLP Group 2 - Building context together! 🚀

**Remember**: Write everything down - we're creating a comprehensive knowledge base.
