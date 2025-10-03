# Next Tasks - Prediction Export & Paper Writing

## Task 1: Prediction Export Module

### Objective
Create a script that loads the trained model and generates `results.txt` with chef predictions for the test set.

### Requirements
- **Input**: `data/test-no-labels.csv` (no chef_id column)
- **Output**: `results.txt` with one chef_id per line (3000+ predictions)
- **Model**: Load from `experiments/distilbert_text_only/artifacts/final_model/`
- **Preprocessing**: Must match training pipeline exactly

### Implementation Checklist
- [ ] Create `experiments/distilbert_text_only/predict.py`
- [ ] CLI interface: `--model-path`, `--test-path`, `--output` flags
- [ ] Load trained model and tokenizer from checkpoint
- [ ] Apply same text concatenation as training (recipe_name → ingredients → tags → description → steps)
- [ ] Tokenize with max_length=512, padding='longest', truncation=True
- [ ] Batch inference for efficiency
- [ ] Map predictions back to original chef_id strings (not integer labels)
- [ ] Write results.txt with one prediction per line
- [ ] Add README section documenting usage

### Technical Notes
```python
# Key imports needed
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data import load_recipes_csv
from src.dataset import add_text_column

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Map integer predictions back to chef IDs
id2label = model.config.id2label  # Already stored in model config
predictions = [id2label[pred_id] for pred_id in predicted_labels]
```

### Expected Output Format
```
results.txt:
4470
4898
6357
...
```

### Validation
- Count lines in results.txt == test set size
- All predictions are valid chef IDs (one of: 4470, 4898, 6357, ...)
- No errors on malformed test samples

---

## Task 2: LaTeX Paper Skeleton

### Objective
Draft the 2-page paper structure in `Project-Template/template.tex` following the assignment requirements.

### Paper Structure (Max 2 Pages)

#### 1. Introduction (~0.25 pages)
- [ ] Brief problem statement (chef classification from recipes)
- [ ] Dataset overview (6 chefs, 2999 train samples, class imbalance 2.17x)
- [ ] Approach summary (DistilBERT fine-tuning on concatenated text)

#### 2. Model Description (~0.5 pages)
- [ ] Architecture: DistilBERT-base-uncased (66M params, 6 layers, 768 hidden)
- [ ] Classification head: Linear layer (768 → 6 classes)
- [ ] Input preprocessing: Text concatenation strategy with field ordering
- [ ] Tokenization: max_length=512, dynamic padding, truncation from end
- [ ] Training setup: AdamW optimizer, lr=2e-5, batch=16, stratified split

#### 3. Results (~0.5 pages)
- [ ] Baseline comparisons:
  - Weak baseline (TF-IDF on description): 30.0%
  - Strong baseline (TF-IDF on all fields): 43.0%
  - Our model (DistilBERT, 5 epochs): **TBD** (run full training first)
- [ ] Include table with accuracy and F1-macro scores
- [ ] Brief analysis of per-class performance if time permits

#### 4. Discussion & Limitations (~0.5 pages)
- [ ] Data quality observations:
  - Class imbalance (2.17x) mitigated with stratification
  - Token length distribution (98.2% fit in 512 tokens)
  - Potential label noise or recipe ambiguity
- [ ] Model limitations:
  - Single text modality (no ingredient quantities, cooking methods)
  - Simple linear head (vs. deeper classification layer)
  - Fixed field ordering may not be optimal
- [ ] Critical analysis:
  - Does model actually learn chef style vs. recipe topics?
  - How would it generalize to new chefs?
  - Computational cost considerations

#### 5. Conclusion (~0.25 pages)
- [ ] Summary of findings
- [ ] Comparison to baselines
- [ ] Future work suggestions

### LaTeX Checklist
- [ ] Update `\title{}` with project name
- [ ] Add author names and affiliations
- [ ] Create results table using `\begin{table}`
- [ ] Add references to `biblio.bib` if needed
- [ ] Compile with `pdflatex template.tex` to verify
- [ ] Check page count (max 2 pages!)

### Key Points to Address
From assignment instructions:
- ✅ Describe models used
- ✅ Present results (with baseline comparisons)
- ✅ Discuss limitations (data quality, model choices)
- ✅ Critical analysis (not just "we did X and got Y")

---

## Dependency: 5-Epoch Training Run

**Before Task 2 (paper)**, complete the full training:

```bash
cd /Users/nicholasbirchdelacalle/Documents/NL/NLP_project/NLP_group2
.venv/bin/python experiments/distilbert_text_only/train.py --config configs/base.yaml
```

**Expected duration**: ~10-15 minutes (5 epochs × 2 min/epoch)

**After training completes**:
1. Check `experiments/distilbert_text_only/artifacts/final_metrics.txt`
2. Record final accuracy and F1-macro in build_logs.md
3. Use these metrics in the paper results section

---

## Timeline Estimate

- **Task 1 (Prediction Export)**: 30-45 minutes
  - Script implementation: 20 min
  - Testing & debugging: 15 min
  - Documentation: 10 min

- **5-Epoch Training**: 10-15 minutes (automated)

- **Task 2 (Paper Skeleton)**: 1-2 hours
  - Structure & sections: 30 min
  - Results table: 15 min
  - Discussion & analysis: 45 min
  - Formatting & compilation: 15 min

**Total**: ~2-3 hours of work

---

## Priority Order

1. **Run 5-epoch training first** (can work in background)
2. **Implement prediction export** (independent of training results)
3. **Draft paper sections** (needs training metrics to complete results)
4. **Final polish** (proofread, check page limit, compile PDF)

---

## Questions to Resolve

- [ ] Should we ensemble multiple runs or use single best model?
- [ ] Do we need cross-validation results or is 80/20 split sufficient?
- [ ] Any specific formatting requirements for results.txt?
- [ ] Paper submission format: PDF only or source files too?

---

**Status**: Ready to dispatch  
**Blockers**: None  
**Next Agent**: Can start immediately on prediction export or paper skeleton
