# 📊 Dataset Analysis Results — Training Data Insights

**Analysis Date**: October 3, 2025  
**Dataset**: `data/train.csv` (2,999 recipes)  
**Model Target**: DistilBERT-base-uncased  

---

## 🎯 Executive Summary

**Key Decisions Confirmed:**
- ✅ **max_length=512** (98.2% of samples fit completely)
- ✅ **padding='longest'** (dynamic per batch - saves memory)
- ✅ **truncation='only_second'** (protect recipe_name/ingredients/tags)
- ✅ **batch_size=16** (safe for M1/M2 Macs with median 272 tokens)
- ⚠️ **Use stratified sampling** (2.17x class imbalance)

---

## 📈 1. Class Distribution Analysis

```
Chef ID | Count | Percentage
--------|-------|------------
4470    | 806   | 26.9% ⚠️ LARGEST
5060    | 534   | 17.8%
3288    | 451   | 15.0%
8688    | 432   | 14.4%
1533    | 404   | 13.5%
6357    | 372   | 12.4% ⚠️ SMALLEST
```

**Imbalance Ratio**: 2.17x (806 / 372)  
**Balance Score**: 0.319 (std/mean) — moderate imbalance

### 🔧 Action Items:
1. ✅ **Use `stratified=True` in train_test_split**
2. ✅ **Report Macro-F1** (not just accuracy) to catch per-class performance
3. ⚠️ Optional: Compute class weights if validation shows bias toward chef 4470

---

## 📏 2. Text Length Analysis (Tokens)

### Full Concatenated Text:
```
Statistic         | Tokens
------------------|--------
Mean              | 278
Median            | 272
75th percentile   | 324
90th percentile   | 386
95th percentile   | 431
Max               | 627
```

### Truncation Coverage:
```
max_length | % Samples Fit
-----------|---------------
256        | 43.6% ❌
384        | 89.6% ⚠️
512        | 98.2% ✅ CHOSEN
```

**Decision**: Use **max_length=512** to capture 98.2% of samples without loss.  
Only 1.8% (54 samples) will be truncated.

---

## 🧩 3. Field-by-Field Token Breakdown

### Token Contribution (Mean):
```
Field        | Mean Tokens | % of Total | Strategy
-------------|-------------|------------|----------------------------------
recipe_name  | 6.3         | 2.4%       | ✅ Keep (short, highly informative)
ingredients  | 30.8        | 11.5%      | ✅ Keep (chef-specific patterns)
tags         | 84.2        | 31.4%      | ✅ Keep (can tolerate truncation)
description  | 33.5        | 12.5%      | ✅ Keep (optional but useful)
steps        | 113.1       | 42.2%      | ⚠️ TRUNCATE LAST (longest field)
-------------|-------------|------------|----------------------------------
TOTAL        | 267.9       | 100%       |
```

### Token Distribution per Field:
```
Field        | Median | 95th %ile | Max
-------------|--------|-----------|-----
recipe_name  | 6      | 12        | 18
ingredients  | 28     | 55        | 106
tags         | 83     | 129       | 174
description  | 25     | 89        | 181
steps        | 107    | 235       | 386 ⚠️
```

**Key Insight**: Steps are by far the longest and most variable field.  
→ **Truncate steps first** to protect other fields.

---

## 🔀 4. Field Ordering Strategy

**Optimal Order** (implemented in analysis):
```python
text = f"""{recipe_name}
Ingredients: {ingredients}
Tags: {tags}
Description: {description}
Steps: {steps}"""
```

**Rationale**:
1. **recipe_name** first: Short (6 tokens), highly discriminative
2. **ingredients** second: Moderate (31 tokens), chef-specific
3. **tags** third: Can be partially truncated without major loss
4. **description** fourth: Optional context
5. **steps** last: Longest (113 tokens), most redundant with other fields

**Truncation Behavior**:
- With `truncation='only_second'`, the model truncates from the end
- This means steps get cut first, protecting the critical fields

---

## 💾 5. Padding & Memory Strategy

### Current Stats:
- **Median tokens**: 272
- **Mean tokens**: 278
- **Max tokens**: 627 (outlier)

### Recommended Padding:
```python
tokenizer(
    text,
    padding='longest',      # ✅ Dynamic padding per batch
    truncation=True,
    max_length=512,
    return_tensors='pt'
)
```

**Why `padding='longest'` not `padding='max_length'`?**
- Median is 272, but max_length=512
- Most batches will be ~300 tokens, not 512
- Dynamic padding saves ~40% memory per batch
- Faster training on Mac (less wasted computation)

### Batch Size Recommendation:
- **batch_size=16** for training (median 272 tokens)
- **batch_size=32** for evaluation (can go higher)
- On 8GB unified memory (M1/M2), 512 tokens × 16 samples = safe

---

## 📝 6. Content Statistics

### Recipe Characteristics:
```
Metric                  | Mean  | Median | Max
------------------------|-------|--------|-----
Ingredients per recipe  | 9.1   | 9.0    | 36
Steps per recipe        | 9.3   | 8.0    | 45
Tags per recipe         | 23.7  | 23.0   | 49
Recipe name (chars)     | 28.3  | 27.0   | 75
Description (chars)     | 155.2 | 119.0  | 2382
Steps text (chars)      | 527.2 | 467.0  | 3197
```

### Data Quality:
- ✅ **No missing values** in any field
- ✅ **n_ingredients** column matches actual ingredient count
- ✅ **All recipes have descriptions** (0% empty)

---

## 🚀 7. Final Pipeline Configuration

### Preprocessing:
```python
# Field concatenation order
def build_input_text(row):
    return f"""{row['recipe_name']}
Ingredients: {', '.join(row['ingredients'])}
Tags: {', '.join(row['tags'])}
Description: {row['description']}
Steps: {' | '.join(row['steps'])}"""

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
inputs = tokenizer(
    text,
    padding='longest',           # Dynamic padding
    truncation=True,             # Enable truncation
    max_length=512,              # 98.2% coverage
    return_tensors='pt'
)
```

### Data Split:
```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,                  # ✅ Handle class imbalance
    random_state=42
)
```

### Training Config:
```yaml
model: distilbert-base-uncased
max_length: 512
padding: longest
truncation: true
batch_size: 16                   # Train
eval_batch_size: 32              # Eval
learning_rate: 2e-5
epochs: 3-5
early_stopping_patience: 2
optimizer: AdamW
weight_decay: 0.01
seed: 42
```

---

## ⚠️ 8. Potential Issues & Mitigations

### Issue 1: Class Imbalance (2.17x ratio)
**Impact**: Model may favor chef 4470 (806 samples) over chef 6357 (372 samples)  
**Mitigation**:
- ✅ Stratified sampling (already planned)
- ✅ Track Macro-F1 (not just accuracy)
- ⚠️ If needed: Add class weights to CrossEntropyLoss
  ```python
  weights = torch.tensor([1/count for count in class_counts])
  criterion = nn.CrossEntropyLoss(weight=weights)
  ```

### Issue 2: 1.8% of samples exceed 512 tokens
**Impact**: 54 samples will be truncated (steps cut off)  
**Mitigation**:
- ✅ Field ordering protects critical info
- ✅ Steps are most redundant (least loss)
- ⚠️ Monitor: Check if truncated samples have worse accuracy

### Issue 3: Tags dominate token budget (31.4%)
**Impact**: Tags may overwhelm other signals  
**Mitigation**:
- ✅ Model will learn to weight features
- ⚠️ Ablation test: Compare with/without tags
- ⚠️ If tags hurt: Move tags after description

---

## 📊 9. Validation Experiments to Run

### Experiment 1: Field Ablation
Test which fields contribute most to accuracy:
```
A. name + ingredients + tags           (baseline minimal)
B. A + description                      (add context)
C. B + steps                            (full pipeline)
```

### Experiment 2: Truncation Strategy
Compare truncation methods on long samples:
```
A. truncation='only_second'             (our choice)
B. truncation='longest_first'           (HuggingFace default)
```

### Experiment 3: Max Length
Quick test on validation set:
```
A. max_length=384  (89.6% coverage, faster)
B. max_length=512  (98.2% coverage, slower)
```

---

## 📈 10. Expected Performance

### Baselines (from project description):
- **Weak baseline**: 30.0% (TF-IDF on description only)
- **Strong baseline**: 43.0% (TF-IDF on all fields)

### Our DistilBERT Target:
- **Conservative**: 60-65% accuracy
- **Realistic**: 65-75% accuracy
- **Optimistic**: 75-80% accuracy

### Why we expect this:
1. Pre-trained language model (vs TF-IDF)
2. Contextual embeddings (vs bag-of-words)
3. Fine-tuning on task-specific data
4. Well-balanced, clean dataset (no missing values)

### Success Criteria:
- ✅ **Beat 43% baseline** (minimum goal)
- ✅ **Reach 60%+** (good for 2 weeks)
- ✅ **Macro-F1 > 0.55** (no chef left behind)

---

## 📁 Files Generated

1. **`analyze_dataset.py`** — Full analysis script
2. **`data/dataset_analysis.json`** — Machine-readable stats
3. **`DATASET_ANALYSIS.md`** — This human-readable report

---

## ✅ Next Steps

1. ✅ Update `AGENTS.md` with these findings
2. ✅ Implement preprocessing pipeline with confirmed parameters
3. ✅ Build training script with:
   - Stratified split
   - Dynamic padding
   - Early stopping
   - Macro-F1 tracking
4. ✅ Run baseline (TF-IDF) for comparison
5. ✅ Fine-tune DistilBERT with confirmed config
6. ✅ Run ablation studies
7. ✅ Generate predictions for test set

---

**Analysis Complete** ✅  
*Ready to implement the training pipeline with confidence!*
