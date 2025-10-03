# Final Pipeline Decision — Chef Classification (DistilBERT, text‑only, Mac‑friendly)

**Goal:** Classify ~3,000 recipes into 6 chefs, beat TF‑IDF baselines, keep it simple, fast, and reproducible on Macs (M‑series), with a 2‑page write‑up.

**Updated**: October 3, 2025 (with dataset analysis)

---

## 0) Big Picture Choices (why we chose them)
- **Model:** `distilbert-base-uncased` (small, fast, strong). RoBERTa is heavier; we prefer speed + simplicity for 2 weeks.
- **Inputs:** **Text‑only** fusion (title + ingredients + tags + description + steps). No separate MLP for structured features now.
- **Head:** **Single linear classifier** on top of `[CLS]` (no extra hidden layers). Less overfitting, fewer knobs.
- **Tokenizer/length:** DistilBERT tokenizer, `max_length=512`, truncate tail. **Order fields** to protect key info from truncation.
- **Training:** AdamW, LR `2e-5`, epochs 3–5 with early stopping, batch 16 (MPS on Mac). Metrics: Accuracy (+ Macro‑F1).

> **Plain English:** We fine‑tune a small but capable language model on our combined recipe text. Keep the head simple and train a few epochs. The whole thing fits on our Macs and avoids over‑engineering.

---

## 1) Data Preprocessing (decisions) 

### Dataset Stats (analyzed Oct 3, 2025):
- **Total samples**: 2,999
- **Class imbalance**: 2.17x (chef 4470: 806 samples, chef 6357: 372 samples)
  - *What this means*: Without stratification, model might just predict chef 4470 for everything
- **Token lengths** (DistilBERT): median=272, mean=278, 98.2% fit in 512
  - *Why this matters*: Most recipes are short → padding='longest' saves memory
- **Field contributions**: steps=42%, tags=31%, description=13%, ingredients=12%, name=2%
  - *Implication*: Steps are longest but also most redundant → truncate last

**Use fields:**
- `recipe_name` ✅ (often most informative, ~6 tokens)
- `ingredients` ✅ (chef tendencies, ~31 tokens)
- `tags` ✅ (as text, not one‑hot, ~84 tokens)
- `description` ✅ (voice/tone, ~34 tokens)
- `steps` ✅ (truncate last if too long, ~113 tokens)
- `date` ❌ (low signal, adds noise)
- `n_ingredients` ➖ *Optional as text only* (e.g., "NumIngredients: 7"). No separate numeric pipeline.

**Concatenate (protect early info):**
```python
f"""{recipe_name}
Ingredients: {ingredients}
Tags: {tags}
Description: {description}
Steps: {steps}"""
```
- Tokenize with DistilBERT tokenizer, `max_length=512`, `truncation=True`, `padding='longest'`.
- **Stratified split**: 80/20 train/val on `chef_id` (critical for 2.17x imbalance).
  - *Stratification*: Keeps same chef % in train AND val (prevents "chef 6357 only in train" problem)

**Why 512 not 384?**
- 384 truncates 10% of samples (300 recipes lose steps)
- `padding='longest'` pads each batch to its longest sample (usually ~280), not always 512
- Memory savings from 384 aren't huge because of dynamic padding
- If training too slow → ablate to 384 and compare val accuracy

**Why padding='longest'?**
- Pads to longest sample in batch (e.g., batch max = 350 → pad to 350, not 512)
- Saves ~40% memory vs `padding=True` (which always pads to max_length)
- No accuracy loss (padding tokens are masked in attention anyway)

---

## 2) Model Architecture (decisions)
- **Encoder:** `distilbert-base-uncased` fine‑tuned end‑to‑end.
- **Pooling:** Use the `[CLS]` token representation for classification.
  - *What [CLS] is*: Special token at start; DistilBERT learns to encode whole sequence here
- **Head:** `Linear(768 → 6)` (no hidden layer). Dropout from DistilBERT is enough initially.

**Why not a deeper head (GELU, multi‑layer)?** With 3k samples, extra layers add parameters and tuning but little guaranteed gain. We keep it lean; if we add one hidden layer later, we'll use **GELU** then.

---

## 3) Training (decisions)
- **Loss:** CrossEntropyLoss.
- **Optimizer:** AdamW, **single LR** `2e-5`, `weight_decay=0.01`. (Skip dual LRs for simplicity.)
- **Schedule:** 3–5 epochs, early stopping on **val accuracy** (patience 1–2), warmup ~5–6% (optional).
- **Batch sizes:** train **16**, eval **32**. Use MPS on Apple Silicon if available.
- **Metrics reported:** Accuracy (main), **Macro‑F1** (class balance - critical with 2.17x imbalance).
  - *Macro-F1*: Average F1 across all 6 chefs (catches if model ignores rare chefs)
- **Seed:** Fix random seed for reproducibility.

**Why:** Canonical fine‑tuning recipe; fast iteration on Mac; minimal hyperparams to babysit. Batch size 16 safe because median is only 272 tokens.

---

## 4) Baseline + Variants (keep it lean)
- **Baseline:** TF‑IDF + Logistic (one run for report context).
- **Ablations (tiny, quick):**
  - **A:** name + ingredients + tags
  - **B:** A + description
  - **C:** B + steps
> Report which jump gives the biggest gain; this justifies our field choices.

- **Only if time permits:** Try `distilroberta-base` at `max_length=384` as a speed/accuracy trade; keep whichever wins **val accuracy**.

---

## 5) Decisions vs Teammate Proposals (clear calls)
**Maxence's pipeline (text + structured fusion, deep head, dual LRs):**
- **Structured fusion (date one‑hot, tags one‑hot, MLP):** ❌ *Skip now.* Complexity↑, small expected gain. We already inject tags as text; date likely low signal.
- **Deep classifier head (Dense→GELU→Dropout ×2):** ❌ *Skip now.* Risk of overfitting; limited data. ✅ If later adding a hidden layer, choose **GELU**.
- **Two LR groups (2e‑5 encoder, 1e‑3 head):** ❌ *Skip now.* One LR works well and reduces tuning.
- **Macro‑F1 metric:** ✅ *Keep* as secondary metric (validated by 2.17x imbalance).

**Javier's pipeline (text‑only, RoBERTa‑base, careful ordering):**
- **Text‑only with field ordering:** ✅ *Adopt.* We use the same ordering logic.
- **RoBERTa‑base:** ➖ We choose **DistilBERT‑uncased** first (Mac‑friendly). RoBERTa can be a later variant if time allows.
- **Single linear head on CLS:** ✅ *Adopt.*
- **Max length 384–512:** ✅ We aim for **512** (98.2% coverage), drop to **384** if memory pushes back.
- **Warmup ~6%, early stopping:** ✅ *Adopt.*
- **Optional tiny MLP for `n_ingredients`:** ➖ Only as **text** for now; add real MLP **only** if it gives ≥1 pt consistent gain in quick tests.

---

## 6) Implementation Sketch (repo, scripts)
```
repo/
  data/
    train.csv  test.csv
  src/
    dataset.py          # load CSV, build text inputs, tokenizer
    model.py            # load distilbert, linear head
    train.py            # train loop (or HF Trainer), early stopping
    eval.py             # metrics, confusion matrix
    predict.py          # generate results.txt for test set
  configs/
    base.yaml           # lr, batch, epochs, max_len, seed
  results/
    best.ckpt  logs/
  decisions/
    dataset_analysis/   # full analysis, visualization (archived)
  README.md
```
**Defaults:** `max_len=512`, `lr=2e-5`, `batch=16`, `epochs=5`, `patience=2`, `seed=42`.

---

## 7) What we'll write in the 2‑page report (ready‑to‑lift)
- **Why DistilBERT:** best accuracy‑per‑minute on Macs; uncased is fine for recipes (case rarely matters).
- **Why text‑only:** most signal is in text; tags come as text; avoids fragile feature engineering.
- **Why simple head + one LR:** fewer knobs → faster convergence, lower overfitting.
- **Why field order:** protects title/ingredients from truncation; steps last (42% of tokens but most redundant).
- **Dataset insights**: 2.17x class imbalance → stratified split + Macro-F1; 98.2% samples fit in 512 tokens.
- **Evidence:** baseline vs our model; tiny ablations show which fields matter.

> **Success criteria:** Beat ~43% TF‑IDF baseline on validation; target 60–70%+ with DistilBERT. If close, try distilroberta‑base.

---

## Quick Checklist (execution)
- [x] Analyze dataset (2,999 samples, 2.17x imbalance, median 272 tokens)
- [ ] **Stratified split** train/val (80/20, critical!)
- [ ] Build text inputs in the chosen order
- [ ] Tokenize (`max_len=512`, `padding='longest'`, truncation)
- [ ] Fine‑tune DistilBERT (AdamW, `2e-5`, batch 16, epochs 3–5, early stopping)
- [ ] Track **Accuracy + Macro‑F1**
- [ ] Save best ckpt; run `predict.py` → `results.txt`
- [ ] Run baseline + ablations for the report

---

## 📝 Key Clarifications (TL;DR)

**Imbalance & Stratification:**
- Chef 4470: 806 samples, Chef 6357: 372 samples (2.17x ratio)
- Without stratification: model might predict chef 4470 for everything
- `stratify=True` keeps same chef % in train AND val splits

**Token Length & Padding:**
- Median: 272 tokens → most recipes are short
- `padding='longest'`: pads each batch to its longest sample, not always 512
- Saves ~40% memory vs always padding to 512

**Why 512 not 384:**
- 384 cuts 10% of samples (loses steps field for 300 recipes)
- Dynamic padding means we rarely actually use all 512 slots
- Easy to ablate later if training is slow

**Metrics:**
- Accuracy: main metric (what we submit)
- Macro-F1: average F1 across all 6 chefs (catches minority class problems)
