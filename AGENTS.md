# 🤖 NLP Group 2 - Chef Classification Project

**Goal**: Build a model that predicts which chef created a recipe based on recipe features.

**Team**: 4 people  
**Deadline**: October 15, 2025  
**Language**: Python 3

---

## 📊 The Challenge

- **Dataset**: Recipe data with features (name, tags, steps, description, ingredients, etc.)
- **Task**: Multi-class classification (6 chefs)
- **Metric**: Accuracy
- **Baselines to Beat**:
  - Weak baseline (TF-IDF on description only): **30.0% accuracy** → 2 pts
  - Strong baseline (TF-IDF on all fields): **43.0% accuracy** → +2 pts

---

## 📂 Data Format

```
chef_id | recipe_name | date | tags | steps | description | ingredients | n_ingredients
```

**Training**: `data/train.csv` (3000 recipes with labels)  
**Test**: `data/test-no-labels.csv` (no chef_id, predict these!)

---

## 🎯 Current Phase: Implementation ✅

**Decision**: Fine-tune **DistilBERT-base-uncased** with text-only approach.  
**Dataset**: 2,999 samples, 6 chefs (imbalance 2.17x), median 272 tokens.  
**Full analysis**: `decisions/dataset_analysis/` (Oct 3, 2025)

### Critical Stats:
- **Class imbalance**: 2.17x (372–806 samples) → **stratify=True**
- **Token lengths**: median=272, 98.2% fit in 512 → **max_length=512, padding='longest'**
- **Field weights**: steps=42% → **truncate last**, protect name/ingredients
- **Batch size**: **16** safe for M1/M2 (median 272 tokens)

---

## 📝 Deliverables

1. **Code**: `.py` or `.ipynb` files (no model weights)
2. **Results**: `results.txt` with predicted chef_ids for test set
3. **Paper**: `NUM.pdf` - 2-page report (max 16 points)
   - Describe models
   - Present results
   - Discuss limitations
   - Critical analysis of data/outputs

---

## 💡 Tips from Instructions

- This is M.Sc level - figure out the best approach yourself
- Be systematic: evaluate every change
- Preprocessing on train = preprocessing on test
- Look at inputs AND outputs, not just accuracy numbers
- Dataset may have errors/imbalance - discuss in paper
- No 100% accuracy expected (research problem!)
- Language is complex - models must generalize

---

## 🏗️ Project Structure

```
research/          # Research phase - exploring approaches
experiments/       # Implementation & testing phase
decisions/         # Team decisions & meeting notes
data/             # Training & test datasets
Project-Template/ # LaTeX template for paper
```

---

## 🚀 Getting Started

Each team member should:
1. Explore the training data
2. Research different approaches
3. Document findings in markdown
4. Share insights with team
5. Contribute to collective decision-making

**Remember**: We're building context together, so write everything down!
