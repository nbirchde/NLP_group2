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

## 🎯 Current Phase: Research

Everyone is exploring different approaches. We'll synthesize findings to make informed decisions.

### Key Questions to Answer:
1. **Which model should we fine-tune?**
   - BERT-based? RoBERTa? DistilBERT?
   - Classical ML? Ensemble?
   
2. **How should we preprocess the data?**
   - Which fields are most informative?
   - How to handle multi-field text?
   - Tokenization strategy?

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
