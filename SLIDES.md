# Chef Classification with DistilBERT

**NLP Group 2**

---

## The Task

- Predict which chef created a recipe (6 chefs)
- 2,999 training recipes, 823 test recipes
- Fields: name, ingredients, tags, description, steps

**Baselines to beat**:
- Weak (TF-IDF description only): 30%
- Strong (TF-IDF all fields): 43%

---

## Our Approach

**Model**: DistilBERT-base-uncased
- 66M parameters (40% smaller than BERT)
- Fine-tuned for 5 epochs
- Concatenate all text fields → classify

**Key decisions**:
- Stratified train/val split (handles 2.17x class imbalance)
- Max length 512 tokens (covers 98.2% of data)
- Field order protects critical info from truncation

---

## Results

### Performance

| Method | Accuracy |
|--------|----------|
| Weak Baseline | 30.0% |
| Strong Baseline | 43.0% |
| **Our Model** | **90.17%** |

**Improvement**: +47 percentage points over strong baseline!

**Validation F1-macro**: 89.67% (balanced across all chefs)

---

## Results - Baseline Comparison

![Baseline Comparison](../results/figures/baseline_comparison.png)

---

## Results - Training Progress

![Training Curves](../results/figures/training_curves.png)

Steady improvement, no overfitting (train loss: 0.40)

---

## Results - Predictions Match Training Distribution

![Distribution Comparison](../results/figures/distribution_comparison.png)

Model learned chef patterns, not just class frequencies!

---

## What Did the Model Learn?

**Chef signatures identified**:

1. **Health-focused chef** (Chef 5060)
   - "diabetic cooking", "low-fat", "fat-free sour cream"
   - Across different recipes: fish, potatoes, pancakes

2. **Make-ahead chef** (Chef 3288)
   - "OAMC" (Once A Month Cooking), batch recipes
   - "freeze for future use", family-friendly

3. **Quick & simple chef** (Chef 6357)
   - "15-minutes-or-less", single servings
   - Minimal ingredients

4. **Southern/traditional chef** (Chef 8688)
   - Bread machine recipes, cornbread, Creole
   - Holiday dishes

**Not just topics** - model distinguishes *how* chefs cook, not just *what* they cook!

---

## Challenges & Solutions

**Challenge 1**: Mac overheating during training
- **Solution**: "Chill mode" config
  - Reduced batch size (16 → 8)
  - Lower GPU usage, system stays responsive
  - Same results, ~25 min training time

**Challenge 2**: Class imbalance (2.17x)
- **Solution**: Stratified splitting + macro-F1 metric
  - Preserves class distribution in train/val
  - Macro-F1 (89.67%) ≈ Accuracy (90.17%) → balanced!

---

## Discussion

**Strengths**:
- Dramatic improvement over baselines (+47 pp)
- Learns chef-specific patterns (not just topics)
- Robust generalization (prediction dist matches training)

**Limitations**:
- Strong textual signals ("diabetic cooking", "OAMC")
  - Some recipes may be easy to classify
- Single model (no ensemble)
- Can't generalize to new chefs without retraining

**Critical question**: Style vs. topic?
- Evidence for both (health patterns across recipes, but also explicit keywords)
- High accuracy might indicate strong topical clustering in dataset

---

## Future Work

- Ensemble methods (boost accuracy further)
- Attention analysis (what does model focus on?)
- Data augmentation (paraphrase recipes)
- Multi-modal features (ingredient quantities, cooking temps)
- Test on new chefs (transfer learning)

---

## Summary

✅ **90.17% accuracy** (beat baseline by 47 pp)  
✅ **Learned chef signatures**: health-focus, make-ahead, quick, traditional  
✅ **Practical solutions**: Thermal management, class imbalance  
✅ **Critical analysis**: Acknowledged limitations, questioned high accuracy

**Key insight**: Look at the predictions, not just metrics! Model captures cooking philosophy across recipe types.

---

## Questions?

**Code & Results**: github.com/nbirchde/NLP_group2

Thank you!
