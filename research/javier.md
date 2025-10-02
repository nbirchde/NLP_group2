# Javier's Research
# A practical pipeline (text-only fine-tuning with clear ordering)

## 1. Data Preprocessing

* **Text fields**: `recipe_name`, `ingredients`, `tags`, `description`, `steps` 

  * **Order (to protect key info from truncation):**
    `recipe_name → ingredients → tags → description → steps`
  * **Concatenate** into one string with explicit markers:
    ```
    {recipe_name} </s> Ingredients: ing1, ing2, ... </s> Tags: tag1, tag2, ... </s> {description} </s> Steps: step1 | step2 | ...
    ```
  * **Tokenize** with the RoBERTa tokenizer
  * **Truncate/pad** to **384–512 tokens** (512 if possible; 384 if resources are tight)
* **Normalization (light):** trim spaces and odd control/HTML chars; **do not** lowercase; **do not** remove stopwords
* **Split:** stratified train/val/test by `chef_id`

## 2. Model Architecture

* **Text encoder:** `roberta-base` fine-tuned end-to-end
* **Classifier head:** single linear layer on top of the pooled CLS representation
* **Output:** Softmax over 6 chefs

## 3. Training
* **Loss:** CrossEntropyLoss (optionally with class weights if needed)
* **Optimizer:** AdamW, single LR (e.g., 2e-5), `weight_decay=0.01`, small warmup (~6%)
* **Schedule:** 5 epochs with early stopping on validation accuracy (patience 1–2)
* **Batch sizes:** train 16, eval 32; use mixed precision if available (MPS on Mac)

## 4. Variants
* **Baseline:** TF-IDF + Linear/SVM for reference
* **Resource-friendly:** `distilroberta-base` with `max_length=256–384`
* **Ablations:**
  * A: `name + ingredients + tags`
  * B: A + `description`
  * C: B + `steps` (last)
* **Small add-on (only if it clearly helps):** fuse **just `n_ingredients`** via a tiny MLP with CLS; keep only if it adds ≥1 point accuracy consistently
