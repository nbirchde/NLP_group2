# A possible pipeline (using pretrained transformer instead of a simple tokenizer seems interesting)

## 1. Data Preprocessing
- **Text fields**: `recipe_name`, `description`, `ingredients`, `steps`
  - Concatenate into one string per recipe
  - Tokenize using a pretrained Transformer tokenizer (e.g. DistilBERT, RoBERTa)
  - Truncate/pad to max sequence length (e.g. 256 tokens)
- **Structured fields**:  
  - `n_ingredients` → numeric  
  - `date` → extract features (year, month, weekday, cyclical encoding)  
  - `tags` → one-hot or embeddings  

## 2. Model Architecture
- **Text encoder**: Pretrained Transformer (BERT/DistilBERT) → contextual embedding (CLS token)
- **Structured encoder**: MLP for numeric + categorical features
- **Fusion**: Concatenate `[Text_Embedding ; Structured_Embedding]`
- **Classifier head**:  
  - Dense → GELU → Dropout  
  - Dense → GELU → Dropout  
  - Output: Softmax over `n_chefs`

## 3. Training
- **Loss**: CrossEntropyLoss (optionally with class weights / label smoothing)  
- **Optimizer**: AdamW with two parameter groups  
  - low LR (e.g. 2e-5) for Transformer  
  - higher LR (e.g. 1e-3) for classifier head  
- **Evaluation metrics**: Accuracy, Macro-F1 (better with imbalanced chefs)  
- **Regularization**: Dropout (0.1–0.3), Early stopping  

## 4. Variants
- **Baseline**: TF-IDF + Logistic Regression  
- **Feature extraction**: Freeze Transformer, train only classifier  
- **Fine-tuning**: End-to-end training of Transformer + classifier  
- **Alternative**: Use extracted embeddings + XGBoost/LightGBM with numeric features
