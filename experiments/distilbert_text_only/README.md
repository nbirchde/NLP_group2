# DistilBERT Text-Only Experiment

Primary fine-tuning run using `distilbert-base-uncased` on concatenated recipe text fields. Configuration sourced from `configs/base.yaml`.

Artifacts (checkpoints, logs, metrics) are saved under `experiments/distilbert_text_only/artifacts/`.

Implementation status:
- [ ] Data preprocessing/module wiring
- [ ] Training script
- [ ] Evaluation + prediction export

Notes:
- Keep max sequence length at 512 with truncation from the end to protect early fields.
- Stratify the train/validation split because class imbalance is 2.17x.
- Track both accuracy and macro-F1 for reporting.
