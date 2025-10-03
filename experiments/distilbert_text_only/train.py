#!/usr/bin/env python3
"""
DistilBERT Fine-Tuning Entry Point

This script fine-tunes DistilBERT-base-uncased for chef classification.
Supports dry-run mode for smoke testing without full training.

Usage:
    python train.py --config ../../configs/base.yaml
    python train.py --config ../../configs/base.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import TrainingConfig
from src.data import load_recipes_csv
from src.dataset import prepare_dataset
from src.tokenization import load_tokenizer, tokenize_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for chef classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (e.g., ../../configs/base.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run smoke test only (single forward pass, no training)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    print(f"Loading config from: {path}")
    config = TrainingConfig.from_yaml(str(path))
    print(f"✓ Config loaded: {config.model_name}, max_length={config.max_length}")
    return config


def load_and_prepare_data(config: TrainingConfig) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """Load raw CSV data and prepare train/val splits with text concatenation."""
    train_path = project_root / "data" / "train.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    print(f"\nLoading data from: {train_path}")
    recipe_data = load_recipes_csv(train_path)
    print(f"✓ Loaded {len(recipe_data.frame)} recipes")
    
    print(f"Preparing dataset with {len(config.text_fields)} text fields...")
    artifacts = prepare_dataset(
        data=recipe_data,
        text_fields=config.text_fields,
        label_column=config.label_column,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    
    train_size = len(artifacts.dataset["train"])
    val_size = len(artifacts.dataset["validation"])
    num_classes = len(artifacts.label2id)
    
    print(f"✓ Dataset prepared:")
    print(f"  - Train: {train_size} samples")
    print(f"  - Val: {val_size} samples")
    print(f"  - Classes: {num_classes}")
    
    return artifacts.dataset, artifacts.label2id, artifacts.id2label


def compute_metrics(eval_pred):
    """Compute accuracy and macro-F1 score for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
    }


def dry_run_test(model, tokenized_ds: DatasetDict, data_collator) -> None:
    """Run single forward pass to validate setup without training."""
    print("\n" + "=" * 60)
    print("DRY RUN MODE: Testing single forward pass")
    print("=" * 60)
    
    # Take small batch from train set
    train_samples = tokenized_ds["train"].select(range(min(4, len(tokenized_ds["train"]))))
    batch = data_collator([train_samples[i] for i in range(len(train_samples))])
    
    # Move batch to model device
    device = model.device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    print(f"\nBatch shape: {batch['input_ids'].shape}")
    print(f"Device: {device}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
    
    print(f"✓ Forward pass successful")
    print(f"  - Logits shape: {outputs.logits.shape}")
    print(f"  - Loss: {outputs.loss.item():.4f}")
    
    # Check predictions
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(f"  - Predictions: {predictions.tolist()}")
    print(f"  - True labels: {batch['labels'].tolist()}")
    
    print("\n✓ Dry run completed successfully!")
    print("=" * 60)


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # 1. Load configuration
    config = load_config(args.config)
    
    # 2. Load and prepare data
    dataset, label2id, id2label = load_and_prepare_data(config)
    
    # 3. Load tokenizer
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = load_tokenizer(config.model_name)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # 4. Tokenize dataset
    print(f"\nTokenizing dataset (max_length={config.max_length}, padding={config.padding_strategy})...")
    tokenized_ds = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=config.max_length,
        padding_strategy=config.padding_strategy,
        truncation_strategy=config.truncation_strategy,
    )
    
    # Remove extra columns that shouldn't be passed to the model
    columns_to_remove = [col for col in tokenized_ds["train"].column_names 
                        if col not in ["input_ids", "attention_mask", "labels"]]
    if columns_to_remove:
        tokenized_ds = tokenized_ds.remove_columns(columns_to_remove)
    
    print("✓ Tokenization complete")
    
    # 5. Load model
    print(f"\nLoading model: {config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    print(f"✓ Model loaded with {len(label2id)} output classes")
    
    # 6. Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 7. Dry-run mode: test and exit
    if args.dry_run:
        dry_run_test(model, tokenized_ds, data_collator)
        return
    
    # 8. Setup training arguments
    output_dir = project_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSetting up training...")
    print(f"Output directory: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=config.seed,
        report_to="none",  # Disable wandb/tensorboard
        use_mps_device=torch.backends.mps.is_available(),  # Use MPS on Mac if available
    )
    
    # 9. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )
    
    # 10. Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {config.epochs}")
    print(f"Train batch size: {config.train_batch_size}")
    print(f"Eval batch size: {config.eval_batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Early stopping patience: {config.patience}")
    print("=" * 60 + "\n")
    
    train_result = trainer.train()
    
    # 11. Evaluate
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    eval_metrics = trainer.evaluate()
    
    print(f"\n✓ Training completed!")
    print(f"  - Final train loss: {train_result.training_loss:.4f}")
    print(f"  - Final validation accuracy: {eval_metrics['eval_accuracy']:.4f}")
    print(f"  - Final validation F1-macro: {eval_metrics['eval_f1_macro']:.4f}")
    
    # 12. Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"\n✓ Model saved to: {final_model_path}")
    
    # 13. Save metrics
    metrics_path = output_dir / "final_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Training Loss: {train_result.training_loss:.4f}\n")
        f.write(f"Validation Accuracy: {eval_metrics['eval_accuracy']:.4f}\n")
        f.write(f"Validation F1-Macro: {eval_metrics['eval_f1_macro']:.4f}\n")
    print(f"✓ Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
