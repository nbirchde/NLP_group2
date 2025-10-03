#!/usr/bin/env python3
"""
Prediction Export Script for Test Set

Loads trained DistilBERT model and generates chef predictions for test data.
Outputs results.txt with one chef_id per line.

Usage:
    python predict.py --model-path artifacts/final_model --test-path ../../data/test-no-labels.csv --output ../../results.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data import load_recipes_csv, concat_text_fields
from src.config import TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate chef predictions for test set"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory (e.g., artifacts/final_model)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        required=True,
        help="Path to test CSV file (e.g., ../../data/test-no-labels.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../results.txt",
        help="Output file path for predictions (default: ../../results.txt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../../configs/base.yaml",
        help="Path to training config (for text field order)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: Path):
    """Load trained model and tokenizer from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Move to available device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    print(f"✓ Model loaded ({model.num_labels} classes)")
    print(f"✓ Device: {device}")
    
    return model, tokenizer, device


def prepare_test_data(test_path: Path, config: TrainingConfig) -> pd.DataFrame:
    """Load and prepare test data with text concatenation."""
    print(f"\nLoading test data from: {test_path}")
    
    # Load raw test data (without requiring chef_id)
    test_data = load_recipes_csv(test_path, require_labels=False)
    
    # Apply same text concatenation as training
    test_data.frame["text"] = test_data.frame.apply(
        lambda row: concat_text_fields(row, config.text_fields), axis=1
    )
    
    print(f"✓ Loaded {len(test_data.frame)} test samples")
    
    return test_data.frame


def batch_predict(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[int]:
    """Run batched inference on text samples."""
    all_predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().tolist())
    
    return all_predictions


def main():
    """Main prediction pipeline."""
    args = parse_args()
    
    # Resolve paths
    model_path = Path(args.model_path).resolve()
    test_path = Path(args.test_path).resolve()
    output_path = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    
    # Validate paths
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load config (for text field order)
    print(f"Loading config from: {config_path}")
    config = TrainingConfig.from_yaml(str(config_path))
    print(f"✓ Text fields: {config.text_fields}")
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    
    # Prepare test data
    test_df = prepare_test_data(test_path, config)
    
    # Run predictions
    print(f"\nRunning inference (batch_size={args.batch_size})...")
    predictions = batch_predict(
        model=model,
        tokenizer=tokenizer,
        texts=test_df["text"].tolist(),
        device=device,
        batch_size=args.batch_size,
        max_length=config.max_length,
    )
    
    # Map predictions back to chef IDs
    id2label = model.config.id2label
    chef_predictions = [id2label[pred_id] for pred_id in predictions]
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chef_id in chef_predictions:
            f.write(f"{chef_id}\n")
    
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"✓ Total predictions: {len(chef_predictions)}")
    
    # Show prediction distribution
    from collections import Counter
    pred_counts = Counter(chef_predictions)
    print(f"\nPrediction distribution:")
    for chef_id, count in sorted(pred_counts.items()):
        print(f"  Chef {chef_id}: {count} predictions ({count/len(chef_predictions)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
