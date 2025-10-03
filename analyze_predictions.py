#!/usr/bin/env python3
"""Analyze predictions by looking at actual recipe content"""
import pandas as pd

# Load test data and predictions
test_df = pd.read_csv("data/test-no-labels.csv", sep=";")
with open("results.txt") as f:
    predictions = [line.strip() for line in f]

test_df["predicted_chef"] = predictions

# Show some examples for each predicted chef
print("=" * 80)
print("PREDICTION ANALYSIS - Looking at actual recipes!")
print("=" * 80)

for chef in sorted(test_df["predicted_chef"].unique()):
    chef_samples = test_df[test_df["predicted_chef"] == chef].head(3)
    print(f"\n{'='*80}")
    print(f"CHEF {chef} - Sample Predictions ({len(test_df[test_df['predicted_chef'] == chef])} total)")
    print(f"{'='*80}")
    
    for idx, row in chef_samples.iterrows():
        print(f"\nRecipe: {row['recipe_name']}")
        print(f"Description: {row['description'][:200]}...")
        if isinstance(row['tags'], str):
            print(f"Tags: {row['tags'][:100]}")
        print(f"Ingredients (first 3): {str(row['ingredients'])[:150]}...")
        print("-" * 80)

# Distribution analysis
print(f"\n{'='*80}")
print("PREDICTION DISTRIBUTION")
print(f"{'='*80}")
dist = test_df["predicted_chef"].value_counts().sort_index()
for chef, count in dist.items():
    pct = 100 * count / len(test_df)
    print(f"Chef {chef}: {count:3d} recipes ({pct:5.1f}%)")
print(f"\nTotal: {len(test_df)} recipes")

# Compare with training distribution
print(f"\n{'='*80}")
print("TRAINING SET DISTRIBUTION (for comparison)")
print(f"{'='*80}")
train_df = pd.read_csv("data/train.csv", sep=";")
train_dist = train_df["chef_id"].value_counts().sort_index()
for chef, count in train_dist.items():
    pct = 100 * count / len(train_df)
    print(f"Chef {chef}: {count:3d} recipes ({pct:5.1f}%)")
print(f"\nTotal: {len(train_df)} recipes")
