#!/usr/bin/env python3
"""
Comprehensive dataset analysis for Chef Classification project.
Analyzes text lengths, distributions, and helps optimize tokenization strategy.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import ast
import json
from collections import Counter

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/train.csv', sep=';')
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

# ============================================================================
# 1. CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("="*80)
print("1. CLASS DISTRIBUTION (Chef Balance)")
print("="*80)
chef_counts = df['chef_id'].value_counts().sort_index()
print(chef_counts)
print(f"\nMost common chef: {chef_counts.max()} samples")
print(f"Least common chef: {chef_counts.min()} samples")
print(f"Imbalance ratio: {chef_counts.max() / chef_counts.min():.2f}x")
print(f"Class balance (std/mean): {chef_counts.std() / chef_counts.mean():.3f}")
print()

# ============================================================================
# 2. MISSING VALUES ANALYSIS
# ============================================================================
print("="*80)
print("2. MISSING VALUES PER FIELD")
print("="*80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing'] > 0])
if missing.sum() == 0:
    print("✓ No missing values found!")
print()

# ============================================================================
# 3. FIELD CONTENT ANALYSIS
# ============================================================================
print("="*80)
print("3. FIELD CONTENT ANALYSIS")
print("="*80)

# Recipe name length
df['recipe_name_len'] = df['recipe_name'].astype(str).apply(len)
print(f"\nRecipe Name:")
print(f"  Mean chars: {df['recipe_name_len'].mean():.1f}")
print(f"  Median chars: {df['recipe_name_len'].median():.1f}")
print(f"  Max chars: {df['recipe_name_len'].max()}")

# Tags analysis
def parse_list_field(field):
    """Safely parse string representations of lists"""
    try:
        if pd.isna(field):
            return []
        return ast.literal_eval(field)
    except:
        return []

df['tags_list'] = df['tags'].apply(parse_list_field)
df['tags_count'] = df['tags_list'].apply(len)
df['tags_text'] = df['tags_list'].apply(lambda x: ', '.join(x))
df['tags_text_len'] = df['tags_text'].apply(len)

print(f"\nTags:")
print(f"  Mean tags per recipe: {df['tags_count'].mean():.1f}")
print(f"  Median tags per recipe: {df['tags_count'].median():.1f}")
print(f"  Max tags: {df['tags_count'].max()}")
print(f"  Mean chars (as text): {df['tags_text_len'].mean():.1f}")
print(f"  Median chars (as text): {df['tags_text_len'].median():.1f}")

# Ingredients analysis
df['ingredients_list'] = df['ingredients'].apply(parse_list_field)
df['ingredients_count'] = df['ingredients_list'].apply(len)
df['ingredients_text'] = df['ingredients_list'].apply(lambda x: ', '.join(x))
df['ingredients_text_len'] = df['ingredients_text'].apply(len)

print(f"\nIngredients:")
print(f"  Mean ingredients per recipe: {df['ingredients_count'].mean():.1f}")
print(f"  Median ingredients: {df['ingredients_count'].median():.1f}")
print(f"  Max ingredients: {df['ingredients_count'].max()}")
print(f"  Mean chars (as text): {df['ingredients_text_len'].mean():.1f}")
print(f"  Median chars (as text): {df['ingredients_text_len'].median():.1f}")
print(f"  Matches n_ingredients column: {(df['ingredients_count'] == df['n_ingredients']).all()}")

# Description length
df['description_len'] = df['description'].fillna('').astype(str).apply(len)
print(f"\nDescription:")
print(f"  Mean chars: {df['description_len'].mean():.1f}")
print(f"  Median chars: {df['description_len'].median():.1f}")
print(f"  Max chars: {df['description_len'].max()}")
print(f"  Empty descriptions: {(df['description_len'] == 0).sum()} ({(df['description_len'] == 0).sum()/len(df)*100:.1f}%)")

# Steps analysis
df['steps_list'] = df['steps'].apply(parse_list_field)
df['steps_count'] = df['steps_list'].apply(len)
df['steps_text'] = df['steps_list'].apply(lambda x: ' | '.join(x))
df['steps_text_len'] = df['steps_text'].apply(len)

print(f"\nSteps:")
print(f"  Mean steps per recipe: {df['steps_count'].mean():.1f}")
print(f"  Median steps: {df['steps_count'].median():.1f}")
print(f"  Max steps: {df['steps_count'].max()}")
print(f"  Mean chars (as text): {df['steps_text_len'].mean():.1f}")
print(f"  Median chars (as text): {df['steps_text_len'].median():.1f}")
print(f"  Max chars: {df['steps_text_len'].max()}")

print()

# ============================================================================
# 4. TOKENIZATION ANALYSIS WITH DISTILBERT
# ============================================================================
print("="*80)
print("4. TOKENIZATION ANALYSIS (DistilBERT)")
print("="*80)
print("Loading DistilBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Build concatenated text in our proposed order
def build_text_input(row):
    """Build text input in our chosen order"""
    parts = []
    
    # Recipe name
    parts.append(str(row['recipe_name']))
    
    # Ingredients
    parts.append(f"Ingredients: {row['ingredients_text']}")
    
    # Tags
    parts.append(f"Tags: {row['tags_text']}")
    
    # Description
    if pd.notna(row['description']) and len(str(row['description'])) > 0:
        parts.append(f"Description: {row['description']}")
    
    # Steps
    parts.append(f"Steps: {row['steps_text']}")
    
    return '\n'.join(parts)

print("Building concatenated text inputs...")
df['full_text'] = df.apply(build_text_input, axis=1)
df['full_text_len'] = df['full_text'].apply(len)

print(f"\nFull Text (chars):")
print(f"  Mean chars: {df['full_text_len'].mean():.1f}")
print(f"  Median chars: {df['full_text_len'].median():.1f}")
print(f"  75th percentile: {df['full_text_len'].quantile(0.75):.1f}")
print(f"  90th percentile: {df['full_text_len'].quantile(0.90):.1f}")
print(f"  95th percentile: {df['full_text_len'].quantile(0.95):.1f}")
print(f"  Max chars: {df['full_text_len'].max()}")

# Tokenize a sample to get token counts
print("\nTokenizing samples (this may take a minute)...")
sample_size = min(500, len(df))
sample_indices = np.random.choice(len(df), sample_size, replace=False)
token_lengths = []

for idx in sample_indices:
    tokens = tokenizer.encode(df.iloc[idx]['full_text'], add_special_tokens=True)
    token_lengths.append(len(tokens))

token_lengths = np.array(token_lengths)

print(f"\nToken Counts (sample of {sample_size}):")
print(f"  Mean tokens: {token_lengths.mean():.1f}")
print(f"  Median tokens: {np.median(token_lengths):.1f}")
print(f"  75th percentile: {np.percentile(token_lengths, 75):.1f}")
print(f"  90th percentile: {np.percentile(token_lengths, 90):.1f}")
print(f"  95th percentile: {np.percentile(token_lengths, 95):.1f}")
print(f"  99th percentile: {np.percentile(token_lengths, 99):.1f}")
print(f"  Max tokens: {token_lengths.max()}")

# Truncation analysis
for max_len in [256, 384, 512]:
    pct_fit = (token_lengths <= max_len).sum() / len(token_lengths) * 100
    print(f"  % samples that fit in {max_len} tokens: {pct_fit:.1f}%")

print()

# ============================================================================
# 5. FIELD-BY-FIELD TOKEN ANALYSIS
# ============================================================================
print("="*80)
print("5. FIELD-BY-FIELD TOKEN CONTRIBUTION")
print("="*80)

# Tokenize each field separately for a sample
field_token_stats = {}

for field_name, field_col in [
    ('recipe_name', 'recipe_name'),
    ('ingredients', 'ingredients_text'),
    ('tags', 'tags_text'),
    ('description', 'description'),
    ('steps', 'steps_text')
]:
    field_tokens = []
    for idx in sample_indices:
        text = str(df.iloc[idx][field_col]) if pd.notna(df.iloc[idx][field_col]) else ""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        field_tokens.append(len(tokens))
    
    field_tokens = np.array(field_tokens)
    field_token_stats[field_name] = {
        'mean': field_tokens.mean(),
        'median': np.median(field_tokens),
        'max': field_tokens.max(),
        'p95': np.percentile(field_tokens, 95)
    }

print("\nAverage token contribution per field:")
for field, stats in field_token_stats.items():
    print(f"\n{field}:")
    print(f"  Mean: {stats['mean']:.1f} tokens")
    print(f"  Median: {stats['median']:.1f} tokens")
    print(f"  95th percentile: {stats['p95']:.1f} tokens")
    print(f"  Max: {stats['max']} tokens")

# Calculate typical budget allocation
total_mean = sum(s['mean'] for s in field_token_stats.values())
print(f"\n\nTypical token budget breakdown (mean):")
for field, stats in field_token_stats.items():
    pct = (stats['mean'] / total_mean) * 100
    print(f"  {field}: {stats['mean']:.1f} tokens ({pct:.1f}%)")

print()

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("="*80)
print("6. RECOMMENDATIONS FOR PIPELINE")
print("="*80)

print("\n📊 KEY FINDINGS:\n")

print("1. CLASS BALANCE:")
if chef_counts.std() / chef_counts.mean() < 0.2:
    print("   ✓ Classes are well balanced - no special weighting needed")
else:
    print(f"   ⚠ Moderate imbalance detected - consider class weights or stratified sampling")
print(f"   Chef distribution: {dict(chef_counts)}")

print("\n2. MISSING DATA:")
if missing.sum() == 0:
    print("   ✓ No missing values - clean dataset!")
else:
    print(f"   ⚠ Some fields have missing values - handle in preprocessing")

print("\n3. MAX_LENGTH RECOMMENDATION:")
pct_512 = (token_lengths <= 512).sum() / len(token_lengths) * 100
pct_384 = (token_lengths <= 384).sum() / len(token_lengths) * 100
if pct_512 >= 90:
    print(f"   ✓ RECOMMENDED: max_length=512 ({pct_512:.1f}% of samples fit)")
elif pct_384 >= 85:
    print(f"   ⚠ RECOMMENDED: max_length=384 ({pct_384:.1f}% fit) or 512 with aggressive truncation")
else:
    print(f"   ⚠ Many long samples - consider max_length=512 with smart truncation")

print("\n4. TRUNCATION STRATEGY:")
steps_pct = (field_token_stats['steps']['mean'] / total_mean) * 100
if steps_pct > 40:
    print(f"   ✓ Steps use {steps_pct:.0f}% of tokens - truncate from TAIL (truncation='only_second')")
    print("   ✓ This protects recipe_name, ingredients, tags, description")
else:
    print(f"   ✓ Use standard truncation='longest_first' (steps use {steps_pct:.0f}% of tokens)")

print("\n5. PADDING STRATEGY:")
median_tokens = np.median(token_lengths)
if median_tokens < 256:
    print(f"   ✓ RECOMMENDED: Dynamic padding per batch (median={median_tokens:.0f} tokens)")
    print("   ✓ Set padding='longest' in tokenizer to save memory")
else:
    print(f"   ⚠ Consider padding='max_length' (median={median_tokens:.0f} tokens)")

print("\n6. BATCH SIZE:")
if median_tokens <= 300:
    print("   ✓ Can use batch_size=16 on most Macs (M1/M2)")
elif median_tokens <= 400:
    print("   ✓ Use batch_size=8-12 on Macs")
else:
    print("   ⚠ Start with batch_size=8, may need to reduce to 4")

print("\n7. FIELD ORDERING:")
print("   ✓ Current order is OPTIMAL:")
print("     1. recipe_name (short, highly informative)")
print("     2. ingredients (moderate length, chef-specific)")
print("     3. tags (moderate, but can be truncated)")
print("     4. description (optional, variable length)")
print("     5. steps (longest, truncate this first)")

print("\n8. DATA INSIGHTS:")
print(f"   • Average recipe has {df['ingredients_count'].mean():.1f} ingredients")
print(f"   • Average recipe has {df['steps_count'].mean():.1f} steps")
print(f"   • Average recipe has {df['tags_count'].mean():.1f} tags")
print(f"   • {(df['description_len'] > 0).sum()} / {len(df)} recipes have descriptions")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save detailed stats to JSON
output_stats = {
    'total_samples': len(df),
    'class_distribution': chef_counts.to_dict(),
    'imbalance_ratio': float(chef_counts.max() / chef_counts.min()),
    'token_stats': {
        'mean': float(token_lengths.mean()),
        'median': float(np.median(token_lengths)),
        'p75': float(np.percentile(token_lengths, 75)),
        'p90': float(np.percentile(token_lengths, 90)),
        'p95': float(np.percentile(token_lengths, 95)),
        'max': int(token_lengths.max())
    },
    'field_token_contribution': {
        field: {k: float(v) if isinstance(v, np.floating) else int(v) 
                for k, v in stats.items()}
        for field, stats in field_token_stats.items()
    },
    'recommendations': {
        'max_length': 512 if pct_512 >= 90 else 384,
        'padding': 'longest',
        'truncation': 'only_second' if steps_pct > 40 else 'longest_first',
        'batch_size': 16 if median_tokens <= 300 else 8
    }
}

with open('data/dataset_analysis.json', 'w') as f:
    json.dump(output_stats, f, indent=2)

print(f"\n✓ Detailed statistics saved to: data/dataset_analysis.json")
