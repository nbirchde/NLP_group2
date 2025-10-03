#!/usr/bin/env python3
"""
Create visualizations of the dataset analysis.
Shows token distributions, class balance, and field contributions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load analysis results
with open('data/dataset_analysis.json', 'r') as f:
    stats = json.load(f)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Chef Classification Dataset Analysis', fontsize=16, fontweight='bold')

# ============================================================================
# 1. Class Distribution
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
chefs = list(stats['class_distribution'].keys())
counts = list(stats['class_distribution'].values())
colors = ['#FF6B6B' if c == max(counts) else '#4ECDC4' if c == min(counts) else '#95E1D3' 
          for c in counts]

bars = ax1.bar(chefs, counts, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Chef ID', fontweight='bold')
ax1.set_ylabel('Number of Recipes', fontweight='bold')
ax1.set_title('Class Distribution (2.17x Imbalance)', fontweight='bold')
ax1.axhline(y=np.mean(counts), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(counts):.0f}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(count)}',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 2. Token Length Distribution
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
token_stats = stats['token_stats']
positions = [1, 2, 3, 4, 5]
values = [token_stats['p75'], token_stats['p90'], token_stats['p95'], 
          token_stats['max'], 512]
labels = ['75th', '90th', '95th', 'Max\n(627)', 'max_len\n(512)']
colors_tokens = ['#A8E6CF', '#FFD3B6', '#FFAAA5', '#FF8B94', '#4ECDC4']

bars = ax2.barh(positions, values, color=colors_tokens, edgecolor='black', linewidth=1.5)
ax2.set_yticks(positions)
ax2.set_yticklabels(labels)
ax2.set_xlabel('Number of Tokens', fontweight='bold')
ax2.set_title('Token Length Distribution', fontweight='bold')
ax2.axvline(x=token_stats['median'], color='blue', linestyle='--', linewidth=2,
            label=f"Median: {token_stats['median']:.0f}")
ax2.axvline(x=token_stats['mean'], color='red', linestyle='--', linewidth=2,
            label=f"Mean: {token_stats['mean']:.0f}")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
             f' {int(val)}',
             ha='left', va='center', fontweight='bold')

# ============================================================================
# 3. Field Token Contribution
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
fields = list(stats['field_token_contribution'].keys())
field_means = [stats['field_token_contribution'][f]['mean'] for f in fields]
field_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

wedges, texts, autotexts = ax3.pie(field_means, labels=fields, autopct='%1.1f%%',
                                     colors=field_colors, startangle=90,
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax3.set_title('Token Budget Breakdown (Mean)', fontweight='bold')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# ============================================================================
# 4. Coverage by max_length
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
max_lengths = [256, 384, 512]
# Calculate coverage (approximation based on percentiles)
# 256 ≈ 43.6%, 384 ≈ 89.6%, 512 ≈ 98.2%
coverage = [43.6, 89.6, 98.2]
colors_cov = ['#FF8B94', '#FFAAA5', '#A8E6CF']

bars = ax4.bar([str(ml) for ml in max_lengths], coverage, 
               color=colors_cov, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('max_length (tokens)', fontweight='bold')
ax4.set_ylabel('% Samples Fit', fontweight='bold')
ax4.set_title('Truncation Coverage Analysis', fontweight='bold')
ax4.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% threshold')
ax4.legend()
ax4.set_ylim([0, 105])
ax4.grid(axis='y', alpha=0.3)

# Add percentage labels
for bar, cov in zip(bars, coverage):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{cov:.1f}%',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 5. Field Token Statistics (Mean vs 95th percentile)
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
x = np.arange(len(fields))
width = 0.35

means = [stats['field_token_contribution'][f]['mean'] for f in fields]
p95s = [stats['field_token_contribution'][f]['p95'] for f in fields]

bars1 = ax5.bar(x - width/2, means, width, label='Mean', 
                color='#4ECDC4', edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x + width/2, p95s, width, label='95th %ile',
                color='#FF6B6B', edgecolor='black', linewidth=1.5)

ax5.set_xlabel('Field', fontweight='bold')
ax5.set_ylabel('Tokens', fontweight='bold')
ax5.set_title('Field Length Variability', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels([f.replace('_', '\n') for f in fields], fontsize=9)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# ============================================================================
# 6. Recommendations Summary (Text Box)
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

recommendations_text = f"""
🎯 KEY RECOMMENDATIONS

Dataset:
  • {stats['total_samples']} samples across 6 chefs
  • Imbalance: {stats['imbalance_ratio']:.2f}x
  • ✅ No missing values

Tokenization:
  • max_length = 512 (98.2% coverage)
  • padding = 'longest' (dynamic)
  • truncation = True

Training:
  • batch_size = 16 (train)
  • eval_batch_size = 32
  • learning_rate = 2e-5
  • epochs = 3-5, early stopping

Data Split:
  • test_size = 0.2
  • stratify = True ⚠️ CRITICAL
  • random_seed = 42

Metrics:
  • Primary: Accuracy
  • Secondary: Macro-F1 ⚠️

Expected Performance:
  • Baseline: 43% (TF-IDF)
  • Target: 65-75% (DistilBERT)
"""

ax6.text(0.1, 0.95, recommendations_text, 
         transform=ax6.transAxes,
         fontsize=10,
         verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Layout and Save
# ============================================================================
plt.tight_layout()
plt.savefig('data/dataset_analysis_viz.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to: data/dataset_analysis_viz.png")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. data/dataset_analysis.json      (machine-readable stats)")
print("  2. data/dataset_analysis_viz.png   (visual summary)")
print("  3. DATASET_ANALYSIS.md             (detailed report)")
print("  4. QUICK_REFERENCE.md              (copy-paste config)")
print("\nReady to implement the training pipeline! 🚀")
