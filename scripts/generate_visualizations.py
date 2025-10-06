#!/usr/bin/env python3
"""Visualization generator using the latest training artifacts.

This script rebuilds all figures used in the paper and slides directly from
project artifacts so that plots always stay in sync with the final pipeline.
It produces the following assets under ``results/figures``:

* ``baseline_comparison.png`` – baselines vs. DistilBERT (with 95% CI)
* ``training_curves.png`` – training loss plus validation accuracy/F1 traces
* ``dataset_overview.png`` – class balance, token lengths, field contributions
* ``metrics_summary.png`` – one-page highlight of final metrics & key facts
* ``model_architecture.png`` – updated pipeline diagram reflecting final setup
* ``distribution_comparison.png`` – train distribution vs. test predictions
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "sans-serif",
})

# Add project root to sys.path so we can reuse project modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig  # noqa: E402
from src.data import (  # noqa: E402
    FIELD_ALIASES,
    load_recipes_csv,
    format_section,
)
from src.dataset import (  # noqa: E402
    add_text_column,
    drop_duplicate_text_rows,
    prepare_dataset,
)
from src.tokenization import load_tokenizer  # noqa: E402


ARTIFACT_DIR = PROJECT_ROOT / "experiments" / "distilbert_text_only" / "artifacts"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "chill_mode.yaml"
FINAL_METRICS_PATH = ARTIFACT_DIR / "final_metrics.txt"
PREDICTIONS_PATH = PROJECT_ROOT / "results.txt"
TRAINER_STATE_PATTERN = "checkpoint-*/trainer_state.json"


# ---------------------------------------------------------------------------
# Helpers to gather artefacts
# ---------------------------------------------------------------------------

def load_final_metrics(metrics_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with metrics_path.open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", maxsplit=1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                lo, hi = value[1:-1].split(",")
                metrics[f"{key}_low"] = float(lo)
                metrics[f"{key}_high"] = float(hi)
            else:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    return metrics


def locate_trainer_state() -> Path:
    candidates = sorted(ARTIFACT_DIR.glob(TRAINER_STATE_PATTERN))
    if not candidates:
        raise FileNotFoundError("No trainer_state.json files found under artifacts/")
    return candidates[-1]


def load_training_logs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return training log dataframe, eval dataframe, and metadata."""
    state_path = locate_trainer_state()
    with state_path.open() as handle:
        state = json.load(handle)

    log_history = state.get("log_history", [])
    train_rows = []
    eval_rows = []
    for entry in log_history:
        step = entry.get("step")
        epoch = entry.get("epoch")
        if step is None or epoch is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_rows.append({
                "step": step,
                "epoch": epoch,
                "train_loss": entry["loss"],
            })
        if "eval_loss" in entry:
            eval_rows.append({
                "step": step,
                "epoch": epoch,
                "eval_loss": entry["eval_loss"],
                "eval_accuracy": entry.get("eval_accuracy"),
                "eval_f1": entry.get("eval_f1_macro"),
            })

    train_df = pd.DataFrame(train_rows).sort_values("step")
    eval_df = pd.DataFrame(eval_rows).sort_values("step")
    meta = {
        "best_step": state.get("best_global_step"),
        "best_metric": state.get("best_metric"),
    }
    return train_df, eval_df, meta


def compute_dataset_assets(config: TrainingConfig) -> dict[str, object]:
    """Load raw data, apply deduplication, and compute statistics."""
    recipe_data = load_recipes_csv(PROJECT_ROOT / "data" / "train.csv")

    # Build text column and deduplicate in the same way as training
    text_df = add_text_column(recipe_data.frame, config.text_fields)
    dedup_df, removed = drop_duplicate_text_rows(text_df)

    artifacts = prepare_dataset(
        data=recipe_data,
        text_fields=config.text_fields,
        label_column=config.label_column,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_df = artifacts.dataset["train"].to_pandas()
    val_df = artifacts.dataset["validation"].to_pandas()

    tokenizer = load_tokenizer(config.model_name)

    # Token length distribution for deduplicated texts
    encoded = tokenizer(
        dedup_df["text"].tolist(),
        padding=False,
        truncation=True,
        max_length=config.max_length,
        return_length=True,
    )
    token_lengths = np.array(encoded["length"], dtype=int)

    # Token contributions per field using caching to avoid re-tokenising duplicates
    cache: dict[tuple[str, str], int] = {}
    field_totals: Counter[str] = Counter()

    def field_to_text(row: pd.Series, field: str) -> str:
        value = row.get(field, "")
        prefix = FIELD_ALIASES.get(field, "")
        if isinstance(value, list):
            return format_section(prefix, value)
        text = str(value).strip() if pd.notna(value) else ""
        return format_section(prefix, [text]) if text else ""

    for _, row in dedup_df.iterrows():
        for field in config.text_fields:
            formatted = field_to_text(row, field)
            if not formatted:
                continue
            key = (field, formatted)
            if key not in cache:
                cache[key] = len(
                    tokenizer(
                        formatted,
                        add_special_tokens=False,
                        padding=False,
                        truncation=True,
                        max_length=config.max_length,
                    )["input_ids"]
                )
            field_totals[field] += cache[key]

    total_tokens = sum(field_totals.values()) or 1
    field_percentages = {
        field: 100 * count / total_tokens for field, count in field_totals.items()
    }

    stats = {
        "dedup_df": dedup_df,
        "train_df": train_df,
        "val_df": val_df,
        "removed_duplicates": removed,
        "token_lengths": token_lengths,
        "field_percentages": field_percentages,
        "tokenizer": tokenizer,
    }
    return stats


def load_predictions(path: Path) -> Counter[str]:
    if not path.exists():
        raise FileNotFoundError("results.txt not found – run predict.py first")
    with path.open() as handle:
        preds = [line.strip() for line in handle if line.strip()]
    return Counter(preds)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def create_baseline_comparison_fig(final_metrics: dict[str, float]) -> plt.Figure:
    weak_baseline = 30.0
    strong_baseline = 43.0
    final_accuracy = final_metrics.get("validation_accuracy", 0.0) * 100
    acc_low = final_metrics.get("accuracy_95%_ci_low", final_accuracy / 100) * 100
    acc_high = final_metrics.get("accuracy_95%_ci_high", final_accuracy / 100) * 100
    ci_half_width = max(final_accuracy - acc_low, acc_high - final_accuracy, 0)

    methods = [
        "Weak Baseline\nTF-IDF (desc)",
        "Strong Baseline\nTF-IDF (all)",
        "DistilBERT\nFinal pipeline",
    ]
    accuracies = [weak_baseline, strong_baseline, final_accuracy]
    colors = ["#e74c3c", "#e67e22", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.85, edgecolor="black")

    # Add CI for DistilBERT
    ax.errorbar(
        x=2,
        y=final_accuracy,
        yerr=ci_half_width,
        fmt="none",
        ecolor="#1e8449",
        elinewidth=2,
        capsize=6,
        capthick=2,
        label="95% CI",
    )

    for idx, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        if idx == 2 and ci_half_width:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height - 7,
                f"CI: [{acc_low:.1f}, {acc_high:.1f}]",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#1e8449",
            )

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance vs. Baselines", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    return fig


def create_training_curves_fig(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    best_step: int | None,
    final_metrics: dict[str, float],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Plot training loss with evaluation loss checkpoints
    axes[0].plot(
        train_df["epoch"],
        train_df["train_loss"],
        color="#3498db",
        linewidth=2,
        label="Train loss",
    )
    axes[0].scatter(
        eval_df["epoch"],
        eval_df["eval_loss"],
        color="#e74c3c",
        s=40,
        label="Validation loss",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs. Validation Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Validation accuracy
    eval_df = eval_df.copy()
    eval_df["eval_accuracy_pct"] = eval_df["eval_accuracy"].astype(float) * 100
    axes[1].plot(
        eval_df["epoch"],
        eval_df["eval_accuracy_pct"],
        marker="o",
        color="#2ecc71",
        linewidth=2,
        label="Validation accuracy",
    )
    if best_step is not None:
        best_row = eval_df.loc[eval_df["step"] == best_step]
        if not best_row.empty:
            axes[1].scatter(
                best_row["epoch"],
                best_row["eval_accuracy_pct"],
                color="#1e8449",
                s=100,
                zorder=5,
                label="Selected checkpoint",
            )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy over Time")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # Validation macro-F1
    eval_df["eval_f1_pct"] = eval_df["eval_f1"].astype(float) * 100
    axes[2].plot(
        eval_df["epoch"],
        eval_df["eval_f1_pct"],
        marker="s",
        color="#5dade2",
        linewidth=2,
        label="Validation macro-F1",
    )
    if best_step is not None:
        best_row = eval_df.loc[eval_df["step"] == best_step]
        if not best_row.empty:
            axes[2].scatter(
                best_row["epoch"],
                best_row["eval_f1_pct"],
                color="#21618c",
                s=100,
                zorder=5,
                label="Selected checkpoint",
            )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1 (%)")
    axes[2].set_title("Validation Macro-F1 over Time")
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle("Training Diagnostics (logging every 25 steps, eval every 100)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def create_dataset_overview_fig(
    stats: dict[str, object],
    config: TrainingConfig,
) -> plt.Figure:
    dedup_df: pd.DataFrame = stats["dedup_df"]  # type: ignore[assignment]
    train_df: pd.DataFrame = stats["train_df"]  # type: ignore[assignment]
    val_df: pd.DataFrame = stats["val_df"]  # type: ignore[assignment]
    token_lengths: np.ndarray = stats["token_lengths"]  # type: ignore[assignment]
    field_percentages: dict[str, float] = stats["field_percentages"]  # type: ignore[assignment]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Class distribution after deduplication
    class_counts = dedup_df["chef_id"].value_counts().sort_index()
    colors = sns.color_palette("viridis", len(class_counts))
    axes[0, 0].bar(class_counts.index.astype(str), class_counts.values, color=colors)
    axes[0, 0].set_ylabel("Recipes")
    axes[0, 0].set_xlabel("Chef ID")
    axes[0, 0].set_title("Class Distribution after Dedup (2.0x imbalance)")
    axes[0, 0].grid(axis="y", alpha=0.25)
    for bar, count in zip(axes[0, 0].patches, class_counts.values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, count + 5, str(int(count)), ha="center")

    # Token length histogram
    axes[0, 1].hist(token_lengths, bins=40, color="#3498db", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title("Token Length Distribution (post-dedup)")
    axes[0, 1].set_xlabel("Tokens per sample")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(np.median(token_lengths), color="#e74c3c", linestyle="--", label=f"Median: {np.median(token_lengths):.0f}")
    axes[0, 1].axvline(np.percentile(token_lengths, 95), color="#f39c12", linestyle="--", label=f"95th %ile: {np.percentile(token_lengths,95):.0f}")
    axes[0, 1].axvline(config.max_length, color="#16a085", linestyle="--", label=f"Max length: {config.max_length}")
    axes[0, 1].legend()

    # Field contributions
    field_labels = [FIELD_ALIASES.get(f, f).replace(" ", "\n") or f for f in config.text_fields]
    contributions = [field_percentages.get(f, 0.0) for f in config.text_fields]
    axes[1, 0].pie(
        contributions,
        labels=field_labels,
        autopct="%1.0f%%",
        startangle=90,
        colors=sns.color_palette("Set2", len(contributions)),
        textprops={"fontsize": 10},
    )
    axes[1, 0].set_title("Average Token Share per Field")

    # Train / validation split sizes
    split_sizes = [len(train_df), len(val_df)]
    split_labels = [
        f"Train\n({len(train_df)} samples)",
        f"Validation\n({len(val_df)} samples)",
    ]
    axes[1, 1].pie(
        split_sizes,
        labels=split_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#3498db", "#e74c3c"],
        textprops={"fontsize": 10},
    )
    axes[1, 1].set_title("Stratified Split after Dedup (80/20)")

    fig.tight_layout()
    return fig


def create_metrics_summary_fig(
    final_metrics: dict[str, float],
    duplicates_removed: int,
    train_counts: pd.Series,
    eval_counts: pd.Series,
    prediction_counts: Counter[str],
) -> plt.Figure:
    final_acc = final_metrics.get("validation_accuracy", 0) * 100
    final_f1 = final_metrics.get("validation_f1-macro", 0) * 100 or final_metrics.get("validation_f1_macro", 0) * 100
    final_f1 = final_f1 if final_f1 else final_metrics.get("validation_f1_macro", 0) * 100
    final_loss = final_metrics.get("training_loss", 0)
    acc_lo = final_metrics.get("accuracy_95%_ci_low", 0) * 100
    acc_hi = final_metrics.get("accuracy_95%_ci_high", 0) * 100
    f1_lo = final_metrics.get("macro-f1_95%_ci_low", 0) * 100
    f1_hi = final_metrics.get("macro-f1_95%_ci_high", 0) * 100

    improvement = final_acc - 43.0

    train_pct = (train_counts / train_counts.sum() * 100).sort_index()
    pred_series = pd.Series(prediction_counts).sort_index()
    pred_pct = pred_series / pred_series.sum() * 100
    diff_pct = (pred_pct - train_pct).reindex(train_pct.index).fillna(0)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.92, "DistilBERT Chef Classification – Final Snapshot", ha="center", fontsize=18, fontweight="bold")

    text_lines = [
        f"Validation accuracy: {final_acc:.2f}% (95% CI: {acc_lo:.2f} – {acc_hi:.2f})",
        f"Macro-F1 score: {final_f1:.2f}% (95% CI: {f1_lo:.2f} – {f1_hi:.2f})",
        f"Training loss: {final_loss:.4f}",
        f"Improvement over strong baseline (43%): +{improvement:.2f} pp",
        f"Duplicates removed before split: {duplicates_removed}",
        f"Validation samples: {eval_counts.sum()} (stratified)",
    ]

    for idx, line in enumerate(text_lines):
        ax.text(0.03, 0.8 - idx * 0.07, f"• {line}", fontsize=12)

    ax.text(0.03, 0.36, "Chef distribution shift (train → test predictions):", fontsize=12, fontweight="bold")
    table_y = 0.3
    ax.table(
        cellText=[
            [chef, f"{train_pct[chef]:5.1f}%", f"{pred_pct.get(chef,0):5.1f}%", f"{diff_pct.get(chef,0):+5.1f} pp"]
            for chef in train_pct.index.astype(str)
        ],
        colLabels=["Chef", "Train %", "Test pred %", "Δ"],
        colLoc="center",
        cellLoc="center",
        loc="upper left",
        bbox=[0.03, table_y, 0.52, 0.22],
    )

    ax.text(
        0.55,
        0.36,
        "Key evaluation choices:\n"
        "– Deduplicated by concatenated text before splitting\n"
        "– Step-wise evaluation every 100 steps with early stopping\n"
        "– GELU classifier head + 1,000 bootstrap resamples for CIs\n"
        "– Chill-mode batch size 8 to stay within Mac GPU limits",
        fontsize=11,
        va="top",
    )

    return fig


def create_distribution_comparison_fig(
    train_counts: pd.Series,
    prediction_counts: Counter[str],
) -> plt.Figure:
    train_pct = (train_counts / train_counts.sum() * 100).sort_index()
    pred_series = pd.Series(prediction_counts).sort_index()
    pred_pct = pred_series / pred_series.sum() * 100

    chefs = train_pct.index.astype(str)
    x = np.arange(len(chefs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, train_pct.values, width, label="Train", color="#3498db")
    ax.bar(x + width / 2, pred_pct.reindex(chefs).fillna(0).values, width, label="Test predictions", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(chefs)
    ax.set_ylabel("Percentage")
    ax.set_title("Class Distribution: Train vs. Test Predictions")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return fig


def create_model_architecture_fig(config: TrainingConfig, removed_duplicates: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def draw_box(center_x, center_y, width, height, text, facecolor="#ffffff", edgecolor="#34495e"):
        from matplotlib.patches import FancyBboxPatch

        box = FancyBboxPatch(
            (center_x - width / 2, center_y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.015",
            linewidth=1.8,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        ax.add_patch(box)
        ax.text(center_x, center_y, text, ha="center", va="center", fontsize=10, fontweight="bold", wrap=True)

    def draw_arrow(xy_from, xy_to, color="#34495e"):
        ax.annotate("", xy=xy_to, xytext=xy_from, arrowprops=dict(arrowstyle="-|>", linewidth=2, color=color))

    draw_box(
        0.12,
        0.75,
        0.2,
        0.27,
        "Raw recipe fields\nname • ingredients • tags\n• description • steps",
        facecolor="#ecf6fd",
    )
    draw_box(
        0.38,
        0.75,
        0.22,
        0.27,
        f"Pre-processing\n• Parse lists, normalise\n• Concatenate fields\n• Remove {removed_duplicates} duplicate texts",
        facecolor="#e8f8f2",
        edgecolor="#16a085",
    )
    draw_box(
        0.64,
        0.75,
        0.22,
        0.27,
        "Tokenizer (DistilBERT)\nmax_length=512, padding='longest'\ntruncation='longest_first'",
        facecolor="#fef5e7",
        edgecolor="#d35400",
    )
    draw_box(
        0.88,
        0.75,
        0.2,
        0.27,
        "DistilBERT encoder\n+ GELU classifier head\n→ 6-way softmax",
        facecolor="#fbeef5",
        edgecolor="#c0392b",
    )

    draw_arrow((0.23, 0.75), (0.27, 0.75), color="#16a085")
    draw_arrow((0.49, 0.75), (0.53, 0.75), color="#d35400")
    draw_arrow((0.75, 0.75), (0.79, 0.75), color="#c0392b")

    draw_box(
        0.2,
        0.3,
        0.24,
        0.25,
        "Dataset split\n2,999 raw → 2,985 unique\nStratified 80/20 (seed=42)",
        facecolor="#edf7fa",
        edgecolor="#2980b9",
    )
    draw_box(
        0.48,
        0.3,
        0.24,
        0.25,
        "Trainer setup\n• Batch size 8/16 (train/eval)\n• AdamW lr=2e-5, wd=0.01\n• Eval every 100 steps (ES patience=2)",
        facecolor="#fff9e6",
        edgecolor="#f39c12",
    )
    draw_box(
        0.76,
        0.3,
        0.24,
        0.25,
        "Evaluation outputs\n• Accuracy & macro-F1\n• 1,000 bootstrap samples\n• Predictions → results.txt",
        facecolor="#f3f8ff",
        edgecolor="#5e6ad2",
    )

    draw_arrow((0.32, 0.3), (0.36, 0.3), color="#2980b9")
    draw_arrow((0.6, 0.3), (0.64, 0.3), color="#f39c12")

    ax.text(0.5, 0.95, "Chef Classification Pipeline (DistilBERT Text-Only)", ha="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.9, "Final training recipe with evaluation safeguards", ha="center", fontsize=11, color="#7f8c8d")

    return fig


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig.from_yaml(CONFIG_PATH)
    final_metrics = load_final_metrics(FINAL_METRICS_PATH)
    train_logs, eval_logs, log_meta = load_training_logs()
    dataset_stats = compute_dataset_assets(config)
    prediction_counts = load_predictions(PREDICTIONS_PATH)

    train_counts = dataset_stats["train_df"]["chef_id"].value_counts().sort_index()
    val_counts = dataset_stats["val_df"]["chef_id"].value_counts().sort_index()

    figures = {
        "baseline_comparison.png": create_baseline_comparison_fig(final_metrics),
        "training_curves.png": create_training_curves_fig(train_logs, eval_logs, log_meta.get("best_step"), final_metrics),
        "dataset_overview.png": create_dataset_overview_fig(dataset_stats, config),
        "metrics_summary.png": create_metrics_summary_fig(final_metrics, dataset_stats["removed_duplicates"], train_counts, val_counts, prediction_counts),
        "distribution_comparison.png": create_distribution_comparison_fig(train_counts + val_counts, prediction_counts),
        "model_architecture.png": create_model_architecture_fig(config, dataset_stats["removed_duplicates"]),
    }

    for filename, fig in figures.items():
        output_path = FIGURE_DIR / filename
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved {output_path.relative_to(PROJECT_ROOT)}")

    print("All visualisations regenerated.")


if __name__ == "__main__":
    main()
