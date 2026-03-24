
#!/usr/bin/env python3
"""
score_aggregator.py
===================
Parses all zero-shot result JSONs from the results/ directory,
builds a summary TSV, and produces seaborn visualisations.

Usage:
    python score_aggregator.py
    python score_aggregator.py --results-dir results/ --output-dir plots/
"""

import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use a non-interactive backend so plots save without a display window
matplotlib.use("Agg")


# ----------------------------------------------------------------------------─
# RESULT LOADING
# ----------------------------------------------------------------------------─

def load_results(results_dir: Path) -> pd.DataFrame:
    """Walk results/ and parse every *_zeroshot.json file.

    Returns a DataFrame with one row per (model, subtask) combination,
    containing all available metrics.
    """
    rows = []

    for path in sorted(results_dir.glob("*_zeroshot.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        model   = data.get("model", path.stem)
        subtask = data.get("subtask", "?")
        agg     = data.get("aggregate_metrics", {})

        row = {"model": model, "subtask": subtask, "file": path.name}

        # -- Subtask 1A metrics ----------------------------------------------
        if subtask == "1A":
            row["macro_f1"]    = agg.get("macro_f1", None)
            per_class          = agg.get("per_class", {})
            row["f1_ayah"]    = per_class.get("ayah",    None)
            row["f1_hadith"]  = per_class.get("hadith",  None)
            row["f1_neither"] = per_class.get("neither", None)
            row["n_samples"]   = agg.get("n_samples", None)
            row["n_failed"]    = agg.get("n_failed",  None)

        # -- Subtask 1B metrics ----------------------------------------------
        elif subtask == "1B":
            row["accuracy"]         = agg.get("accuracy",        None)
            row["accuracy_ayah"]    = agg.get("accuracy_ayah",   None)
            row["accuracy_hadith"]  = agg.get("accuracy_hadith", None)
            row["n_total"]          = agg.get("n_total",  None)
            row["n_ayah"]           = agg.get("n_ayah",   None)
            row["n_hadith"]         = agg.get("n_hadith", None)

        # -- Subtask 1C metrics ----------------------------------------------
        elif subtask == "1C":
            row["accuracy"]         = agg.get("accuracy",        None)
            row["accuracy_ayah"]    = agg.get("accuracy_ayah",   None)
            row["accuracy_hadith"]  = agg.get("accuracy_hadith", None)
            row["n_total"]          = agg.get("n_total",  None)

        rows.append(row)

    if not rows:
        raise ValueError(f"No *_zeroshot.json files found in {results_dir}")

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------─
# TSV SUMMARY
# ----------------------------------------------------------------------------─

def build_summary_tsv(df: pd.DataFrame, output_dir: Path) -> None:
    """Save a human-readable TSV with all metrics per model and subtask."""

    out_path = output_dir / "results_summary.tsv"
    sort_col = "macro_f1" if "macro_f1" in df.columns else "accuracy" if "accuracy" in df.columns else None
    if sort_col:
        df_sorted = df.sort_values(["subtask", sort_col], ascending=[True, False])
    else:
        df_sorted = df.sort_values("subtask")
    df_sorted.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
    print(f"Saved TSV -> {out_path}")

    # Also print a compact console table
    print("\n=== RESULTS SUMMARY ===")
    for subtask, grp in df_sorted.groupby("subtask"):
        print(f"\n-- Subtask {subtask} --")
        if subtask == "1A":
            cols = ["model", "macro_f1", "f1_ayah", "f1_hadith", "f1_neither", "n_failed"]
        else:
            cols = ["model", "accuracy", "accuracy_ayah", "accuracy_hadith"]
        available = [c for c in cols if c in grp.columns]
        print(grp[available].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ----------------------------------------------------------------------------─
# PLOTS
# ----------------------------------------------------------------------------─

# Colour palette — one colour per model, consistent across all plots
PALETTE = "tab10"


def _model_order(df_subtask: pd.DataFrame, metric_col: str) -> list:
    """Return model names sorted descending by a metric (for consistent axes)."""
    return (
        df_subtask.dropna(subset=[metric_col])
        .sort_values(metric_col, ascending=False)["model"]
        .tolist()
    )


def plot_1a(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: Subtask 1A macro F1 per model + stacked per-class breakdown."""

    df1a = df[df["subtask"] == "1A"].copy()
    if df1a.empty:
        print("No 1A results to plot.")
        return

    order = _model_order(df1a, "macro_f1")

    # -- Plot 1: Macro F1 bar chart ------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df1a, x="model", y="macro_f1",
        order=order, palette=PALETTE, ax=ax
    )
    ax.set_title("Subtask 1A — Macro F1 (character-level)", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    # Annotate bars with value
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
            ha="center", va="bottom", fontsize=8
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = output_dir / "1A_macro_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")

    # -- Plot 2: Per-class F1 grouped bar chart ------------------------------
    # Melt the per-class columns into long form for seaborn
    id_cols = ["model"]
    value_cols = ["f1_ayah", "f1_hadith", "f1_neither"]
    available = [c for c in value_cols if c in df1a.columns]

    df_melt = df1a[id_cols + available].melt(
        id_vars="model", var_name="class", value_name="f1"
    )
    # Nicer class labels
    df_melt["class"] = df_melt["class"].map({
        "f1_ayah":    "Ayah",
        "f1_hadith":  "Hadith",
        "f1_neither": "Neither",
    })

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=df_melt, x="model", y="f1", hue="class",
        order=order, palette="Set2", ax=ax
    )
    ax.set_title("Subtask 1A — Per-Class F1 by Model", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Class")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = output_dir / "1A_perclass_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_1b(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: Subtask 1B accuracy per model, with Ayah/Hadith breakdown."""

    df1b = df[df["subtask"] == "1B"].copy()
    if df1b.empty:
        print("No 1B results to plot.")
        return

    order = _model_order(df1b, "accuracy")

    # -- Overall accuracy ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df1b, x="model", y="accuracy",
        order=order, palette=PALETTE, ax=ax
    )
    ax.set_title("Subtask 1B — Validation Accuracy", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
            ha="center", va="bottom", fontsize=8
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = output_dir / "1B_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")

    # -- Ayah vs Hadith breakdown --------------------------------------------
    breakdown_cols = ["accuracy_ayah", "accuracy_hadith"]
    available = [c for c in breakdown_cols if c in df1b.columns]
    if available:
        df_melt = df1b[["model"] + available].melt(
            id_vars="model", var_name="type", value_name="accuracy"
        )
        df_melt["type"] = df_melt["type"].map({
            "accuracy_ayah":   "Ayah",
            "accuracy_hadith": "Hadith",
        })
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=df_melt, x="model", y="accuracy", hue="type",
            order=order, palette="Set2", ax=ax
        )
        ax.set_title("Subtask 1B — Accuracy by Span Type", fontsize=14)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.legend(title="Type")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = output_dir / "1B_accuracy_by_type.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved -> {out}")


def plot_1c(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: Subtask 1C correction exact-match accuracy per model."""

    df1c = df[df["subtask"] == "1C"].copy()
    if df1c.empty:
        print("No 1C results to plot.")
        return

    order = _model_order(df1c, "accuracy")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df1c, x="model", y="accuracy",
        order=order, palette=PALETTE, ax=ax
    )
    ax.set_title("Subtask 1C — Correction Exact-Match Accuracy", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
            ha="center", va="bottom", fontsize=8
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = output_dir / "1C_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")

    # -- Ayah vs Hadith breakdown --------------------------------------------
    breakdown_cols = ["accuracy_ayah", "accuracy_hadith"]
    available = [c for c in breakdown_cols if c in df1c.columns]
    if available:
        df_melt = df1c[["model"] + available].melt(
            id_vars="model", var_name="type", value_name="accuracy"
        )
        df_melt["type"] = df_melt["type"].map({
            "accuracy_ayah":   "Ayah",
            "accuracy_hadith": "Hadith",
        })
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=df_melt, x="model", y="accuracy", hue="type",
            order=order, palette="Set2", ax=ax
        )
        ax.set_title("Subtask 1C — Correction Accuracy by Span Type", fontsize=14)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.legend(title="Type")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = output_dir / "1C_accuracy_by_type.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved -> {out}")


def plot_overview(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: all models × all available metrics in one overview plot."""

    # Build a wide pivot table: rows=model, cols=metric
    pivot_rows = []

    for _, row in df.iterrows():
        base = {"model": row["model"]}
        if row["subtask"] == "1A":
            base["1A macro F1"]  = row.get("macro_f1")
            base["1A F1 Ayah"]   = row.get("f1_ayah")
            base["1A F1 Hadith"] = row.get("f1_hadith")
        elif row["subtask"] == "1B":
            base["1B Acc"]       = row.get("accuracy")
            base["1B Acc Ayah"]  = row.get("accuracy_ayah")
            base["1B Acc Hadith"]= row.get("accuracy_hadith")
        elif row["subtask"] == "1C":
            base["1C Acc"]       = row.get("accuracy")
            base["1C Acc Ayah"]  = row.get("accuracy_ayah")
            base["1C Acc Hadith"]= row.get("accuracy_hadith")
        pivot_rows.append(base)

    # Merge all rows per model into a single wide row
    wide = pd.DataFrame(pivot_rows).groupby("model").first().reset_index()

    # Sort by 1A macro F1 descending if available
    if "1A macro F1" in wide.columns:
        wide = wide.sort_values("1A macro F1", ascending=False)

    wide = wide.set_index("model")

    # Drop columns that are all NaN
    wide = wide.dropna(axis=1, how="all")

    if wide.empty:
        print("Not enough data for overview heatmap.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(wide.columns) * 1.2), max(5, len(wide) * 0.7)))
    sns.heatmap(
        wide.astype(float), annot=True, fmt=".3f",
        cmap="YlGnBu", vmin=0, vmax=1,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Model Performance Overview — All Subtasks", fontsize=14)
    ax.set_ylabel("Model")
    ax.set_xlabel("Metric")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = output_dir / "overview_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


# ----------------------------------------------------------------------------─
# MAIN
# ----------------------------------------------------------------------------─

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate zero-shot results and generate visualisations."
    )
    parser.add_argument(
        "--results-dir", default="results/",
        help="Directory containing *_zeroshot.json files (default: results/)"
    )
    parser.add_argument(
        "--output-dir", default="plots/",
        help="Directory to save TSV and PNG plots (default: plots/)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results into a single DataFrame
    print(f"Loading results from {results_dir} ...")
    df = load_results(results_dir)
    print(f"Found {len(df)} result files covering subtasks: {sorted(df['subtask'].unique())}")

    # Save TSV summary
    build_summary_tsv(df, output_dir)

    # Generate plots (each function skips gracefully if no data for that subtask)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plot_1a(df, output_dir)
    plot_1b(df, output_dir)
    plot_1c(df, output_dir)
    plot_overview(df, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
