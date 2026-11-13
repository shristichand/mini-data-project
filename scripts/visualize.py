"""
scripts/visualize.py

Generates publication-quality charts for the FineWeb-2 Nepali Devanagari data profiling report:
  1. Text length histogram (log scale on x-axis for character and word lengths)
  2. Language score (npi_Deva_score) distribution histogram
  3. Top 15 domains horizontal bar chart
  4. Suspicious record counts by category bar chart

Saves all figures to reports/figures/.
Reuses shared feature enrichment from `src/text_utils.py`.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from text_utils import add_derived_features

OUTPUT_DIR = "reports/figures"
DATA_PATH = "raw/sample_10k.parquet"


def load_and_prepare_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load sample dataset and compute derived profiling metrics using text_utils."""
    if not os.path.exists(path):
        path = "raw/000_00000.parquet"
    print(f"Loading data from '{path}'...")
    df = pd.read_parquet(path)
    df = add_derived_features(df)
    return df


def plot_text_length_log_histogram(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Plot text length histogram with a log x-axis for character and word lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    char_med = df["char_len"].median()
    char_p05 = df["char_len"].quantile(0.05)
    char_p99 = df["char_len"].quantile(0.99)

    word_med = df["word_len"].median()
    word_p05 = df["word_len"].quantile(0.05)
    word_p99 = df["word_len"].quantile(0.99)

    # 1. Character Length
    log_bins_char = np.logspace(np.log10(df["char_len"].min()), np.log10(df["char_len"].max()), 45)
    ax1.hist(df["char_len"], bins=log_bins_char, color="#1f77b4", edgecolor="#0d3b66", alpha=0.85)
    ax1.set_xscale("log")
    ax1.axvline(char_med, color="#d62728", linestyle="--", linewidth=1.8, label=f"Median: {char_med:,.0f} chars")
    ax1.axvline(char_p05, color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"5th percentile: {char_p05:,.0f} chars")
    ax1.axvline(char_p99, color="#9467bd", linestyle=":", linewidth=1.8, label=f"99th percentile: {char_p99:,.0f} chars")
    ax1.set_title("Character Length Distribution (Log x-axis)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Characters (log scale)", fontsize=10)
    ax1.set_ylabel("Document Count", fontsize=10)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)

    # 2. Word Length
    log_bins_word = np.logspace(np.log10(df["word_len"].min()), np.log10(df["word_len"].max()), 45)
    ax2.hist(df["word_len"], bins=log_bins_word, color="#2ca02c", edgecolor="#134611", alpha=0.85)
    ax2.set_xscale("log")
    ax2.axvline(word_med, color="#d62728", linestyle="--", linewidth=1.8, label=f"Median: {word_med:,.0f} words")
    ax2.axvline(word_p05, color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"5th percentile: {word_p05:,.0f} words")
    ax2.axvline(word_p99, color="#9467bd", linestyle=":", linewidth=1.8, label=f"99th percentile: {word_p99:,.0f} words")
    ax2.set_title("Word Length Distribution (Log x-axis)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Words (log scale)", fontsize=10)
    ax2.set_ylabel("Document Count", fontsize=10)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.suptitle("FineWeb-2 (npi_Deva): Text Length Profiles", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "text_length_histogram_log.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_language_score_histogram(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Plot language score distribution with an inset zoom for the lower-score tail."""
    scores = df["npi_Deva_score"].dropna()
    mean_val = scores.mean()
    median_val = scores.median()
    p01_val = scores.quantile(0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Full Range
    ax1.hist(scores, bins=50, color="#3470a3", edgecolor="#1d4263", alpha=0.85)
    ax1.axvline(median_val, color="#d62728", linestyle="--", linewidth=1.8, label=f"Median: {median_val:.4f}")
    ax1.axvline(mean_val, color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"Mean: {mean_val:.4f}")
    ax1.set_title("Full Language Score Range [0.89 - 1.00]", fontsize=12, fontweight="bold")
    ax1.set_xlabel("npi_Deva_score (Confidence Probability)", fontsize=10)
    ax1.set_ylabel("Document Count", fontsize=10)
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(loc="upper left")

    # Panel 2: Lower-score tail zoom [0.88 - 0.99]
    tail_scores = scores[scores < 0.99]
    ax2.hist(tail_scores, bins=35, color="#e26d5c", edgecolor="#872b1d", alpha=0.85)
    ax2.axvline(p01_val, color="#9467bd", linestyle=":", linewidth=1.8, label=f"1st percentile: {p01_val:.4f}")
    ax2.set_title("Detailed View: Lower-Score Tail (< 0.99)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("npi_Deva_score (< 0.99)", fontsize=10)
    ax2.set_ylabel("Document Count", fontsize=10)
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(loc="upper left")

    plt.suptitle("FineWeb-2 (npi_Deva): Language Identification Confidence Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "language_score_histogram.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_top_domains(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Plot horizontal bar chart of the top 15 domains by document count."""
    domain_counts = df["domain"].value_counts().head(15).sort_values(ascending=True)
    total_docs = len(df)
    unique_domains = df["domain"].nunique()
    top_15_total = domain_counts.sum()
    top_15_pct = (top_15_total / total_docs) * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(domain_counts.index, domain_counts.values, color="#488f78", edgecolor="#24493c", height=0.65)

    for bar in bars:
        width = bar.get_width()
        pct = (width / total_docs) * 100
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2, f"{int(width):,} ({pct:.2f}%)",
                va="center", ha="left", fontsize=9, fontweight="bold", color="#222222")

    ax.set_xlim(0, max(domain_counts.values) * 1.25)
    ax.set_title("Top 15 Domains by Document Count in FineWeb-2 (npi_Deva)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Documents", fontsize=10)
    ax.set_ylabel("Domain", fontsize=10)
    ax.grid(axis="x", ls="--", alpha=0.5)

    callout_text = (
        f"Domain Diversity Metrics:\n"
        f"• Total Documents: {total_docs:,}\n"
        f"• Total Unique Domains: {unique_domains:,}\n"
        f"• Top 1 Domain: {domain_counts.values[-1]} docs ({(domain_counts.values[-1]/total_docs)*100:.2f}%)\n"
        f"• Top 15 Combined: {top_15_total:,} docs ({top_15_pct:.2f}%)\n"
        f"• Status: Heavy long-tail, NO single domain dominance"
    )
    ax.text(0.55, 0.15, callout_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8", edgecolor="#b0c4de", alpha=0.95),
            fontsize=9.5, linespacing=1.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "top_domains_barchart.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_suspicious_records(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Plot bar chart of suspicious record categories with threshold justifications."""
    total_docs = len(df)

    categories = [
        ("Very short texts\n(< 500 chars)", (df["char_len"] < 500).sum(), "#e76f51"),
        ("High punctuation / digits\n(> 5% ratio)", (df["punct_digit_ratio"] > 0.05).sum(), "#f4a261"),
        ("Extremely long texts\n(> 12k chars)", (df["char_len"] > 12000).sum(), "#e9c46a"),
        ("Low Devanagari ratio\n(< 70% U+0900..U+097F)", (df["deva_ratio"] < 0.70).sum(), "#2a9d8f"),
        ("Repeated lines\n(>= 3 times)", (df["max_line_repeat"] >= 3).sum(), "#264653"),
    ]

    labels = [cat[0] for cat in categories]
    counts = [cat[1] for cat in categories]
    colors = [cat[2] for cat in categories]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", width=0.55, alpha=0.9)

    for bar, count in zip(bars, counts):
        pct = (count / total_docs) * 100
        height = bar.get_height()
        ax.annotate(f"{count:,}\n({pct:.2f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_title("Suspicious Records by Category (10,000 Sample Audit)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Flagged Documents", fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "suspicious_records_barchart.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_prepare_data()

    print("\nGenerating charts...")
    plot_text_length_log_histogram(df, OUTPUT_DIR)
    plot_language_score_histogram(df, OUTPUT_DIR)
    plot_top_domains(df, OUTPUT_DIR)
    plot_suspicious_records(df, OUTPUT_DIR)
    print("\nAll figures generated successfully in 'reports/figures/'!")


if __name__ == "__main__":
    main()
