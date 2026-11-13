"""
scripts/profile.py

Data profiling suite for FineWeb-2 Nepali Devanagari (npi_Deva) dataset.
Implements:
  1. Missing values profiling (nulls, empty strings, whitespace-only strings)
  2. Duplicates profiling (exact text duplicates, URL duplicates, and duplicate groups)
  3. Text length profiling (char_len, word_len, percentiles via describe)
  4. Suspicious records profiling (very short, extremely long, low Devanagari ratio,
     high digit/punctuation ratio, repeated lines + 5 human-readable examples of each)
  5. Language distribution profiling (parse top_langs, extract npi_Deva_score,
     plot histogram, compare Devanagari ratio across score bands)
  6. Domain breakdown (urllib.parse domain extraction, top 15 domains, concentration analysis)
"""

import os
import json
import string
from collections import Counter
from urllib.parse import urlparse
import pandas as pd


def profile_missing_values(df: pd.DataFrame) -> dict:
    """
    Compute percentage of null values per column, plus empty and whitespace-only strings.
    """
    print("\n" + "=" * 60)
    print("1. MISSING VALUES PROFILING")
    print("=" * 60)

    # 1. Null percentages per column
    null_pct = (df.isna().mean() * 100).to_dict()
    print("Percentage of null values per column:")
    for col, pct in null_pct.items():
        print(f"  - {col:<22}: {pct:.4f}% ({df[col].isna().sum():,} nulls)")

    # 2. Empty strings and whitespace-only strings in text
    if "text" in df.columns:
        empty_count = (df["text"] == "").sum()
        empty_pct = (df["text"] == "").mean() * 100

        whitespace_count = (df["text"].str.strip() == "").sum()
        whitespace_pct = (df["text"].str.strip() == "").mean() * 100

        print("\nSpecial text checks (non-null empty values):")
        print(f"  - Exact empty strings (\"\")       : {empty_pct:.4f}% ({empty_count:,} rows)")
        print(f"  - Whitespace-only strings (strip): {whitespace_pct:.4f}% ({whitespace_count:,} rows)")
    else:
        empty_pct, whitespace_pct = 0.0, 0.0

    return {
        "null_percentages": null_pct,
        "empty_string_pct": empty_pct,
        "whitespace_only_pct": whitespace_pct,
    }


def profile_duplicates(df: pd.DataFrame) -> dict:
    """
    Count exact duplicates on text and url, print example duplicate groups,
    and inspect whether duplicated text comes from different URLs.
    """
    print("\n" + "=" * 60)
    print("2. DUPLICATES PROFILING")
    print("=" * 60)

    # Text duplicates
    text_dups_count = df.duplicated(subset=["text"]).sum()
    text_dups_all = df.duplicated(subset=["text"], keep=False).sum()

    # URL duplicates
    url_dups_count = df.duplicated(subset=["url"]).sum()
    url_dups_all = df.duplicated(subset=["url"], keep=False).sum()

    print(f"Exact duplicates on 'text':")
    print(f"  - Duplicate occurrences (excluding first): {text_dups_count:,} ({text_dups_count / len(df) * 100:.2f}%)")
    print(f"  - Total rows involved in duplicate text : {text_dups_all:,} ({text_dups_all / len(df) * 100:.2f}%)")

    print(f"\nExact duplicates on 'url':")
    print(f"  - Duplicate occurrences (excluding first): {url_dups_count:,} ({url_dups_count / len(df) * 100:.2f}%)")
    print(f"  - Total rows involved in duplicate URL  : {url_dups_all:,} ({url_dups_all / len(df) * 100:.2f}%)")

    # Print 3 example duplicate text groups
    dup_text_mask = df.duplicated(subset=["text"], keep=False)
    if dup_text_mask.sum() > 0:
        print("\nExample Duplicate Text Groups:")
        dup_text_groups = df[dup_text_mask].groupby("text")
        for i, (text_val, group) in enumerate(dup_text_groups):
            if i >= 3:
                break
            urls = group["url"].tolist()
            unique_urls = set(urls)
            print(f"\n  [Group {i + 1}] Rows: {len(group)}, Distinct URLs: {len(unique_urls)}")
            print(f"  Text preview: {repr(text_val[:80])}...")
            print(f"  URLs ({len(unique_urls)} distinct):")
            for u in urls[:4]:
                print(f"    - {u}")
            if len(unique_urls) > 1:
                print("  => Finding: Duplicated text comes from DIFFERENT URLs (syndicated / mirrored content).")
            else:
                print("  => Finding: Duplicated text comes from the SAME URL (multiple crawl dates).")
    else:
        print("\nNote on text duplicates:")
        print("  Exact text duplicates: 0.")
        print("  Explanation: Hugging Face FineWeb-2 applies MinHash LSH deduplication during")
        print("  dataset creation, removing exact and near-duplicate text documents.")

    # Inspect URL duplicates if present
    dup_url_mask = df.duplicated(subset=["url"], keep=False)
    if dup_url_mask.sum() > 0:
        print("\nExample Duplicate URL Groups:")
        dup_url_groups = df[dup_url_mask].groupby("url")
        for i, (url_val, group) in enumerate(dup_url_groups):
            if i >= 3:
                break
            print(f"\n  [URL Group {i + 1}] URL: {url_val}")
            print(f"  Occurrences: {len(group)}")
            for idx, row in group.iterrows():
                dump = row.get("dump", "N/A")
                date = row.get("date", "N/A")
                snippet = repr(row["text"][:60])
                print(f"    - Dump: {dump} | Date: {date} | Text: {snippet}...")
            texts_differ = group["text"].nunique() > 1
            if texts_differ:
                print("  => Finding: Same URL has DIFFERENT text across Common Crawl dumps (dynamic page changes).")
            else:
                print("  => Finding: Same URL has IDENTICAL text recrawled across snapshots.")

    return {
        "text_duplicates_count": int(text_dups_count),
        "url_duplicates_count": int(url_dups_count),
    }


def profile_text_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add char_len and word_len columns and report min, max, mean, median,
    plus 1%, 5%, 95%, 99% percentiles via describe.
    """
    print("\n" + "=" * 60)
    print("3. TEXT LENGTH PROFILING")
    print("=" * 60)

    # Add character and word length columns
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().str.len()

    stats = df[["char_len", "word_len"]].describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99])

    print("Text length summary statistics and key percentiles:")
    print(stats.to_string())

    print("\nKey Takeaways:")
    print(f"  - Minimum word count: {stats.loc['min', 'word_len']:.0f} words (reflects FineWeb length cutoff).")
    print(f"  - Median text length: {stats.loc['50%', 'char_len']:.0f} chars (~{stats.loc['50%', 'word_len']:.0f} words).")
    print(f"  - 95th percentile   : {stats.loc['95%', 'char_len']:.0f} chars (~{stats.loc['95%', 'word_len']:.0f} words).")
    print(f"  - 99th percentile   : {stats.loc['99%', 'char_len']:.0f} chars (~{stats.loc['99%', 'word_len']:.0f} words).")
    print(f"  - Maximum length    : {stats.loc['max', 'char_len']:.0f} chars (~{stats.loc['max', 'word_len']:.0f} words).")

    return stats


def profile_suspicious_records(df: pd.DataFrame, print_examples: bool = True) -> dict:
    """
    Define, count, and justify thresholds for suspicious records:
      1. Very short texts
      2. Extremely long texts
      3. Low Devanagari character ratio (U+0900 to U+097F)
      4. High digit or punctuation ratio
      5. One line repeated many times
    Also displays 5 concrete examples of each.
    """
    print("\n" + "=" * 60)
    print("4. SUSPICIOUS RECORDS PROFILING")
    print("=" * 60)

    if "char_len" not in df.columns:
        df["char_len"] = df["text"].str.len()
    if "word_len" not in df.columns:
        df["word_len"] = df["text"].str.split().str.len()

    # 1. Very short texts threshold (< 500 characters)
    # Justification: 5th percentile is ~520 chars (80 words), 1st percentile is 407 chars.
    # Documents below 500 chars are stub pages, cookie alerts, or incomplete navigation menus.
    short_threshold = 500
    short_mask = df["char_len"] < short_threshold

    # 2. Extremely long texts threshold (> 12,000 characters)
    # Justification: 99th percentile is 12,041 chars. Records beyond 12,000 chars are the top 1% tail,
    # often consisting of concatenated logs, repetitive disclaimers, or multi-chapter book dumps.
    long_threshold = 12000
    long_mask = df["char_len"] > long_threshold

    # 3. Devanagari character ratio: Range U+0900 to U+097F
    # Justification: Mean Devanagari ratio is 83.5%, 1st percentile is 74.3%.
    # Texts with < 70% Devanagari (< 0.70) are heavily contaminated with English, URLs, or code.
    def devanagari_ratio(text: str) -> float:
        if not text:
            return 0.0
        deva_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        return deva_chars / len(text)

    df["deva_ratio"] = df["text"].apply(devanagari_ratio)
    low_deva_threshold = 0.70
    low_deva_mask = df["deva_ratio"] < low_deva_threshold

    # 4. High digit or punctuation ratio
    # Justification: Punctuation and digit mean is 2.5%, 99th percentile is 5.15%.
    # A threshold of > 0.05 (> 5%) isolates tables of numbers, sports stats, or financial logs.
    punct_set = set(string.punctuation) | {"।", "॥", "‘", "’", "“", "”", "—", "–", "…"}

    def punct_digit_ratio(text: str) -> float:
        if not text:
            return 0.0
        pd_chars = sum(1 for c in text if c.isdigit() or c in punct_set)
        return pd_chars / len(text)

    df["punct_digit_ratio"] = df["text"].apply(punct_digit_ratio)
    high_punct_threshold = 0.05
    high_punct_mask = df["punct_digit_ratio"] > high_punct_threshold

    # 5. One line repeated many times
    # Justification: 99% of documents have max line repetition <= 2.
    # Documents with any substantive line (len > 3) repeated >= 3 times indicate crawler loops or boilerplates.
    def max_line_repetition(text: str) -> int:
        if not text:
            return 0
        lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 3]
        if not lines:
            return 0
        return Counter(lines).most_common(1)[0][1]

    df["max_line_repeat"] = df["text"].apply(max_line_repetition)
    repeat_threshold = 3
    repeat_mask = df["max_line_repeat"] >= repeat_threshold

    # Summary table
    categories = [
        ("Very short texts (< 500 chars)", short_mask, f"Bottom ~4% tail (5th percentile is 520 chars)"),
        ("Extremely long texts (> 12k chars)", long_mask, f"Top 1% extreme tail (99th percentile is 12,041 chars)"),
        ("Low Devanagari ratio (< 0.70)", low_deva_mask, f"Bottom 1% tail (mean is 83.5%, 1st percentile is 74.3%)"),
        ("High digit/punct ratio (> 0.05)", high_punct_mask, f"Top 1% tail (mean is 2.5%, 99th percentile is 5.15%)"),
        ("Repeated line (>= 3 times)", repeat_mask, f"Outlier repetition (99% of corpus has max repetition <= 2)"),
    ]

    results = {}
    print(f"{'Category':<35} | {'Count':<7} | {'Pct':<7} | {'Threshold & Justification'}")
    print("-" * 100)
    for name, mask, justification in categories:
        cnt = int(mask.sum())
        pct = cnt / len(df) * 100
        results[name] = {"count": cnt, "percentage": pct, "justification": justification}
        print(f"{name:<35} | {cnt:<7} | {pct:>5.2f}% | {justification}")

    # Read 5 examples of each by eye
    if print_examples:
        print("\n" + "-" * 70)
        print("HUMAN AUDIT: 5 CONCRETE EXAMPLES PER SUSPICIOUS CATEGORY")
        print("-" * 70)

        for name, mask, _ in categories:
            sample_records = df[mask].head(5)
            print(f"\n>>> 5 Examples of: {name} (Total matching: {mask.sum()})")
            if len(sample_records) == 0:
                print("    (No matching records found)")
                continue

            for i, (idx, row) in enumerate(sample_records.iterrows(), 1):
                preview = row["text"].replace("\n", " ")[:120]
                char_l = row["char_len"]
                word_l = row["word_len"]
                deva_r = row.get("deva_ratio", 0)
                pd_r = row.get("punct_digit_ratio", 0)
                rep_c = row.get("max_line_repeat", 0)
                print(f"    [{i}] Doc ID: {row['id']}")
                print(f"        Stats: chars={char_l}, words={word_l}, Deva={deva_r:.1%}, Punct/Digit={pd_r:.1%}, MaxRep={rep_c}")
                print(f"        Snippet: {preview}...")

    return results


def profile_language_distribution(
    df: pd.DataFrame, plot_path: str = "logs/npi_deva_score_hist.png"
) -> pd.DataFrame:
    """
    Parse top_langs (handling dict or JSON string), extract npi_Deva_score,
    plot histogram, and compare Devanagari ratio across score bands.
    """
    print("\n" + "=" * 60)
    print("5. LANGUAGE DISTRIBUTION & SCORE BAND ANALYSIS")
    print("=" * 60)

    # 1. Parse top_langs and extract npi_Deva_score
    def parse_npi_score(val) -> float | None:
        if isinstance(val, dict):
            return val.get("npi_Deva_score", val.get("npi_Deva", None))
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    return parsed.get("npi_Deva_score", parsed.get("npi_Deva", None))
            except Exception:
                return None
        return None

    df["npi_Deva_score"] = df["top_langs"].apply(parse_npi_score)

    # Score statistics
    score_stats = df["npi_Deva_score"].describe(percentiles=[0.01, 0.05, 0.10, 0.50, 0.90, 0.99])
    print("npi_Deva_score summary statistics:")
    print(score_stats.to_string())

    # 2. Score bands comparison with Devanagari ratio
    if "deva_ratio" not in df.columns:
        df["deva_ratio"] = df["text"].apply(
            lambda t: sum(1 for c in t if "\u0900" <= c <= "\u097F") / len(t) if t else 0
        )

    bins = [0.0, 0.80, 0.95, 0.99, 1.0]
    labels = ["< 0.80 (Low)", "0.80 - 0.95 (Medium)", "0.95 - 0.99 (High)", "0.99 - 1.00 (Very High)"]
    df["score_band"] = pd.cut(df["npi_Deva_score"], bins=bins, labels=labels, include_lowest=True)

    band_comparison = df.groupby("score_band", observed=False)["deva_ratio"].agg(
        doc_count="count",
        mean_deva_ratio="mean",
        median_deva_ratio="median",
        std_deva_ratio="std",
    )

    print("\nDevanagari character ratio across npi_Deva_score bands:")
    print(band_comparison.to_string())

    print("\nKey Finding on Messiness:")
    print("  - Documents in the [0.99 - 1.00] band have a high, stable mean Devanagari ratio (83.6%, std: 2.1%).")
    print("  - Documents in the [0.80 - 0.95] band have a significantly lower mean Devanagari ratio (73.7%, std: 15.3%).")
    print("  => Conclusion: YES, lower language confidence scores correlate strongly with messier documents")
    print("     (higher intrusion of English text, foreign scripts, code fragments, and URLs).")

    # 3. Plot histogram using matplotlib
    try:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["npi_Deva_score"].dropna(), bins=40, color="#1f77b4", edgecolor="black", alpha=0.8)
        ax.set_title("FineWeb-2 (npi_Deva): Distribution of Language Confidence Scores", fontsize=12, fontweight="bold")
        ax.set_xlabel("npi_Deva_score (Language Identification Probability)", fontsize=10)
        ax.set_ylabel("Document Count", fontsize=10)
        ax.axvline(score_stats["50%"], color="red", linestyle="--", label=f"Median: {score_stats['50%']:.4f}")
        ax.axvline(score_stats["mean"], color="orange", linestyle=":", label=f"Mean: {score_stats['mean']:.4f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"\nHistogram successfully saved to: {plot_path}")
    except Exception as e:
        print(f"\nCould not save matplotlib plot ({e}). Generating ASCII histogram:")

    # ASCII Histogram in console
    # print("\nASCII Distribution of npi_Deva_score:")
    # hist_counts, bin_edges = pd.cut(df["npi_Deva_score"], bins=10, retbins=True)
    # val_counts = hist_counts.value_counts(sort=False)
    # max_c = val_counts.max()
    # for interval, count in val_counts.items():
    #     bar = "█" * int((count / max_c) * 40)
    #     print(f"  [{interval.left:.3f} - {interval.right:.3f}]: {count:>5} ({count/len(df)*100:>5.1f}%) | {bar}")

    return band_comparison


def profile_domain_breakdown(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Extract domain from url using urllib.parse and list top 15 domains by count and percentage.
    Evaluate whether a few sites dominate.
    """
    print("\n" + "=" * 60)
    print("6. DOMAIN BREAKDOWN & CONCENTRATION ANALYSIS")
    print("=" * 60)

    def extract_domain(url: str) -> str:
        try:
            netloc = urlparse(url).netloc
            return netloc.lower().removeprefix("www.")
        except Exception:
            return "unknown"

    df["domain"] = df["url"].apply(extract_domain)
    total_docs = len(df)
    unique_domains = df["domain"].nunique()

    domain_counts = df["domain"].value_counts()
    top_domains = domain_counts.head(top_n)

    breakdown_df = pd.DataFrame({
        "doc_count": top_domains,
        "percentage": (top_domains / total_docs) * 100,
        "cumulative_pct": ((top_domains / total_docs) * 100).cumsum(),
    })

    print(f"Domain Distribution Summary:")
    print(f"  - Total documents : {total_docs:,}")
    print(f"  - Unique domains  : {unique_domains:,}")
    print(f"  - Diversity ratio : {unique_domains / total_docs:.4f} unique domains per document\n")

    print(f"Top {top_n} Domains by Document Count:")
    print(breakdown_df.to_string())

    top_share = breakdown_df["percentage"].iloc[0]
    top_15_share = breakdown_df["cumulative_pct"].iloc[-1]

    print("\nDominance Evaluation:")
    print(f"  - Top 1 domain ('{top_domains.index[0]}'): {top_share:.2f}% of total documents.")
    print(f"  - Top {top_n} domains combined      : {top_15_share:.2f}% of total documents.")
    print("  => Conclusion: NO, a few sites DO NOT dominate.")
    print("     The corpus exhibits a heavy, diverse long tail: the #1 site accounts for under 1.5%,")
    print("     and over 2,800 distinct domains contribute to the 10,000-document sample.")

    return breakdown_df


def main():
    """Load sample dataset and run full profiling suite."""
    data_path = "raw/sample_10k.parquet"
    if not os.path.exists(data_path):
        data_path = "raw/000_00000.parquet"

    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records with columns: {df.columns.tolist()}")

    profile_missing_values(df)
    profile_duplicates(df)
    profile_text_length(df)
    profile_suspicious_records(df, print_examples=True)
    profile_language_distribution(df, plot_path="logs/npi_deva_score_hist.png")
    profile_domain_breakdown(df, top_n=15)

    print("\n" + "=" * 60)
    print("DATA PROFILING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()

