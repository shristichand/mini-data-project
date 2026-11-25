
"""
scripts/clean.py

Data cleaning pipeline for FineWeb-2 Nepali Devanagari (npi_Deva).

Applies six quality filters:
1. Language confidence
2. Devanagari script purity
3. Punctuation and digit noise
4. Repeated lines
5. Minimum text length
6. Maximum text length

Outputs:
- processed/clean.parquet
- processed/cleaning_log.json
"""

import os
import json
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from mini_data_project.config import FilterThresholds
from mini_data_project.text import (
    add_derived_features,
    clean_text_pipeline,
    clean_unicode_pipeline,
)


DEFAULT_INPUT = "raw/sample_10k.parquet"
DEFAULT_OUTPUT = "processed/clean.parquet"
DEFAULT_LOG = "processed/cleaning_log.json"


def apply_filter(df, name, description, drop_mask, initial_count):
    """Apply one quality filter and record its results."""

    removed_count = int(drop_mask.sum())

    # Keep records that do not satisfy the removal condition.
    df = df.loc[~drop_mask].copy()

    remaining_count = len(df)

    percent_removed = (
        round((removed_count / initial_count) * 100, 3)
        if initial_count > 0 else 0
    )

    step_log = {
        "name": name,
        "description": description,
        "records_removed": removed_count,
        "percent_removed": percent_removed,
        "remaining_records": remaining_count,
    }

    print(
        f"{name:<30} "
        f"Removed: {removed_count:>5,} | "
        f"Remaining: {remaining_count:>6,}"
    )

    return df, step_log


def run_cleaning_pipeline(
    input_path=DEFAULT_INPUT,
    output_path=DEFAULT_OUTPUT,
    log_path=DEFAULT_LOG,
    thresholds: Optional[FilterThresholds] = None,
    profile_name: str = "standard",
):
    """Load, normalize, clean, and save the dataset with a cleaning report."""
    if thresholds is None:
        thresholds = (
            FilterThresholds.strict()
            if profile_name == "strict"
            else FilterThresholds.standard()
        )

    # 1. Load dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading dataset...")
    df = pd.read_parquet(input_path)

    initial_count = len(df)
    print(f"Initial records: {initial_count:,}")

    # 2. Normalize text (single source of truth: measure normalized text, not raw uncleaned markup)
    print("Normalizing text (Unicode NFC, joiners, HTML, punctuation)...")
    if "cleaned_text" not in df.columns:
        df["cleaned_text"] = df["text"].apply(
            lambda t: clean_text_pipeline(clean_unicode_pipeline(str(t)))
        )

    # 3. Calculate derived quality metrics on cleaned text
    print("Calculating derived features on cleaned text...")
    df = add_derived_features(df, text_column="cleaned_text")

    # 4. Apply quality filters sequentially using canonical thresholds
    cleaning_steps = []

    print("\nApplying quality filters (canonical thresholds)...")

    # Filter 1: Language confidence
    min_npi = thresholds.min_npi_score or 0.95
    df, log = apply_filter(
        df,
        "language_confidence_filter",
        f"Remove missing or low language confidence scores (< {min_npi:.2f})",
        df["npi_Deva_score"].isna() | (df["npi_Deva_score"] < min_npi),
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 2: Devanagari script purity
    df, log = apply_filter(
        df,
        "script_purity_filter",
        f"Remove records with Devanagari ratio below {thresholds.min_deva_ratio:.2f}",
        df["deva_ratio"] < thresholds.min_deva_ratio,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 3: Punctuation and digit noise
    df, log = apply_filter(
        df,
        "tabular_symbol_noise_filter",
        f"Remove records with punctuation and digit ratio above {thresholds.max_punct_digit_ratio:.2f}",
        df["punct_digit_ratio"] > thresholds.max_punct_digit_ratio,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 4: Repeated lines
    df, log = apply_filter(
        df,
        "boilerplate_repetition_filter",
        f"Remove records with a line repeated {thresholds.max_line_repeat} or more times",
        df["max_line_repeat"] >= thresholds.max_line_repeat,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 5: Minimum text length
    df, log = apply_filter(
        df,
        "short_stub_filter",
        f"Remove records shorter than {thresholds.min_char_len} characters",
        df["char_len"] < thresholds.min_char_len,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 6: Maximum text length
    df, log = apply_filter(
        df,
        "extreme_length_filter",
        f"Remove records longer than {thresholds.max_char_len:,} characters",
        df["char_len"] > thresholds.max_char_len,
        initial_count,
    )
    cleaning_steps.append(log)

    # 4. Calculate final statistics
    final_count = len(df)
    total_removed = initial_count - final_count

    retention_pct = (
        round((final_count / initial_count) * 100, 2)
        if initial_count > 0 else 0
    )

    removal_pct = round(100 - retention_pct, 2)

    # 5. Save cleaned dataset
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    df.to_parquet(output_path, index=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nCleaned dataset saved: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")

    # 6. Create cleaning report
    log_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset": input_path,
        "output_dataset": output_path,
        "initial_record_count": initial_count,
        "final_record_count": final_count,
        "total_records_removed": total_removed,
        "removal_percentage": removal_pct,
        "retention_percentage": retention_pct,
        "thresholds_used": asdict(thresholds),
        "measured_text_target": "cleaned_text (normalized)",
        "cleaning_steps": cleaning_steps,
    }

    # 7. Save cleaning report
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(log_data, file, indent=2, ensure_ascii=False)

    print(f"Cleaning report saved: {log_path}")

    # 8. Print summary
    print("\n" + "=" * 50)
    print("CLEANING SUMMARY")
    print("=" * 50)
    print(f"Initial records : {initial_count:,}")
    print(f"Final records   : {final_count:,}")
    print(f"Records removed : {total_removed:,}")
    print(f"Retention rate  : {retention_pct}%")
    print("=" * 50)

    return log_data


def main():
    parser = argparse.ArgumentParser(
        description="Clean FineWeb-2 Nepali Devanagari dataset using canonical thresholds."
    )

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument(
        "--profile",
        choices=["standard", "strict"],
        default="standard",
        help="Canonical filter threshold profile (default: standard).",
    )
    parser.add_argument("--min-deva", type=float, default=None, help="Override min Devanagari ratio")
    parser.add_argument("--max-noise", type=float, default=None, help="Override max punct/digit noise ratio")
    parser.add_argument("--min-len", type=int, default=None, help="Override min char length")
    parser.add_argument("--max-len", type=int, default=None, help="Override max char length")

    args = parser.parse_args()

    # Base thresholds from canonical profile
    base = FilterThresholds.strict() if args.profile == "strict" else FilterThresholds.standard()
    thresholds = FilterThresholds(
        min_char_len=args.min_len if args.min_len is not None else base.min_char_len,
        max_char_len=args.max_len if args.max_len is not None else base.max_char_len,
        min_deva_ratio=args.min_deva if args.min_deva is not None else base.min_deva_ratio,
        max_punct_digit_ratio=args.max_noise if args.max_noise is not None else base.max_punct_digit_ratio,
        max_line_repeat=base.max_line_repeat,
        min_npi_score=base.min_npi_score,
        max_ngram_repetition=base.max_ngram_repetition,
        min_quality_score=base.min_quality_score,
    )

    run_cleaning_pipeline(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        thresholds=thresholds,
        profile_name=args.profile,
    )


if __name__ == "__main__":
    main()
