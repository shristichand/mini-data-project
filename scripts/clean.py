
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
from datetime import datetime, timezone

import pandas as pd
from text_utils import add_derived_features


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
):
    """Load, clean, and save the dataset with a cleaning report."""

    # 1. Load dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading dataset...")
    df = pd.read_parquet(input_path)

    initial_count = len(df)
    print(f"Initial records: {initial_count:,}")

    # 2. Calculate derived quality metrics
    print("Calculating derived features...")
    df = add_derived_features(df)

    # 3. Apply quality filters sequentially
    cleaning_steps = []

    print("\nApplying quality filters...")

    # Filter 1: Language confidence
    df, log = apply_filter(
        df,
        "language_confidence_filter",
        "Remove missing or low language confidence scores",
        df["npi_Deva_score"].isna() | (df["npi_Deva_score"] < 0.95),
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 2: Devanagari script purity
    df, log = apply_filter(
        df,
        "script_purity_filter",
        "Remove records with Devanagari ratio below 0.70",
        df["deva_ratio"] < 0.70,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 3: Punctuation and digit noise
    df, log = apply_filter(
        df,
        "tabular_symbol_noise_filter",
        "Remove records with punctuation and digit ratio above 0.05",
        df["punct_digit_ratio"] > 0.05,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 4: Repeated lines
    df, log = apply_filter(
        df,
        "boilerplate_repetition_filter",
        "Remove records with a line repeated three or more times",
        df["max_line_repeat"] >= 3,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 5: Minimum text length
    df, log = apply_filter(
        df,
        "short_stub_filter",
        "Remove records shorter than 500 characters",
        df["char_len"] < 500,
        initial_count,
    )
    cleaning_steps.append(log)

    # Filter 6: Maximum text length
    df, log = apply_filter(
        df,
        "extreme_length_filter",
        "Remove records longer than 25,000 characters",
        df["char_len"] > 25000,
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
        description="Clean FineWeb-2 Nepali Devanagari dataset."
    )

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log", default=DEFAULT_LOG)

    args = parser.parse_args()

    run_cleaning_pipeline(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
