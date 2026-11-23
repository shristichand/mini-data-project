"""
scripts/run_text_pipeline.py

Command-line interface to execute the Text Dataset Pipeline:
- Unicode normalization (NFC/NFKC, controls, ZWJ/ZWNJ, Devanagari nukta)
- Text normalization (HTML stripping, whitespace, quotes, dashes, danda, digits)
- Language profiling & quality scoring
- Quality filtering (attrition logs)
- Tokenization (Word, Sentence, Akshara, or Subword BPE)
- Parquet export and comprehensive audit report generation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from mini_data_project.config import SAMPLE_10K_PATH, SHARD_PATH
from mini_data_project.download import (
    REPO_ID,
    SUBSET,
    download_shard,
    find_shards,
    get_or_create_fineweb2_sample,
    inspect_schema,
)
from mini_data_project.text import BPETokenizer, PipelineConfig, TextDatasetPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text dataset preprocessing pipeline on FineWeb-2 / Devanagari Parquet data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=SAMPLE_10K_PATH,
        help="Path to input Parquet file (default: raw/sample_10k.parquet from download_and_sample.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed/pipeline_clean_tokenized.parquet",
        help="Path to output processed Parquet file.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/text_pipeline_report.json",
        help="Path to save execution and quality audit JSON report.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["word", "sentence", "akshara", "bpe", "none"],
        default="word",
        help="Tokenization strategy to apply (default: word).",
    )
    parser.add_argument(
        "--bpe-vocab-size",
        type=int,
        default=3000,
        help="Vocabulary size for BPE tokenizer if training BPE (default: 3000).",
    )
    parser.add_argument(
        "--save-bpe",
        type=str,
        default="reports/bpe_tokenizer.json",
        help="Path to save trained BPE tokenizer model JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to process for rapid verification.",
    )
    parser.add_argument(
        "--min-deva",
        type=float,
        default=0.50,
        help="Minimum Devanagari character ratio filter (default: 0.50).",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.35,
        help="Minimum composite quality score filter (default: 0.35).",
    )
    parser.add_argument(
        "--min-npi",
        type=float,
        default=0.95,
        help="Minimum FineWeb-2 language classifier confidence filter (default: 0.95).",
    )
    parser.add_argument(
        "--convert-digits",
        action="store_true",
        help="Convert Devanagari digits to ASCII digits (0-9).",
    )
    parser.add_argument(
        "--download-fineweb2",
        action="store_true",
        help="Force download FineWeb-2 shard from Hugging Face via download_and_sample.py if missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("TEXT DATASET PIPELINE: Normalization, Unicode, Tokenization & Profiling")
    print("Dataset: Hugging Face FineWeb-2 ('HuggingFaceFW/fineweb-2' / 'npi_Deva')")
    print("Source:  Generated and ingested via src/download_and_sample.py")
    print("=" * 70)
    print(f"Target input:   {args.input}")
    print(f"Output file:    {args.output}")
    print(f"Report file:    {args.report}")
    print(f"Tokenizer:      {args.tokenizer}")
    print(f"Min Deva ratio: {args.min_deva}")
    print(f"Min Quality:    {args.min_quality}")
    print(f"Min NPI Score:  {args.min_npi}")

    # Connect to FineWeb-2 sample via download_and_sample
    input_path = args.input
    if args.download_fineweb2 or not os.path.exists(input_path):
        print(f"\n[Connector] Ensuring FineWeb-2 sample availability via download_and_sample.py...")
        input_path = get_or_create_fineweb2_sample(preferred_path=args.input)

    # 1. Inspect FineWeb-2 schema using PyArrow without full memory load
    print(f"\n[1/5] Inspecting FineWeb-2 metadata & schema via download_and_sample.inspect_schema...")
    schema, total_rows = inspect_schema(input_path)
    print(f"Verified FineWeb-2 shard at '{input_path}' with {total_rows:,} total rows.")

    # 2. Load data
    print(f"\n[2/5] Loading records from {input_path}...")
    df = pd.read_parquet(input_path)
    if args.limit and args.limit < len(df):
        print(f"Limiting to first {args.limit:,} records as requested.")
        df = df.iloc[: args.limit].copy()
    initial_count = len(df)
    print(f"Loaded {initial_count:,} records.")

    # 2. Setup configuration & tokenizer
    print(f"\n[2/5] Configuring pipeline stages...")
    tokenizer_mode = None if args.tokenizer == "none" else args.tokenizer
    bpe_model_path = None

    if tokenizer_mode == "bpe":
        print(f"Training BPE Tokenizer (target vocab size: {args.bpe_vocab_size})...")
        bpe = BPETokenizer(vocab_size=args.bpe_vocab_size)
        sample_texts = df["text"].dropna().sample(min(1000, len(df)), random_state=42)
        bpe.train(sample_texts, vocab_size=args.bpe_vocab_size, max_corpus_words=10000)
        bpe.save(args.save_bpe)
        bpe_model_path = args.save_bpe
        print(f"BPE Tokenizer trained and saved to {args.save_bpe} (Vocab size: {len(bpe.vocab)}).")

    config = PipelineConfig(
        unicode_form="NFC",
        strip_controls=True,
        clean_invisibles=True,
        sanitize_joiners=True,
        canonical_nukta=True,
        strip_html_tags=True,
        clean_urls=False,
        clean_emails=False,
        normalize_whitespace=True,
        normalize_punct=True,
        ascii_digits=args.convert_digits,
        lowercase=False,
        min_char_len=40,
        max_char_len=100000,
        min_deva_ratio=args.min_deva,
        max_punct_digit_ratio=0.40,
        max_ngram_repetition=0.45,
        min_quality_score=args.min_quality,
        min_npi_score=args.min_npi,
        tokenization_mode=tokenizer_mode,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_model_path=bpe_model_path,
    )

    pipeline = TextDatasetPipeline(config=config)

    # 3. Execute pipeline
    print(f"\n[3/5] Executing text normalization, Unicode cleaning, profiling & tokenization...")
    t0 = time.time()
    processed_df, run_stats = pipeline.process_dataframe(
        df,
        text_column="text",
        output_tokens=True,
        filter_rejected=True,
    )
    elapsed = time.time() - t0
    kept_count = len(processed_df)

    print(f"\n[4/5] Pipeline Execution Summary:")
    print(f"  - Initial Records:  {initial_count:>7,}")
    print(f"  - Kept Records:     {kept_count:>7,} ({run_stats['retention_rate_pct']}%)")
    print(f"  - Dropped Records:  {initial_count - kept_count:>7,}")
    print(f"  - Elapsed Time:     {elapsed:.2f} s ({run_stats['throughput_docs_sec']} docs/sec)")
    print("  - Rejection breakdown:")
    for reason, count in run_stats.get("rejection_breakdown", {}).items():
        print(f"      * {reason:<25}: {count:>5,}")

    if kept_count > 0 and "token_count" in processed_df.columns:
        total_tokens = int(processed_df["token_count"].sum())
        avg_tokens = float(processed_df["token_count"].mean())
        print(f"  - Total Tokens Produced: {total_tokens:,} (Avg {avg_tokens:.1f} tokens/doc)")

    # 4. Save output Parquet
    print(f"\n[5/5] Saving processed dataset and report...")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(args.output, index=False, compression="snappy")
    print(f"  -> Saved Parquet to: {args.output} ({os.path.getsize(args.output) / (1024*1024):.2f} MB)")

    # 5. Save report
    report_dir = Path(args.report).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "input_file": str(args.input),
        "output_file": str(args.output),
        "pipeline_statistics": run_stats,
        "config": {
            "unicode_form": config.unicode_form,
            "min_deva_ratio": config.min_deva_ratio,
            "min_quality_score": config.min_quality_score,
            "tokenizer_mode": config.tokenization_mode,
            "ascii_digits": config.ascii_digits,
        },
        "sample_output": {
            "cleaned_excerpt": processed_df["cleaned_text"].iloc[0][:200] if kept_count > 0 else "",
            "tokens_excerpt": processed_df["tokens"].iloc[0][:15] if kept_count > 0 and "tokens" in processed_df.columns else [],
        },
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved Audit Report to: {args.report}")
    print("\nText Dataset Pipeline completed successfully!")

if __name__ == "__main__":
    main()
