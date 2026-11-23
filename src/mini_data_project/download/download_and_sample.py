"""
src/download_and_sample.py

Hugging Face FineWeb-2 ('HuggingFaceFW/fineweb-2') shard discovery,
download, low-memory schema inspection, sampling, and direct connection
to the Text Dataset Pipeline.
"""

import argparse
import json
import os
import shutil
from typing import Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-2"
SUBSET = "npi_Deva"
DATASET_PATH = f"datasets/{REPO_ID}/data/{SUBSET}"
RAW_DIR = "raw"
SAMPLE_PARQUET_PATH = os.path.join(RAW_DIR, "sample_10k.parquet")
SAMPLE_JSONL_PATH = os.path.join(RAW_DIR, "sample_10k.jsonl")


def find_shards() -> str:
    """List shards and check file sizes using HfFileSystem."""
    print("=== Step 1: Finding Shards on Hugging Face FineWeb-2 ===")
    fs = HfFileSystem()

    # Base list under data/npi_Deva
    base_items = fs.ls(DATASET_PATH, detail=True)
    print(f"\nEntries under '{DATASET_PATH}':")
    for item in base_items:
        name = item["name"]
        size_bytes = item.get("size", 0)
        size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0
        item_type = item.get("type", "unknown")
        print(f"  - {name} [{item_type}]: {size_mb:.2f} MB ({size_bytes} bytes)")

    # Test directory files
    test_path = f"{DATASET_PATH}/test"
    test_files = fs.ls(test_path, detail=True)
    print(f"\nFiles under '{test_path}':")
    for item in test_files:
        name = item["name"]
        size_bytes = item.get("size", 0)
        size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0
        print(f"  - {name}: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # Train directory files
    train_path = f"{DATASET_PATH}/train"
    train_files = fs.ls(train_path, detail=True)
    print(f"\nFiles under '{train_path}':")
    for item in train_files:
        name = item["name"]
        size_bytes = item.get("size", 0)
        size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0
        print(f"  - {name}: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # Check test file condition
    is_single_parquet = (
        len(test_files) == 1 and test_files[0]["name"].endswith(".parquet")
    )
    print(f"\nCheck: Is the test file a single Parquet file? {'Yes' if is_single_parquet else 'No'}")
    if is_single_parquet:
        print(f"  Test file: {test_files[0]['name']} ({test_files[0].get('size', 0) / (1024 * 1024):.2f} MB)")

    return test_files[0]["name"]


def download_shard(file_path: str = "data/npi_Deva/test/000_00000.parquet") -> str:
    """Download a shard using hf_hub_download and save under raw/."""
    print("\n=== Step 2: Downloading FineWeb-2 Shard ===")
    os.makedirs(RAW_DIR, exist_ok=True)

    # hf_hub_download expects filename relative to repository root
    hf_filename = file_path.removeprefix(f"datasets/{REPO_ID}/")
    print(f"Downloading '{hf_filename}' from '{REPO_ID}' (repo_type='dataset') into '{RAW_DIR}'...")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=hf_filename,
        repo_type="dataset",
        local_dir=RAW_DIR,
    )
    size_bytes = os.path.getsize(downloaded_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Downloaded to: {downloaded_path}")
    print(f"Local file size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

    return downloaded_path


def inspect_schema(file_path: str) -> Tuple[pq.ParquetSchema, int]:
    """Inspect schema and metadata using pyarrow.parquet.ParquetFile without full load."""
    print(f"\n=== Inspecting FineWeb-2 Schema with PyArrow: {file_path} ===")
    parquet_file = pq.ParquetFile(file_path)

    num_rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow

    print(f"Total row count (from metadata): {num_rows:,}")
    print("\nFineWeb-2 Column names and types:")
    for field in schema:
        print(f"  - {field.name}: {field.type}")

    return schema, num_rows


def sample_and_store(parquet_path: str, n_samples: int = 10000) -> str:
    """Read shard with pd.read_parquet, sample rows, and save to parquet and jsonl."""
    print("\n=== Step 3: Sampling and Storing FineWeb-2 Data ===")
    print(f"Reading {parquet_path} into pandas DataFrame...")
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    sample_size = min(n_samples, total_rows)
    print(f"Loaded DataFrame with {total_rows:,} rows.")

    print(f"Sampling {sample_size:,} rows (random_state=42)...")
    sample_df = df.sample(n=sample_size, random_state=42)

    os.makedirs(RAW_DIR, exist_ok=True)
    sample_parquet_path = SAMPLE_PARQUET_PATH
    sample_jsonl_path = SAMPLE_JSONL_PATH

    print(f"Saving sample to '{sample_parquet_path}'...")
    sample_df.to_parquet(sample_parquet_path, index=False)

    print(f"Saving sample to '{sample_jsonl_path}' (ensure_ascii=False)...")
    sample_df.to_json(sample_jsonl_path, orient="records", lines=True, force_ascii=False)

    # Confirm reload
    print("\nReloading saved samples to verify row counts:")
    reloaded_parquet = pd.read_parquet(sample_parquet_path)
    reloaded_jsonl = pd.read_json(sample_jsonl_path, orient="records", lines=True)

    print(f"  - Parquet rows: {len(reloaded_parquet):,} (Expected: {sample_size:,})")
    print(f"  - JSONL rows:   {len(reloaded_jsonl):,} (Expected: {sample_size:,})")

    assert len(reloaded_parquet) == sample_size, f"Parquet row count mismatch: {len(reloaded_parquet)} != {sample_size}"
    assert len(reloaded_jsonl) == sample_size, f"JSONL row count mismatch: {len(reloaded_jsonl)} != {sample_size}"
    print("Verification SUCCESS: Both files match the expected row count!")

    return sample_parquet_path


def get_or_create_fineweb2_sample(
    n_samples: int = 10000,
    preferred_path: Optional[str] = None,
) -> str:
    """
    Ensure the canonical FineWeb-2 sample is available locally.
    1. If sample_10k.parquet (or preferred_path) exists, verify and return it.
    2. If a local shard exists (000_00000.parquet), sample from it.
    3. If neither exists, discover and download the shard from Hugging Face, then sample.
    """
    target = preferred_path or SAMPLE_PARQUET_PATH
    if os.path.exists(target):
        print(f"Found existing FineWeb-2 sample: {target}")
        return target

    # Search for local full shard
    candidate_shards = [
        os.path.join(RAW_DIR, "data", "npi_Deva", "test", "000_00000.parquet"),
        os.path.join(RAW_DIR, "000_00000.parquet"),
    ]
    shard_path = next((p for p in candidate_shards if os.path.exists(p)), None)

    if not shard_path:
        print(f"No local FineWeb-2 shard found. Connecting to Hugging Face Hub '{REPO_ID}'...")
        remote_shard = find_shards()
        shard_path = download_shard(remote_shard)

    inspect_schema(shard_path)
    created_sample = sample_and_store(shard_path, n_samples=n_samples)
    return created_sample


def process_fineweb2_sample_with_pipeline(
    sample_path: Optional[str] = None,
    output_path: str = "processed/fineweb2_clean_tokenized.parquet",
    report_path: str = "reports/fineweb2_pipeline_report.json",
    tokenizer_mode: str = "word",
    bpe_vocab_size: int = 3000,
    min_deva_ratio: float = 0.50,
    min_quality_score: float = 0.35,
    min_npi_score: Optional[float] = 0.95,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Bridge function: takes the FineWeb-2 sample (created by download_and_sample)
    and executes the complete Text Dataset Pipeline:
    - Unicode Normalization & ZWJ/Nukta handling
    - Text sanitization, HTML stripping, punctuation & whitespace cleanup
    - Language characteristics profiling (TTR, script ratios, FineWeb-2 language scores)
    - Quality filtering
    - Tokenization (Word, Sentence, Akshara, or Subword BPE)
    """
    from mini_data_project.text.text_pipeline import PipelineConfig, TextDatasetPipeline
    from mini_data_project.text.tokenizers import BPETokenizer

    # 1. Get or create FineWeb-2 sample
    input_file = get_or_create_fineweb2_sample(preferred_path=sample_path)
    print(f"\n[FineWeb-2 Pipeline] Loading sample from {input_file}...")
    df = pd.read_parquet(input_file)

    if limit and limit < len(df):
        print(f"Limiting to first {limit:,} records...")
        df = df.iloc[:limit].copy()

    initial_count = len(df)
    print(f"Input records: {initial_count:,}")

    # 2. Setup BPE if requested
    bpe_model_path = None
    if tokenizer_mode == "bpe":
        bpe_save = "reports/fineweb2_bpe_tokenizer.json"
        print(f"Training BPE Tokenizer on FineWeb-2 sample (vocab size: {bpe_vocab_size})...")
        bpe = BPETokenizer(vocab_size=bpe_vocab_size)
        sample_texts = df["text"].dropna().sample(min(1000, len(df)), random_state=42)
        bpe.train(sample_texts, vocab_size=bpe_vocab_size, max_corpus_words=10000)
        bpe.save(bpe_save)
        bpe_model_path = bpe_save
        print(f"BPE Tokenizer saved to {bpe_save} (Vocab size: {len(bpe.vocab)}).")

    # 3. Configure Pipeline
    config = PipelineConfig(
        unicode_form="NFC",
        strip_controls=True,
        clean_invisibles=True,
        sanitize_joiners=True,
        canonical_nukta=True,
        strip_html_tags=True,
        normalize_whitespace=True,
        normalize_punct=True,
        min_char_len=40,
        max_char_len=100000,
        min_deva_ratio=min_deva_ratio,
        max_punct_digit_ratio=0.40,
        max_ngram_repetition=0.45,
        min_quality_score=min_quality_score,
        min_npi_score=min_npi_score,
        tokenization_mode=tokenizer_mode,
        bpe_vocab_size=bpe_vocab_size,
        bpe_model_path=bpe_model_path,
    )

    pipeline = TextDatasetPipeline(config=config)
    print(f"\n[FineWeb-2 Pipeline] Executing pipeline with tokenizer='{tokenizer_mode}'...")
    processed_df, report = pipeline.process_dataframe(
        df,
        text_column="text",
        output_tokens=True,
        filter_rejected=True,
    )

    # 4. Save results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    processed_df.to_parquet(output_path, index=False, compression="snappy")
    print(f"  -> Saved cleaned & tokenized FineWeb-2 dataset to: {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")

    full_report = {
        "dataset": REPO_ID,
        "subset": SUBSET,
        "source_sample": input_file,
        "output_file": output_path,
        "pipeline_statistics": report,
        "config": {
            "tokenizer_mode": tokenizer_mode,
            "min_deva_ratio": min_deva_ratio,
            "min_quality_score": min_quality_score,
            "min_npi_score": min_npi_score,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved FineWeb-2 pipeline report to: {report_path}")

    return processed_df, report


def main():
    parser = argparse.ArgumentParser(
        description="FineWeb-2 Nepali Devanagari discovery, download, sampling, and text pipeline integration."
    )
    parser.add_argument("--samples", type=int, default=10000, help="Number of rows to sample from FineWeb-2 shard (default: 10,000).")
    parser.add_argument("--pipeline", action="store_true", help="Execute the text dataset pipeline on the FineWeb-2 sample.")
    parser.add_argument("--tokenizer", type=str, default="word", choices=["word", "sentence", "akshara", "bpe", "none"], help="Tokenizer to apply.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for rapid testing.")
    args = parser.parse_args()

    if args.pipeline:
        process_fineweb2_sample_with_pipeline(
            tokenizer_mode=args.tokenizer,
            limit=args.limit,
        )
    else:
        test_shard = find_shards()
        local_path = download_shard(test_shard)
        inspect_schema(local_path)
        sample_and_store(local_path, n_samples=args.samples)


if __name__ == "__main__":
    main()
