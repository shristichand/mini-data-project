"""
src/mini_data_project/download/download_and_sample.py

Hugging Face FineWeb-2 ('HuggingFaceFW/fineweb-2') shard discovery,
download, low-memory schema inspection, and raw baseline sampling.
"""

import argparse
import json
import os
import shutil
from typing import Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

# Canonical re-export of orchestration function to prevent codebase drift
from mini_data_project.text.text_pipeline import process_fineweb2_sample_with_pipeline

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
    from mini_data_project.sampling.sampler import get_parquet_metadata

    num_rows, col_names, metadata = get_parquet_metadata(file_path)
    parquet_file = pq.ParquetFile(file_path)
    schema = parquet_file.schema_arrow

    print(f"\n=== Inspecting FineWeb-2 Schema with PyArrow: {file_path} ===")
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


def main():
    parser = argparse.ArgumentParser(
        description="FineWeb-2 Nepali Devanagari discovery, download, and baseline sampling."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of rows to sample from FineWeb-2 shard (default: 10,000).",
    )
    args = parser.parse_args()

    test_shard = find_shards()
    local_path = download_shard(test_shard)
    inspect_schema(local_path)
    sample_and_store(local_path, n_samples=args.samples)


if __name__ == "__main__":
    main()
