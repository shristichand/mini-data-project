import os
import shutil
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-2"
SUBSET = "npi_Deva"
DATASET_PATH = f"datasets/{REPO_ID}/data/{SUBSET}"
RAW_DIR = "raw"


def find_shards():
    """List shards and check file sizes using HfFileSystem."""
    print("=== Step 1: Finding Shards ===")
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


def download_shard(file_path: str = "data/npi_Deva/test/000_00000.parquet"):
    """Download a shard using hf_hub_download and save under raw/."""
    print("\n=== Step 2: Downloading Shard ===")
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

    # Also ensure raw/000_00000.parquet exists for convenience
    # direct_path = os.path.join(RAW_DIR, "000_00000.parquet")
    # if not os.path.exists(direct_path):
    #     shutil.copy2(downloaded_path, direct_path)

    return downloaded_path


def inspect_schema(file_path: str):
    """Inspect schema and metadata using pyarrow.parquet.ParquetFile without full load."""
    print("\n=== Step 3: Inspecting Schema with PyArrow ===")
    parquet_file = pq.ParquetFile(file_path)

    num_rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow

    print(f"Total row count (from metadata): {num_rows:,}")
    print("\nColumn names and types:")
    for field in schema:
        print(f"  - {field.name}: {field.type}")

    return schema, num_rows


def sample_and_store(parquet_path: str, n_samples: int = 10000):
    """Read shard with pd.read_parquet, sample rows, and save to parquet and jsonl."""
    print("\n=== Step 4: Sampling and Storing Data ===")
    print(f"Reading {parquet_path} into pandas DataFrame...")
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    sample_size = min(n_samples, total_rows)
    print(f"Loaded DataFrame with {total_rows:,} rows.")

    print(f"Sampling {sample_size:,} rows (random_state=42)...")
    sample_df = df.sample(n=sample_size, random_state=42)

    sample_parquet_path = os.path.join(RAW_DIR, "sample_10k.parquet")
    sample_jsonl_path = os.path.join(RAW_DIR, "sample_10k.jsonl")

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


def main():
    test_shard = find_shards()
    local_path = download_shard(test_shard)
    inspect_schema(local_path)
    sample_and_store(local_path, n_samples=10000)


if __name__ == "__main__":
    main()

