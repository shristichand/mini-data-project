"""
examples/stream_parquet.py

Example: Remote Parquet inspection and streaming using HfFileSystem.
Demonstrates reading schema and batches directly over the network without downloading the full shard.
"""

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


def main():
    fs = HfFileSystem()

    repo_id = "HuggingFaceFW/fineweb-2"
    subset = "npi_Deva"

    parquet_path = f"datasets/{repo_id}/data/{subset}/train/000_00000.parquet"

    print(f"Opening remote Parquet file: {parquet_path}...")

    with fs.open(parquet_path, "rb") as f:
        parquet_file = pq.ParquetFile(f)

        print("\nSchema:")
        print(parquet_file.schema)

        print("\nTotal rows:", parquet_file.metadata.num_rows)

        print("\nReading first batch (5 rows)...")
        for batch in parquet_file.iter_batches(batch_size=5):
            for row in batch.to_pylist():
                print(row)
            break


if __name__ == "__main__":
    main()
