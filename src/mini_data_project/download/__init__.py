"""
mini_data_project.download

Dataset discovery, streaming, schema inspection, and shard downloading.
"""

from mini_data_project.download.download_and_sample import (
    DATASET_PATH,
    RAW_DIR,
    REPO_ID,
    SAMPLE_JSONL_PATH,
    SAMPLE_PARQUET_PATH,
    SUBSET,
    download_shard,
    find_shards,
    get_or_create_fineweb2_sample,
    inspect_schema,
)

__all__ = [
    "download_shard",
    "find_shards",
    "inspect_schema",
    "get_or_create_fineweb2_sample",
    "REPO_ID",
    "SUBSET",
    "DATASET_PATH",
    "RAW_DIR",
    "SAMPLE_PARQUET_PATH",
    "SAMPLE_JSONL_PATH",
]
