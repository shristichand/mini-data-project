"""
src/config.py

Centralized configuration for dataset paths, sampling parameters, and output directories.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Candidate paths for the downloaded dataset shard
_CANDIDATE_SHARDS = [
    PROJECT_ROOT / "raw" / "000_00000.parquet",
    PROJECT_ROOT / "raw" / "data" / "npi_Deva" / "test" / "000_00000.parquet",
    PROJECT_ROOT / "raw" / "sample_10k.parquet",
]

# Set SHARD_PATH in one config place
SHARD_PATH = str(
    next((p for p in _CANDIDATE_SHARDS if p.exists()), PROJECT_ROOT / "raw" / "000_00000.parquet")
)

# Sampling directories
SAMPLING_REPORTS_DIR = str(PROJECT_ROOT / "reports" / "sampling")
RAW_DATA_DIR = str(PROJECT_ROOT / "raw")
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "processed")

