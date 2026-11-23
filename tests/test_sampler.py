"""
tests/test_sampler.py

Unit and integration tests for Parquet batch processing, sampling algorithms, and reproducibility.
Executes P9 reproducibility testing across all sampling methods:
  - random_sample_inmemory
  - reservoir_sample
  - stratified_sample (proportional)
  - stratified_equal (equal allocation)
"""

import os
import sys
import json
import unittest
from pathlib import Path

# Add project root and src to sys.path for test discovery
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mini_data_project.config import SHARD_PATH, SAMPLING_REPORTS_DIR
from mini_data_project.sampling import (
    get_parquet_metadata,
    print_metadata,
    iter_batches,
    benchmark_batch_reader,
    random_sample_inmemory,
    reservoir_sample,
    add_strata,
    stratified_sample,
    stratified_equal,
    compute_canonical_data_hash,
    save_sample_with_manifest,
)


class TestSampler(unittest.TestCase):
    """Test suite for Parquet metadata inspection, batch processing, and sampling."""

    def setUp(self):
        self.shard_path = SHARD_PATH
        self.assertTrue(
            os.path.exists(self.shard_path),
            f"Configured SHARD_PATH does not exist: {self.shard_path}"
        )
        # Preload a small stratified subset for fast stratified testing
        self.df_strata = add_strata(
            pd.read_parquet(self.shard_path, columns=["id", "text", "url"]),
            verbose=False,
        )

    # -----------------------------------------------------------------
    # P0 & P1 Tests
    # -----------------------------------------------------------------

    def test_config_shard_path_resolution(self):
        """Verify SHARD_PATH points to an existing file with .parquet extension."""
        self.assertTrue(self.shard_path.endswith(".parquet"))
        self.assertGreater(os.path.getsize(self.shard_path), 0)

    def test_parquet_metadata_inspection(self):
        """Verify metadata inspection retrieves correct row count and column names."""
        num_rows, col_names, metadata = get_parquet_metadata(self.shard_path)
        self.assertGreater(num_rows, 0)
        self.assertEqual(num_rows, 49462)
        self.assertIn("text", col_names)
        self.assertIn("id", col_names)
        self.assertIn("language_score", col_names)
        self.assertEqual(len(col_names), 11)
        self.assertEqual(metadata.num_rows, num_rows)

    def test_iter_batches_yields_record_batches(self):
        """Verify iter_batches generator yields pyarrow.RecordBatch instances."""
        batch_gen = iter_batches(self.shard_path, batch_size=1024)
        first_batch = next(batch_gen)
        self.assertIsInstance(first_batch, pa.RecordBatch)
        self.assertEqual(len(first_batch), 1024)

    def test_iter_batches_column_projection(self):
        """Verify reading a column subset yields batches containing only those columns."""
        requested_cols = ["id", "language_score"]
        for batch in iter_batches(self.shard_path, columns=requested_cols, batch_size=2048):
            self.assertEqual(batch.schema.names, requested_cols)
            self.assertEqual(batch.num_columns, 2)
            break

    def test_iter_batches_total_row_count_assertion(self):
        """Verify summing rows across all streamed batches equals metadata row count."""
        num_rows, _, _ = get_parquet_metadata(self.shard_path)
        total_streamed = sum(len(b) for b in iter_batches(self.shard_path, batch_size=2048))
        self.assertEqual(total_streamed, num_rows)

    def test_iter_batches_across_standard_batch_sizes(self):
        """Test streaming across batch sizes 512, 2048, and 8192."""
        num_rows, _, _ = get_parquet_metadata(self.shard_path)
        for bs in (512, 2048, 8192):
            batches = list(iter_batches(self.shard_path, batch_size=bs))
            total = sum(len(b) for b in batches)
            self.assertEqual(total, num_rows, f"Row count mismatch for batch size {bs}")
            for b in batches[:-1]:
                self.assertEqual(len(b), bs)
            expected_remainder = num_rows % bs if num_rows % bs != 0 else bs
            self.assertEqual(len(batches[-1]), expected_remainder)

    # -----------------------------------------------------------------
    # P9 Reproducibility Tests: At least 6 asserts per requirement
    # Testing: same seed -> identical hash, different seed -> different sample, len == n
    # -----------------------------------------------------------------

    def test_reproducibility_inmemory_sampling(self):
        """P9: In-memory sampling reproducibility (same seed, diff seed, exact n)."""
        n = 100
        cols = ["id", "text"]
        s1 = random_sample_inmemory(self.shard_path, n=n, seed=42, columns=cols)
        s2 = random_sample_inmemory(self.shard_path, n=n, seed=42, columns=cols)
        s3 = random_sample_inmemory(self.shard_path, n=n, seed=99, columns=cols)

        # Assert sample size is exactly n
        self.assertEqual(len(s1), n, "Sample size must equal n exactly")
        self.assertEqual(len(s2), n, "Sample size must equal n exactly")
        self.assertEqual(len(s3), n, "Sample size must equal n exactly")

        # Assert same seed gives identical hashes
        hash1 = compute_canonical_data_hash(s1)
        hash2 = compute_canonical_data_hash(s2)
        self.assertEqual(hash1, hash2, "Identical seed must yield identical data hash")

        # Assert different seed gives different sample
        hash3 = compute_canonical_data_hash(s3)
        self.assertNotEqual(hash1, hash3, "Different seed must yield different data hash")

    def test_reproducibility_reservoir_sampling(self):
        """P9: Reservoir sampling reproducibility (same seed, diff seed, exact n)."""
        n = 100
        cols = ["id", "text"]
        r1 = reservoir_sample(self.shard_path, n=n, seed=42, columns=cols, batch_size=2048)
        r2 = reservoir_sample(self.shard_path, n=n, seed=42, columns=cols, batch_size=2048)
        r3 = reservoir_sample(self.shard_path, n=n, seed=99, columns=cols, batch_size=2048)

        # Assert sample size is exactly n
        self.assertEqual(len(r1), n, "Reservoir sample size must equal n exactly")
        self.assertEqual(len(r2), n, "Reservoir sample size must equal n exactly")
        self.assertEqual(len(r3), n, "Reservoir sample size must equal n exactly")

        # Assert same seed gives identical hashes
        hash1 = compute_canonical_data_hash(r1)
        hash2 = compute_canonical_data_hash(r2)
        self.assertEqual(hash1, hash2, "Reservoir with same seed must yield identical hash")

        # Assert different seed gives different sample
        hash3 = compute_canonical_data_hash(r3)
        self.assertNotEqual(hash1, hash3, "Reservoir with different seed must yield different hash")

    def test_reproducibility_stratified_proportional(self):
        """P9: Proportional stratified sampling reproducibility (same seed, diff seed, exact n)."""
        n = 120
        p1 = stratified_sample(self.df_strata, n=n, seed=42)
        p2 = stratified_sample(self.df_strata, n=n, seed=42)
        p3 = stratified_sample(self.df_strata, n=n, seed=99)

        # Assert sample size is exactly n
        self.assertEqual(len(p1), n, "Proportional stratified sample size must equal n exactly")
        self.assertEqual(len(p2), n, "Proportional stratified sample size must equal n exactly")
        self.assertEqual(len(p3), n, "Proportional stratified sample size must equal n exactly")

        # Assert same seed gives identical hashes
        hash1 = compute_canonical_data_hash(p1)
        hash2 = compute_canonical_data_hash(p2)
        self.assertEqual(hash1, hash2, "Proportional with same seed must yield identical hash")

        # Assert different seed gives different sample
        hash3 = compute_canonical_data_hash(p3)
        self.assertNotEqual(hash1, hash3, "Proportional with different seed must yield different hash")

    def test_reproducibility_stratified_equal(self):
        """P9: Equal-allocation stratified sampling reproducibility (same seed, diff seed, exact n)."""
        n = 120
        e1 = stratified_equal(self.df_strata, n=n, seed=42)
        e2 = stratified_equal(self.df_strata, n=n, seed=42)
        e3 = stratified_equal(self.df_strata, n=n, seed=99)

        # Assert sample size is exactly n
        self.assertEqual(len(e1), n, "Equal stratified sample size must equal n exactly")
        self.assertEqual(len(e2), n, "Equal stratified sample size must equal n exactly")
        self.assertEqual(len(e3), n, "Equal stratified sample size must equal n exactly")

        # Assert same seed gives identical hashes
        hash1 = compute_canonical_data_hash(e1)
        hash2 = compute_canonical_data_hash(e2)
        self.assertEqual(hash1, hash2, "Equal allocation with same seed must yield identical hash")

        # Assert different seed gives different sample
        hash3 = compute_canonical_data_hash(e3)
        self.assertNotEqual(hash1, hash3, "Equal allocation with different seed must yield different hash")

    def test_manifest_metadata_and_hash_integrity(self):
        """P8: Verify companion manifest is generated with valid SHA-256 and parameter fields."""
        tmp_parquet = os.path.join(SAMPLING_REPORTS_DIR, "test_manifest_sample.parquet")
        sample_df = random_sample_inmemory(self.shard_path, n=50, seed=123, columns=["id", "text"])
        _, manifest_path = save_sample_with_manifest(
            sample_df=sample_df,
            output_path=tmp_parquet,
            method="test_random",
            seed=123,
            parameters={"n": 50, "test": True},
            source_path=self.shard_path,
        )

        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["method"], "test_random")
        self.assertEqual(manifest["seed"], 123)
        self.assertEqual(manifest["sample_row_count"], 50)
        self.assertEqual(manifest["source_row_count"], 49462)
        self.assertEqual(len(manifest["data_hash_sha256"]), 64)
        self.assertEqual(len(manifest["file_bytes_sha256"]), 64)
        self.assertEqual(manifest["data_hash_sha256"], compute_canonical_data_hash(sample_df))

        # Cleanup test artifacts
        if os.path.exists(tmp_parquet):
            os.remove(tmp_parquet)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)


if __name__ == "__main__":
    unittest.main()
