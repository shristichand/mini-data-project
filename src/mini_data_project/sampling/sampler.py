"""
src/mini_data_project/sampler.py

Unified sampling and batch processing module for FineWeb-2 Parquet datasets.
Implements:
  - P0: Parquet metadata inspection (row count, column names)
  - P1: Low-memory batch reader generator with projection pushdown
  - P2: Random in-memory sampling (pd.read_parquet + df.sample)
  - P3: Reservoir sampling over streamed batches (bounded memory: n + 1 batch)
  - P5: Stratum generation (len_bucket quartiles + top 10 domains + "other")
  - P6: Proportional stratified sampling (exact quota allocation)
  - P7: Equal-allocation stratified sampling with tiny-strata redistribution
  - P8: Sample manifest generation with dual SHA-256 (canonical rows + file bytes)
"""

import os
import json
import time
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mini_data_project.config import SHARD_PATH, SAMPLING_REPORTS_DIR
from mini_data_project.text.text_utils import extract_domain


def get_parquet_metadata(path: str = SHARD_PATH) -> tuple[int, list[str], pq.FileMetaData]:
    """Inspect Parquet file footer metadata without reading payload into memory."""
    pf = pq.ParquetFile(path)
    metadata = pf.metadata
    num_rows = metadata.num_rows
    col_names = pf.schema_arrow.names
    return num_rows, col_names, metadata


inspect_parquet_metadata = get_parquet_metadata


def inspect_schema(path: str = SHARD_PATH):
    """Bridge to unified schema inspection."""
    from mini_data_project.download.download_and_sample import inspect_schema as _inspect_schema
    return _inspect_schema(path)


def print_metadata(path: str = SHARD_PATH) -> tuple[int, list[str]]:
    """Print the row count and column names from the Parquet metadata."""
    num_rows, col_names, metadata = get_parquet_metadata(path)
    print("=" * 70)
    print("PARQUET METADATA INSPECTION")
    print("=" * 70)
    print(f"Target Shard : {path}")
    print(f"Total Rows   : {num_rows:,}")
    print(f"Row Groups   : {metadata.num_row_groups:,}")
    print(f"Columns ({len(col_names)}):")
    for idx, col in enumerate(col_names, start=1):
        col_type = metadata.schema.column(idx - 1).physical_type
        print(f"  {idx:>2}. {col:<24} (physical: {col_type})")
    print("=" * 70)
    return num_rows, col_names

def iter_batches(
    path: str = SHARD_PATH,
    columns: Sequence[str] | None = None,
    batch_size: int = 2048,
) -> Generator[pa.RecordBatch, None, None]:
    """
    Generator yielding pyarrow.RecordBatch instances over ParquetFile.iter_batches.
    Reads only specified columns, minimizing memory allocation.
    """
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch


def benchmark_batch_reader(
    path: str = SHARD_PATH,
    columns: Sequence[str] | None = None,
    batch_sizes: Sequence[int] = (512, 2048, 8192),
) -> dict[int, dict]:
    """Evaluate batch reader throughput and assert row counts match metadata."""
    metadata_rows, col_names, _ = get_parquet_metadata(path)
    projected_cols = list(columns) if columns is not None else col_names

    print("\n" + "=" * 70)
    print("BATCH READER BENCHMARK (pyarrow.parquet.ParquetFile.iter_batches)")
    print("=" * 70)
    print(f"Target File     : {path}")
    print(f"Metadata Rows   : {metadata_rows:,}")
    print(f"Projected Cols  : {projected_cols}")
    print("-" * 70)

    results: dict[int, dict] = {}
    for bs in batch_sizes:
        batch_counts: list[int] = []
        t0 = time.perf_counter()

        for batch in iter_batches(path=path, columns=projected_cols, batch_size=bs):
            batch_counts.append(len(batch))

        elapsed = time.perf_counter() - t0
        total_rows = sum(batch_counts)
        num_batches = len(batch_counts)
        first_b = batch_counts[0] if batch_counts else 0
        last_b = batch_counts[-1] if batch_counts else 0
        throughput = total_rows / elapsed if elapsed > 0 else 0.0

        assert total_rows == metadata_rows, (
            f"Row count mismatch for batch size {bs}: {total_rows} != {metadata_rows}"
        )

        results[bs] = {
            "batch_size": bs,
            "num_batches": num_batches,
            "first_batch_rows": first_b,
            "last_batch_rows": last_b,
            "total_rows": total_rows,
            "elapsed_seconds": round(elapsed, 4),
            "rows_per_second": round(throughput, 1),
            "matches_metadata": True,
        }

        print(f"\n[Batch Size: {bs:>5,}]")
        print(f"  • Number of Batches : {num_batches:>4,}")
        print(f"  • Rows Per Batch    : {first_b:,} (first) ... {last_b:,} (final remainder)")
        print(f"  • Total Streamed    : {total_rows:>6,} rows (matches metadata: ✓)")
        print(f"  • Elapsed Time      : {elapsed:>7.4f} seconds ({throughput:>11,.0f} rows/s)")

    return results


def random_sample_inmemory(
    path: str = SHARD_PATH,
    n: int = 1000,
    seed: int = 42,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Read dataset with pd.read_parquet and take a random sample of size n.
    Returns the sampled DataFrame.
    """
    df = pd.read_parquet(path, columns=columns)
    sample_size = min(n, len(df))
    sample_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return sample_df


def reservoir_sample(
    path: str = SHARD_PATH,
    n: int = 1000,
    seed: int = 42,
    columns: Sequence[str] | None = None,
    batch_size: int = 2048,
) -> pd.DataFrame:
    """
    Single-pass reservoir sampling (Algorithm R) over streamed Parquet batches.
    Memory invariant: Never holds more than n rows plus the active batch.
    Uses Python's random.Random(seed).
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total_seen = 0

    for batch in iter_batches(path, columns=columns, batch_size=batch_size):
        # Active batch in memory: at most batch_size records
        records = batch.to_pylist()
        for record in records:
            total_seen += 1
            if len(reservoir) < n:
                reservoir.append(record)
            else:
                j = rng.randint(0, total_seen - 1)
                if j < n:
                    reservoir[j] = record
        # After processing, records is released before fetching the next batch

    sample_df = pd.DataFrame(reservoir)
    return sample_df


def add_strata(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add stratification attributes to DataFrame:
      - len_bucket: quartiles of char_len (Q1_short, Q2_med_low, Q3_med_high, Q4_long)
      - domain_group: top 10 domains plus 'other'
      - stratum: combination of len_bucket and domain_group
    """
    df = df.copy()

    # Ensure char_len exists
    if "char_len" not in df.columns:
        df["char_len"] = df["text"].astype(str).str.len()

    # Quartiles of character length
    df["len_bucket"] = pd.qcut(
        df["char_len"],
        q=4,
        labels=["Q1_short", "Q2_med_low", "Q3_med_high", "Q4_long"],
        duplicates="drop",
    ).astype(str)

    # Domain group: top 10 domains plus 'other'
    if "domain" not in df.columns and "url" in df.columns:
        df["domain"] = df["url"].apply(extract_domain)

    if "domain" in df.columns:
        top_10 = df["domain"].value_counts().head(10).index.tolist()
        df["domain_group"] = df["domain"].apply(lambda d: d if d in top_10 else "other")
    else:
        df["domain_group"] = "other"

    # Composite stratum
    df["stratum"] = df["len_bucket"] + "__" + df["domain_group"]

    if verbose:
        print("\nStratum Summary:")
        counts = df["stratum"].value_counts()
        print(f"  Total Strata Formed : {len(counts)}")
        print(f"  Largest Stratum     : '{counts.index[0]}' ({counts.iloc[0]:,} rows)")
        print(f"  Smallest Stratum    : '{counts.index[-1]}' ({counts.iloc[-1]:,} rows)")

    return df


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
    strata_col: str = "stratum",
) -> pd.DataFrame:
    """
    Proportional stratified sampling using the Largest Remainder Method (Hamilton method).
    Guarantees exact total sample size of n while maintaining population shares.
    """
    if strata_col not in df.columns:
        df = add_strata(df, verbose=False)

    N = len(df)
    if n >= N:
        return df.sample(n=N, random_state=seed)

    strata_groups = {s: group.index.tolist() for s, group in df.groupby(strata_col, observed=True)}
    strata_sizes = {s: len(idx) for s, idx in strata_groups.items()}

    # 1. Exact mathematical quotas
    quotas = {s: (size / N) * n for s, size in strata_sizes.items()}

    # 2. Floor allocations clamped to available stratum size
    allocations = {s: min(strata_sizes[s], int(quotas[s])) for s in strata_sizes}
    current_total = sum(allocations.values())
    deficit = n - current_total

    # 3. Distribute remainder by highest fractional remainder
    if deficit > 0:
        remainders = [
            (quotas[s] - int(quotas[s]), s)
            for s in strata_sizes
            if allocations[s] < strata_sizes[s]
        ]
        remainders.sort(reverse=True, key=lambda x: (x[0], x[1]))
        for _, s in remainders[:deficit]:
            allocations[s] += 1

    # 4. Fallback if deficit remains (all fractional strata filled)
    current_total = sum(allocations.values())
    if current_total < n:
        available = [s for s in strata_sizes if allocations[s] < strata_sizes[s]]
        available.sort(key=lambda s: strata_sizes[s] - allocations[s], reverse=True)
        for s in available:
            needed = n - sum(allocations.values())
            if needed <= 0:
                break
            give = min(needed, strata_sizes[s] - allocations[s])
            allocations[s] += give

    assert sum(allocations.values()) == n, f"Proportional allocation sum {sum(allocations.values())} != {n}"

    # 5. Draw samples per stratum with seeded generator
    sampled_indices = []
    for s in sorted(allocations.keys()):
        k = allocations[s]
        if k > 0:
            stratum_sample = df.loc[strata_groups[s]].sample(n=k, random_state=seed)
            sampled_indices.extend(stratum_sample.index.tolist())

    sample_df = df.loc[sampled_indices].copy().reset_index(drop=True)
    assert len(sample_df) == n, f"Proportional sample size {len(sample_df)} != {n}"
    return sample_df


def stratified_equal(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
    strata_col: str = "stratum",
) -> pd.DataFrame:
    """
    Equal-allocation stratified sampling.
    Allocates an equal target per stratum. For tiny strata with size < quota,
    clamps to stratum size and redistributes remaining deficit equally among active strata.
    Guarantees exact total sample size of n.
    """
    if strata_col not in df.columns:
        df = add_strata(df, verbose=False)

    N = len(df)
    if n >= N:
        return df.sample(n=N, random_state=seed)

    strata_groups = {s: group.index.tolist() for s, group in df.groupby(strata_col, observed=True)}
    strata_sizes = {s: len(idx) for s, idx in strata_groups.items()}

    allocations = {s: 0 for s in strata_sizes}
    active_strata = set(strata_sizes.keys())
    remaining_needed = n

    # Iterative redistribution for tiny strata
    while active_strata and remaining_needed > 0:
        target_per_stratum = remaining_needed // len(active_strata)
        remainder = remaining_needed % len(active_strata)

        tiny_strata = [s for s in active_strata if strata_sizes[s] <= target_per_stratum]
        if tiny_strata:
            for s in tiny_strata:
                allocations[s] = strata_sizes[s]
                remaining_needed -= strata_sizes[s]
                active_strata.remove(s)
        else:
            sorted_active = sorted(active_strata, key=lambda s: strata_sizes[s], reverse=True)
            for i, s in enumerate(sorted_active):
                extra = 1 if i < remainder else 0
                to_give = min(strata_sizes[s], target_per_stratum + extra)
                allocations[s] = to_give
            break

    # Secondary adjustment if remainder remains
    current_total = sum(allocations.values())
    if current_total < n:
        for s in sorted(strata_sizes.keys(), key=lambda s: strata_sizes[s] - allocations[s], reverse=True):
            needed = n - sum(allocations.values())
            if needed <= 0:
                break
            can_take = strata_sizes[s] - allocations[s]
            give = min(needed, can_take)
            allocations[s] += give

    assert sum(allocations.values()) == n, f"Equal allocation sum {sum(allocations.values())} != {n}"

    sampled_indices = []
    for s in sorted(allocations.keys()):
        k = allocations[s]
        if k > 0:
            stratum_sample = df.loc[strata_groups[s]].sample(n=k, random_state=seed)
            sampled_indices.extend(stratum_sample.index.tolist())

    sample_df = df.loc[sampled_indices].copy().reset_index(drop=True)
    assert len(sample_df) == n, f"Equal sample size {len(sample_df)} != {n}"
    return sample_df


def compute_canonical_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 hash over sorted records.
    Immune to Parquet metadata timestamp variations.
    """
    sort_cols = ["id"] if "id" in df.columns else list(df.columns)
    sorted_df = df.sort_values(sort_cols).reset_index(drop=True)

    hasher = hashlib.sha256()
    json_lines = sorted_df.to_json(orient="records", lines=True, force_ascii=False)
    for line in json_lines.splitlines():
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def compute_file_sha256(path: str) -> str:
    """Compute binary SHA-256 of physical file bytes."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_sample_with_manifest(
    sample_df: pd.DataFrame,
    output_path: str,
    method: str,
    seed: int,
    parameters: dict,
    source_path: str = SHARD_PATH,
) -> tuple[str, str]:
    """
    Persist sample DataFrame to Parquet and generate a companion <name>.manifest.json.
    Records method, seed, parameters, source row count, sample count, canonical row hash,
    and physical file SHA-256.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sample_df.to_parquet(output_path, index=False)

    out_p = Path(output_path)
    manifest_path = str(out_p.parent / f"{out_p.stem}.manifest.json")

    data_hash = compute_canonical_data_hash(sample_df)
    file_hash = compute_file_sha256(output_path)
    source_rows = pq.ParquetFile(source_path).metadata.num_rows

    manifest_payload = {
        "method": method,
        "seed": seed,
        "parameters": parameters,
        "source_file": str(source_path),
        "source_row_count": source_rows,
        "sample_row_count": len(sample_df),
        "output_parquet": str(output_path),
        "data_hash_sha256": data_hash,
        "file_bytes_sha256": file_hash,
        "hashing_decision_rationale": (
            "data_hash_sha256 is computed over sorted canonical row records to guarantee "
            "semantic reproducibility across environments where Parquet metadata timestamps vary. "
            "file_bytes_sha256 verifies physical transport integrity."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

    return output_path, manifest_path


def main():
    # P0: Print Parquet metadata
    print_metadata(SHARD_PATH)

    # P1: Benchmark batch reader
    cols = ["id", "text", "language_score"]
    benchmark_batch_reader(SHARD_PATH, columns=cols, batch_sizes=(512, 2048, 8192))


if __name__ == "__main__":
    main()
