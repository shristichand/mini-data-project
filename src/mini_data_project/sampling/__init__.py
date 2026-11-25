"""
mini_data_project.sampling

Parquet batch reader, streaming reservoir sampling, stratified sampling,
and reproducibility manifests.
"""

from mini_data_project.sampling.sampler import (
    add_strata,
    benchmark_batch_reader,
    compute_canonical_data_hash,
    compute_file_sha256,
    get_parquet_metadata,
    inspect_parquet_metadata,
    inspect_schema,
    iter_batches,
    print_metadata,
    random_sample_inmemory,
    reservoir_sample,
    save_sample_with_manifest,
    stratified_equal,
    stratified_sample,
)

__all__ = [
    "iter_batches",
    "get_parquet_metadata",
    "inspect_parquet_metadata",
    "inspect_schema",
    "print_metadata",
    "benchmark_batch_reader",
    "random_sample_inmemory",
    "reservoir_sample",
    "add_strata",
    "stratified_sample",
    "stratified_equal",
    "compute_canonical_data_hash",
    "compute_file_sha256",
    "save_sample_with_manifest",
]
