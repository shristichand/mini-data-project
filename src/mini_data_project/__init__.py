"""
mini_data_project: End-to-end Text Dataset Engineering, Profiling, and Sampling Suite.

Organized into three functional domains:
- download: Dataset discovery, schema inspection, and shard downloading.
- sampling: Parquet batch streaming, reservoir sampling, stratified sampling, and manifests.
- text: Unicode normalization, text cleaning, language profiling, tokenizers, and dataset pipeline.
"""

from mini_data_project import download, sampling, text
from mini_data_project.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SAMPLE_10K_JSONL_PATH,
    SAMPLE_10K_PATH,
    SAMPLING_REPORTS_DIR,
    SHARD_PATH,
)
from mini_data_project.download import (
    download_shard,
    find_shards,
    get_or_create_fineweb2_sample,
    inspect_schema,
    process_fineweb2_sample_with_pipeline,
)
from mini_data_project.sampling import (
    add_strata,
    benchmark_batch_reader,
    compute_canonical_data_hash,
    compute_file_sha256,
    get_parquet_metadata,
    inspect_parquet_metadata,
    iter_batches,
    print_metadata,
    random_sample_inmemory,
    reservoir_sample,
    save_sample_with_manifest,
    stratified_equal,
    stratified_sample,
)
from mini_data_project.text import (
    BOM,
    NEPALI_STOPWORDS,
    ZWJ,
    ZWNJ,
    ZWSP,
    AksharaTokenizer,
    BPETokenizer,
    PipelineConfig,
    PipelineResult,
    SentenceTokenizer,
    TextDatasetPipeline,
    WordTokenizer,
    add_derived_features,
    calculate_quality_score,
    canonicalize_devanagari_nukta,
    clean_text_pipeline,
    clean_unicode_pipeline,
    convert_digits,
    devanagari_ratio,
    digit_ratio,
    extract_domain,
    get_unicode_block_distribution,
    latin_ratio,
    mask_urls_and_emails,
    ngram_repetition_ratio,
    normalize_punctuation,
    normalize_unicode,
    normalize_whitespace,
    parse_npi_score,
    profile_text,
    punct_ratio,
    punctuation_ratio,
    remove_control_characters,
    remove_invisible_characters,
    sanitize_zwj_zwnj,
    stopword_ratio,
    strip_html,
    type_token_ratio,
)

__version__ = "0.1.0"

__all__ = [
    # Subpackages
    "download",
    "sampling",
    "text",
    # Config
    "PROJECT_ROOT",
    "SHARD_PATH",
    "SAMPLE_10K_PATH",
    "SAMPLE_10K_JSONL_PATH",
    "SAMPLING_REPORTS_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "REPORTS_DIR",
    # Download
    "find_shards",
    "download_shard",
    "inspect_schema",
    "get_or_create_fineweb2_sample",
    "process_fineweb2_sample_with_pipeline",
    # Sampling
    "iter_batches",
    "get_parquet_metadata",
    "print_metadata",
    "inspect_parquet_metadata",
    "benchmark_batch_reader",
    "random_sample_inmemory",
    "reservoir_sample",
    "add_strata",
    "stratified_sample",
    "stratified_equal",
    "compute_canonical_data_hash",
    "compute_file_sha256",
    "save_sample_with_manifest",
    # Unicode & Normalizer
    "normalize_unicode",
    "remove_control_characters",
    "remove_invisible_characters",
    "sanitize_zwj_zwnj",
    "canonicalize_devanagari_nukta",
    "get_unicode_block_distribution",
    "clean_unicode_pipeline",
    "ZWJ",
    "ZWNJ",
    "ZWSP",
    "BOM",
    "clean_text_pipeline",
    "strip_html",
    "normalize_whitespace",
    "normalize_punctuation",
    "convert_digits",
    "mask_urls_and_emails",
    # Profiler & Metrics
    "profile_text",
    "calculate_quality_score",
    "devanagari_ratio",
    "latin_ratio",
    "digit_ratio",
    "punct_ratio",
    "punctuation_ratio",
    "stopword_ratio",
    "type_token_ratio",
    "ngram_repetition_ratio",
    "NEPALI_STOPWORDS",
    # Tokenizers
    "WordTokenizer",
    "SentenceTokenizer",
    "AksharaTokenizer",
    "BPETokenizer",
    # Pipeline
    "TextDatasetPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Utils
    "extract_domain",
    "add_derived_features",
    "parse_npi_score",
]


def main() -> None:
    """CLI entry point delegating to mini_data_project.cli."""
    from mini_data_project.cli import main as cli_main

    cli_main()
