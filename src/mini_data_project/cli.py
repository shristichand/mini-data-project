"""
mini_data_project.cli

Unified Command-Line Interface for the FineWeb-2 Nepali Devanagari
Data Engineering and Processing Suite.
"""

import argparse
import sys
from pathlib import Path


def cmd_info(args):
    """Display project environment, configuration paths, and dataset status."""
    from mini_data_project.config import (
        PROJECT_ROOT,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        REPORTS_DIR,
        SAMPLE_10K_PATH,
        SAMPLING_REPORTS_DIR,
        SHARD_PATH,
    )

    print("=" * 65)
    print("MINI-DATA-PROJECT: ENVIRONMENT & CONFIGURATION")
    print("=" * 65)
    print(f"Project Root          : {PROJECT_ROOT}")
    print(f"Raw Data Directory    : {RAW_DATA_DIR} (exists: {Path(RAW_DATA_DIR).exists()})")
    print(f"Processed Directory   : {PROCESSED_DATA_DIR} (exists: {Path(PROCESSED_DATA_DIR).exists()})")
    print(f"Reports Directory     : {REPORTS_DIR} (exists: {Path(REPORTS_DIR).exists()})")
    print(f"Sampling Reports      : {SAMPLING_REPORTS_DIR} (exists: {Path(SAMPLING_REPORTS_DIR).exists()})")
    print(f"Active Shard Path     : {SHARD_PATH} (exists: {Path(SHARD_PATH).exists()})")
    print(f"Sample 10k Path       : {SAMPLE_10K_PATH} (exists: {Path(SAMPLE_10K_PATH).exists()})")
    print("=" * 65)


def cmd_download(args):
    """Run shard discovery and download."""
    from mini_data_project.download import (
        download_shard,
        find_shards,
        get_or_create_fineweb2_sample,
        inspect_schema,
    )

    if args.action == "find":
        find_shards()
    elif args.action == "inspect":
        from mini_data_project.config import SHARD_PATH
        inspect_schema(args.file or SHARD_PATH)
    else:
        get_or_create_fineweb2_sample(sample_size=args.sample_size)


def cmd_profile(args):
    """Execute the comprehensive data profiling audit."""
    import pandas as pd
    from mini_data_project.config import SAMPLE_10K_PATH, SHARD_PATH
    from mini_data_project.text import add_derived_features

    path = args.input or (SAMPLE_10K_PATH if Path(SAMPLE_10K_PATH).exists() else SHARD_PATH)
    print(f"Profiling dataset at: {path}")

    # Import profile functions dynamically
    from scripts.text.profile import (
        profile_domain_breakdown,
        profile_duplicates,
        profile_language_distribution,
        profile_missing_values,
        profile_suspicious_records,
        profile_text_length,
    )

    df = pd.read_parquet(path)
    df = add_derived_features(df)

    profile_missing_values(df)
    profile_duplicates(df)
    profile_text_length(df)
    profile_suspicious_records(df)
    profile_language_distribution(df)
    profile_domain_breakdown(df)


def cmd_clean(args):
    """Run declarative quality filtering pipeline."""
    from scripts.text.clean import (
        DEFAULT_INPUT,
        DEFAULT_LOG,
        DEFAULT_OUTPUT,
        run_cleaning_pipeline,
    )

    input_path = args.input or DEFAULT_INPUT
    output_path = args.output or DEFAULT_OUTPUT
    log_path = args.log or DEFAULT_LOG

    run_cleaning_pipeline(input_path, output_path, log_path)


def cmd_visualize(args):
    """Generate publication-quality charts."""
    from scripts.text.visualize import (
        load_and_prepare_data,
        plot_language_score_histogram,
        plot_suspicious_records_barchart,
        plot_text_length_log_histogram,
        plot_top_domains_barchart,
    )

    input_path = args.input or "raw/sample_10k.parquet"
    output_dir = args.output_dir or "reports/figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(input_path)
    plot_text_length_log_histogram(df, output_dir)
    plot_language_score_histogram(df, output_dir)
    plot_top_domains_barchart(df, output_dir)
    plot_suspicious_records_barchart(df, output_dir)
    print(f"\nAll visualization figures successfully written to '{output_dir}'.")


def cmd_sample(args):
    """Execute sampling benchmarks, uniformity tests, and manifest generation."""
    from scripts.sampling.eval_sampling import run_full_sampling_evaluation
    run_full_sampling_evaluation()


def cmd_pipeline(args):
    """Run end-to-end TextDatasetPipeline."""
    from scripts.text.run_pipeline import main as run_pipeline_main
    sys.argv = [sys.argv[0]] + args.extra_args
    run_pipeline_main()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="mini-data-project",
        description="FineWeb-2 Nepali Devanagari Data Engineering & Sampling Suite.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    p_info = subparsers.add_parser("info", help="Display project configuration and paths")
    p_info.set_defaults(func=cmd_info)

    # download
    p_dl = subparsers.add_parser("download", help="Download shard and create sample")
    p_dl.add_argument(
        "--action",
        choices=["sample", "find", "inspect"],
        default="sample",
        help="Action to perform (default: sample)",
    )
    p_dl.add_argument("--file", type=str, default=None, help="File to inspect")
    p_dl.add_argument("--sample-size", type=int, default=10000, help="Sample size")
    p_dl.set_defaults(func=cmd_download)

    # profile
    p_prof = subparsers.add_parser("profile", help="Run 6-dimension data quality profiling")
    p_prof.add_argument("--input", type=str, default=None, help="Path to input Parquet file")
    p_prof.set_defaults(func=cmd_profile)

    # clean
    p_clean = subparsers.add_parser("clean", help="Run declarative cleaning pipeline")
    p_clean.add_argument("--input", type=str, default=None, help="Input parquet path")
    p_clean.add_argument("--output", type=str, default=None, help="Output parquet path")
    p_clean.add_argument("--log", type=str, default=None, help="Output cleaning log path")
    p_clean.set_defaults(func=cmd_clean)

    # visualize
    p_viz = subparsers.add_parser("visualize", help="Generate publication-quality charts")
    p_viz.add_argument("--input", type=str, default=None, help="Input parquet path")
    p_viz.add_argument("--output-dir", type=str, default=None, help="Output figures directory")
    p_viz.set_defaults(func=cmd_visualize)

    # sample
    p_sample = subparsers.add_parser("sample", help="Run sampling benchmarks and uniformity tests")
    p_sample.set_defaults(func=cmd_sample)

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run end-to-end TextDatasetPipeline")
    p_pipe.add_argument("extra_args", nargs="*", help="Arguments passed to pipeline runner")
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        print("\nQuick Info:")
        cmd_info(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
