"""
src/text_pipeline.py

Composable, high-throughput text dataset preprocessing pipeline.
Integrates:
1. Unicode Normalization & Devanagari ZWJ/Nukta handling
2. Text Normalization, HTML stripping, punctuation & whitespace cleanup
3. Language Characteristics Profiling & Quality Scoring
4. Quality Filtering (rejection tracking)
5. Multi-level Tokenization (Word, Sentence, Akshara, BPE)
6. Parquet / Batch / DataFrame streaming orchestration
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mini_data_project.config import FilterThresholds
from mini_data_project.text.language_profiler import profile_text, calculate_quality_score
from mini_data_project.text.text_normalizer import clean_text_pipeline
from mini_data_project.text.tokenizers import (
    AksharaTokenizer,
    BPETokenizer,
    SentenceTokenizer,
    WordTokenizer,
)
from mini_data_project.text.unicode_utils import clean_unicode_pipeline


@dataclass
class PipelineConfig:
    """Configuration for all stages of the text dataset pipeline."""

    # Unicode normalization
    unicode_form: str = "NFC"
    strip_controls: bool = True
    clean_invisibles: bool = True
    sanitize_joiners: bool = True
    canonical_nukta: bool = True

    # Text normalization
    strip_html_tags: bool = True
    clean_urls: bool = False
    clean_emails: bool = False
    normalize_whitespace: bool = True
    normalize_punct: bool = True
    ascii_digits: bool = False
    lowercase: bool = False

    # Quality filters (canonical single source of truth from FilterThresholds)
    min_char_len: int = FilterThresholds.min_char_len
    max_char_len: int = FilterThresholds.max_char_len
    min_deva_ratio: float = FilterThresholds.min_deva_ratio
    max_punct_digit_ratio: float = FilterThresholds.max_punct_digit_ratio
    max_line_repeat: Optional[int] = FilterThresholds.max_line_repeat
    max_ngram_repetition: float = FilterThresholds.max_ngram_repetition
    min_quality_score: float = FilterThresholds.min_quality_score
    min_npi_score: Optional[float] = None  # FineWeb-2 language classifier confidence (None or float)

    # Tokenization
    tokenization_mode: Optional[str] = "word"  # "word", "sentence", "akshara", "bpe", None
    bpe_vocab_size: int = 4000
    bpe_model_path: Optional[str] = None


@dataclass
class PipelineResult:
    """Output for a single processed text document."""
    raw_text: str
    cleaned_text: str
    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)
    tokens: Optional[List[str]] = None
    profile: Dict[str, Any] = field(default_factory=dict)


class TextDatasetPipeline:
    """
    Production-grade text processing pipeline for dataset curation, profiling,
    filtering, and tokenization.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._init_tokenizers()

    def _init_tokenizers(self):
        """Instantiate the active tokenizer according to configuration."""
        mode = self.config.tokenization_mode
        if mode == "word":
            self.tokenizer = WordTokenizer()
        elif mode == "sentence":
            self.tokenizer = SentenceTokenizer()
        elif mode == "akshara":
            self.tokenizer = AksharaTokenizer()
        elif mode == "bpe":
            if self.config.bpe_model_path and os.path.exists(self.config.bpe_model_path):
                self.tokenizer = BPETokenizer.load(self.config.bpe_model_path)
            else:
                self.tokenizer = BPETokenizer(vocab_size=self.config.bpe_vocab_size)
        else:
            self.tokenizer = None

    def normalize(self, text: str) -> str:
        """Execute Unicode and text normalization stages."""
        if not text:
            return ""

        # Step 1: Unicode Normalization & sanitation
        text = clean_unicode_pipeline(
            text,
            form=self.config.unicode_form,
            strip_controls=self.config.strip_controls,
            clean_invisibles=self.config.clean_invisibles,
            sanitize_joiners=self.config.sanitize_joiners,
            canonical_nukta=self.config.canonical_nukta,
        )

        # Step 2: Surface text cleaning & normalization
        text = clean_text_pipeline(
            text,
            strip_html_tags=self.config.strip_html_tags,
            clean_urls=self.config.clean_urls,
            clean_emails=self.config.clean_emails,
            normalize_spaces=self.config.normalize_whitespace,
            normalize_punct=self.config.normalize_punct,
            ascii_digits=self.config.ascii_digits,
            lowercase=self.config.lowercase,
        )

        return text

    def evaluate_filters(
        self, text: str, profile: Dict[str, Any], npi_score: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """Check all configured quality filter thresholds."""
        reasons = []

        char_len = profile.get("char_count", profile.get("char_len", len(text)))
        if char_len < self.config.min_char_len:
            reasons.append(f"below_min_char_len ({char_len} < {self.config.min_char_len})")
        if char_len > self.config.max_char_len:
            reasons.append(f"above_max_char_len ({char_len} > {self.config.max_char_len})")

        deva_ratio = profile.get("deva_ratio", 0.0)
        if deva_ratio < self.config.min_deva_ratio:
            reasons.append(f"low_devanagari_ratio ({deva_ratio:.3f} < {self.config.min_deva_ratio})")

        pd_ratio = profile.get(
            "punct_digit_ratio",
            profile.get("punct_ratio", 0.0) + profile.get("digit_ratio", 0.0),
        )
        if pd_ratio > self.config.max_punct_digit_ratio:
            reasons.append(f"high_punct_digit_noise ({pd_ratio:.3f} > {self.config.max_punct_digit_ratio})")

        if self.config.max_line_repeat is not None:
            max_repeat = profile.get("max_line_repeat", 0)
            if max_repeat >= self.config.max_line_repeat:
                reasons.append(f"high_line_repetition ({max_repeat} >= {self.config.max_line_repeat})")

        rep_ratio = profile.get("ngram_3_repetition", 0.0)
        if rep_ratio > self.config.max_ngram_repetition:
            reasons.append(f"high_ngram_repetition ({rep_ratio:.3f} > {self.config.max_ngram_repetition})")

        q_score = profile.get("quality_score", 0.0)
        if q_score < self.config.min_quality_score:
            reasons.append(f"low_quality_score ({q_score:.3f} < {self.config.min_quality_score})")

        if self.config.min_npi_score is not None:
            if npi_score is None or npi_score < self.config.min_npi_score:
                score_str = f"{npi_score:.4f}" if npi_score is not None else "None"
                reasons.append(f"low_npi_score ({score_str} < {self.config.min_npi_score})")

        passed = len(reasons) == 0
        return passed, reasons

    def process_text(self, text: str, npi_score: Optional[float] = None) -> PipelineResult:
        """Process a single text sample end-to-end."""
        cleaned = self.normalize(text)
        profile = profile_text(cleaned, min_char_len=self.config.min_char_len)
        passed, reasons = self.evaluate_filters(cleaned, profile, npi_score=npi_score)

        tokens = None
        if passed and self.tokenizer is not None:
            if isinstance(self.tokenizer, BPETokenizer):
                tokens = self.tokenizer.encode(cleaned)
            else:
                tokens = self.tokenizer.tokenize(cleaned)

        return PipelineResult(
            raw_text=text,
            cleaned_text=cleaned,
            passed=passed,
            rejection_reasons=reasons,
            tokens=tokens,
            profile=profile,
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        output_tokens: bool = True,
        filter_rejected: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a pandas DataFrame of text records.
        Supports FineWeb-2 metadata integration (top_langs, language_score, url).
        Returns the processed DataFrame and an execution metrics report.
        """
        start_time = time.time()
        initial_count = len(df)
        rejection_stats = Counter()

        # Extract FineWeb-2 language scores if present
        npi_scores = None
        if "top_langs" in df.columns:
            from mini_data_project.text.text_utils import parse_npi_score
            npi_scores = [parse_npi_score(v) for v in df["top_langs"]]
        elif "language_score" in df.columns:
            npi_scores = [float(v) if pd.notna(v) else None for v in df["language_score"]]

        cleaned_texts = []
        pass_flags = []
        token_lists = []
        quality_scores = []
        deva_ratios = []
        char_counts = []
        word_counts = []

        raw_texts = df[text_column].astype(str).tolist()
        for i, raw_val in enumerate(raw_texts):
            score = npi_scores[i] if npi_scores is not None else None
            res = self.process_text(raw_val, npi_score=score)
            cleaned_texts.append(res.cleaned_text)
            pass_flags.append(res.passed)
            token_lists.append(res.tokens if res.tokens is not None else [])
            quality_scores.append(res.profile.get("quality_score", 0.0))
            deva_ratios.append(res.profile.get("deva_ratio", 0.0))
            char_counts.append(res.profile.get("char_count", 0))
            word_counts.append(res.profile.get("word_count", 0))

            for r in res.rejection_reasons:
                tag = r.split()[0]
                rejection_stats[tag] += 1

        processed_df = df.copy()
        processed_df["cleaned_text"] = cleaned_texts
        processed_df["passed_pipeline"] = pass_flags
        processed_df["char_count"] = char_counts
        processed_df["char_len"] = char_counts
        processed_df["word_count"] = word_counts
        processed_df["word_len"] = word_counts
        processed_df["deva_ratio"] = deva_ratios
        processed_df["quality_score"] = quality_scores

        if npi_scores is not None:
            processed_df["npi_Deva_score"] = npi_scores

        if output_tokens and self.tokenizer is not None:
            processed_df["tokens"] = token_lists
            processed_df["token_count"] = [len(t) for t in token_lists]

        if filter_rejected:
            result_df = processed_df[processed_df["passed_pipeline"]].copy().reset_index(drop=True)
        else:
            result_df = processed_df

        elapsed = round(time.time() - start_time, 3)
        kept_count = len(result_df) if filter_rejected else int(sum(pass_flags))

        report = {
            "initial_records": initial_count,
            "retained_records": kept_count,
            "dropped_records": initial_count - kept_count,
            "retention_rate_pct": round((kept_count / initial_count) * 100, 2) if initial_count > 0 else 0,
            "elapsed_seconds": elapsed,
            "throughput_docs_sec": round(initial_count / max(0.001, elapsed), 2),
            "rejection_breakdown": dict(rejection_stats),
            "tokenizer_mode": self.config.tokenization_mode,
        }

        return result_df, report

    def process_parquet(
        self,
        input_path: str,
        output_path: str,
        text_column: str = "text",
        batch_size: int = 2000,
        filter_rejected: bool = True,
    ) -> Dict[str, Any]:
        """
        Stream a large Parquet dataset in memory-bounded batches,
        apply pipeline stages, and write retained records to output Parquet.
        """
        start_time = time.time()
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = None
        total_kept = 0
        total_dropped = 0
        overall_rejections = Counter()

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = batch.to_pandas()
            proc_df, batch_report = self.process_dataframe(
                batch_df,
                text_column=text_column,
                output_tokens=True,
                filter_rejected=filter_rejected,
            )

            total_kept += len(proc_df)
            total_dropped += batch_report["dropped_records"]
            for k, v in batch_report["rejection_breakdown"].items():
                overall_rejections[k] += v

            # Write batches to output
            if len(proc_df) > 0:
                table = pa.Table.from_pandas(proc_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                writer.write_table(table)

        if writer is not None:
            writer.close()

        elapsed = round(time.time() - start_time, 3)
        summary = {
            "input_file": input_path,
            "output_file": output_path,
            "total_input_rows": total_rows,
            "retained_rows": total_kept,
            "dropped_rows": total_dropped,
            "retention_rate_pct": round((total_kept / total_rows) * 100, 2) if total_rows > 0 else 0,
            "elapsed_seconds": elapsed,
            "throughput_docs_sec": round(total_rows / max(0.001, elapsed), 2),
            "rejections": dict(overall_rejections),
            "config": asdict(self.config),
        }

        return summary


def process_fineweb2_sample_with_pipeline(
    sample_path: Optional[str] = None,
    output_path: str = "processed/fineweb2_clean_tokenized.parquet",
    report_path: str = "reports/fineweb2_pipeline_report.json",
    tokenizer_mode: str = "word",
    bpe_vocab_size: int = 3000,
    min_deva_ratio: float = 0.50,
    min_quality_score: float = 0.35,
    min_npi_score: Optional[float] = 0.95,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Execute the complete Text Dataset Pipeline on a FineWeb-2 dataset sample:
    - Unicode Normalization & ZWJ/Nukta handling
    - Text sanitization, HTML stripping, punctuation & whitespace cleanup
    - Language characteristics profiling (TTR, script ratios, FineWeb-2 language scores)
    - Quality filtering
    - Tokenization (Word, Sentence, Akshara, or Subword BPE)
    """
    from mini_data_project.download import (
        REPO_ID,
        SUBSET,
        get_or_create_fineweb2_sample,
    )
    from mini_data_project.text.tokenizers import BPETokenizer

    # 1. Get or create FineWeb-2 sample
    input_file = get_or_create_fineweb2_sample(preferred_path=sample_path)
    print(f"\n[FineWeb-2 Pipeline] Loading sample from {input_file}...")
    df = pd.read_parquet(input_file)

    if limit and limit < len(df):
        print(f"Limiting to first {limit:,} records...")
        df = df.iloc[:limit].copy()

    initial_count = len(df)
    print(f"Input records: {initial_count:,}")

    # 2. Setup BPE if requested
    bpe_model_path = None
    if tokenizer_mode == "bpe":
        bpe_save = "reports/fineweb2_bpe_tokenizer.json"
        print(f"Training BPE Tokenizer on FineWeb-2 sample (vocab size: {bpe_vocab_size})...")
        bpe = BPETokenizer(vocab_size=bpe_vocab_size)
        sample_texts = df["text"].dropna().sample(min(1000, len(df)), random_state=42)
        bpe.train(sample_texts, vocab_size=bpe_vocab_size, max_corpus_words=10000)
        bpe.save(bpe_save)
        bpe_model_path = bpe_save
        print(f"BPE Tokenizer saved to {bpe_save} (Vocab size: {len(bpe.vocab)}).")

    # 3. Configure Pipeline
    config = PipelineConfig(
        unicode_form="NFC",
        strip_controls=True,
        clean_invisibles=True,
        sanitize_joiners=True,
        canonical_nukta=True,
        strip_html_tags=True,
        normalize_whitespace=True,
        normalize_punct=True,
        min_char_len=40,
        max_char_len=100000,
        min_deva_ratio=min_deva_ratio,
        max_punct_digit_ratio=0.40,
        max_ngram_repetition=0.45,
        min_quality_score=min_quality_score,
        min_npi_score=min_npi_score,
        tokenization_mode=tokenizer_mode,
        bpe_vocab_size=bpe_vocab_size,
        bpe_model_path=bpe_model_path,
    )

    pipeline = TextDatasetPipeline(config=config)
    print(f"\n[FineWeb-2 Pipeline] Executing pipeline with tokenizer='{tokenizer_mode}'...")
    processed_df, report = pipeline.process_dataframe(
        df,
        text_column="text",
        output_tokens=True,
        filter_rejected=True,
    )

    # 4. Save results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    processed_df.to_parquet(output_path, index=False, compression="snappy")
    print(f"  -> Saved cleaned & tokenized FineWeb-2 dataset to: {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")

    full_report = {
        "dataset": REPO_ID,
        "subset": SUBSET,
        "source_sample": input_file,
        "output_file": output_path,
        "pipeline_statistics": report,
        "config": {
            "tokenizer_mode": tokenizer_mode,
            "min_deva_ratio": min_deva_ratio,
            "min_quality_score": min_quality_score,
            "min_npi_score": min_npi_score,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved FineWeb-2 pipeline report to: {report_path}")

    return processed_df, report
