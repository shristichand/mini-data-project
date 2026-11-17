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

from language_profiler import profile_text, calculate_quality_score
from text_normalizer import clean_text_pipeline
from tokenizers import AksharaTokenizer, BPETokenizer, SentenceTokenizer, WordTokenizer
from unicode_utils import clean_unicode_pipeline


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

    # Quality filters (set to None or 0 to disable)
    min_char_len: int = 50
    max_char_len: int = 100000
    min_deva_ratio: float = 0.50
    max_punct_digit_ratio: float = 0.40
    max_ngram_repetition: float = 0.45
    min_quality_score: float = 0.35
    min_npi_score: Optional[float] = None  # FineWeb-2 language classifier confidence

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

        char_len = profile.get("char_count", len(text))
        if char_len < self.config.min_char_len:
            reasons.append(f"below_min_char_len ({char_len} < {self.config.min_char_len})")
        if char_len > self.config.max_char_len:
            reasons.append(f"above_max_char_len ({char_len} > {self.config.max_char_len})")

        deva_ratio = profile.get("deva_ratio", 0.0)
        if deva_ratio < self.config.min_deva_ratio:
            reasons.append(f"low_devanagari_ratio ({deva_ratio:.3f} < {self.config.min_deva_ratio})")

        pd_ratio = profile.get("punct_ratio", 0.0) + profile.get("digit_ratio", 0.0)
        if pd_ratio > self.config.max_punct_digit_ratio:
            reasons.append(f"high_punct_digit_noise ({pd_ratio:.3f} > {self.config.max_punct_digit_ratio})")

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
            from text_utils import parse_npi_score
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
        processed_df["word_count"] = word_counts
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
