"""
mini_data_project.text

Comprehensive linguistic text preprocessing suite:
- Unicode normalization (NFC/NFKC, Nukta, ZWJ/ZWNJ sanitization)
- Text normalization (whitespace, punctuation, digits, HTML, masking)
- Language profiling and quality scoring (Devanagari ratio, TTR, stopwords)
- Tokenizers (Sentence, Word, Akshara, Byte-Pair Encoding)
- TextDatasetPipeline execution engine
"""

from mini_data_project.text.language_profiler import (
    NEPALI_STOPWORDS,
    calculate_quality_score,
    devanagari_ratio,
    digit_ratio,
    latin_ratio,
    ngram_repetition_ratio,
    profile_text,
    punct_ratio,
    punctuation_ratio,
    stopword_ratio,
    type_token_ratio,
)
from mini_data_project.text.text_normalizer import (
    clean_text_pipeline,
    convert_digits,
    mask_urls_and_emails,
    normalize_punctuation,
    normalize_whitespace,
    strip_html,
)
from mini_data_project.text.text_pipeline import (
    PipelineConfig,
    PipelineResult,
    TextDatasetPipeline,
    process_fineweb2_sample_with_pipeline,
)
from mini_data_project.text.text_utils import (
    EXTENDED_PUNCT_SET,
    PUNCT_SET,
    add_derived_features,
    extract_domain,
    parse_npi_score,
    punct_digit_ratio,
)
from mini_data_project.text.tokenizers import (
    AksharaTokenizer,
    BPETokenizer,
    SentenceTokenizer,
    WordTokenizer,
)
from mini_data_project.text.unicode_utils import (
    BOM,
    ZWJ,
    ZWNJ,
    ZWSP,
    canonicalize_devanagari_nukta,
    clean_unicode_pipeline,
    get_unicode_block_distribution,
    normalize_unicode,
    remove_control_characters,
    remove_invisible_characters,
    sanitize_zwj_zwnj,
)

__all__ = [
    # Unicode
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
    # Normalizer
    "clean_text_pipeline",
    "strip_html",
    "normalize_whitespace",
    "normalize_punctuation",
    "convert_digits",
    "mask_urls_and_emails",
    # Profiler
    "profile_text",
    "calculate_quality_score",
    "devanagari_ratio",
    "latin_ratio",
    "digit_ratio",
    "punct_ratio",
    "punct_digit_ratio",
    "punctuation_ratio",
    "PUNCT_SET",
    "EXTENDED_PUNCT_SET",
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
    "process_fineweb2_sample_with_pipeline",
    # Utils
    "extract_domain",
    "add_derived_features",
    "parse_npi_score",
]
