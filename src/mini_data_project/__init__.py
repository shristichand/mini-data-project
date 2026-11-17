"""mini_data_project package root."""

from language_profiler import (
    NEPALI_STOPWORDS,
    calculate_quality_score,
    devanagari_ratio,
    latin_ratio,
    profile_text,
    stopword_ratio,
    type_token_ratio,
)
from text_normalizer import (
    clean_text_pipeline,
    convert_digits,
    normalize_punctuation,
    normalize_whitespace,
    strip_html,
)
from text_pipeline import PipelineConfig, PipelineResult, TextDatasetPipeline
from tokenizers import (
    AksharaTokenizer,
    BPETokenizer,
    SentenceTokenizer,
    WordTokenizer,
)
from unicode_utils import (
    canonicalize_devanagari_nukta,
    clean_unicode_pipeline,
    get_unicode_block_distribution,
    normalize_unicode,
    remove_control_characters,
    remove_invisible_characters,
    sanitize_zwj_zwnj,
)

__version__ = "0.1.0"


def main() -> None:
    print("mini-data-project: Text dataset engineering and preprocessing suite active.")
