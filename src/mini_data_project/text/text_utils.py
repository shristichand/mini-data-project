"""
src/text_utils.py

Shared text processing, feature extraction, and language parsing utilities
for profiling and cleaning FineWeb-2 Nepali Devanagari (npi_Deva) data.
"""

import json
import string
from collections import Counter
from urllib.parse import urlparse
import pandas as pd

# Extended punctuation set covering ASCII and standard Nepali/Devanagari marks (including guillemets « »)
PUNCT_SET: set[str] = set(string.punctuation) | {
    "।", "॥", "‘", "’", "“", "”", "—", "–", "…", "·", "•", "«", "»"
}
EXTENDED_PUNCT_SET: set[str] = PUNCT_SET  # Canonical alias for backward compatibility


def devanagari_ratio(text: str) -> float:
    """Compute the ratio of Devanagari characters (Unicode U+0900 to U+097F)."""
    if not text:
        return 0.0
    deva_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return deva_chars / len(text)


def punct_ratio(text: str) -> float:
    """Compute the ratio of punctuation marks against the canonical PUNCT_SET."""
    if not text:
        return 0.0
    p_chars = sum(1 for c in text if c in PUNCT_SET)
    return p_chars / len(text)


def digit_ratio(text: str) -> float:
    """Compute the ratio of numeric digits (both ASCII 0-9 and Devanagari ०-९)."""
    if not text:
        return 0.0
    d_chars = sum(1 for c in text if c.isdigit() or ("\u0966" <= c <= "\u096F"))
    return d_chars / len(text)


def punct_digit_ratio(text: str) -> float:
    """Compute the combined ratio of numeric digits and punctuation marks."""
    if not text:
        return 0.0
    return punct_ratio(text) + digit_ratio(text)


def max_line_repetition(text: str, min_line_len: int = 4) -> int:
    """Compute the maximum frequency of any non-trivial line in the text."""
    if not text:
        return 0
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= min_line_len]
    if not lines:
        return 0
    return Counter(lines).most_common(1)[0][1]


def parse_npi_score(val) -> float | None:
    """Extract npi_Deva_score from top_langs (handling dict or JSON-encoded string)."""
    if isinstance(val, dict):
        return val.get("npi_Deva_score", val.get("npi_Deva"))
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed.get("npi_Deva_score", parsed.get("npi_Deva"))
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def extract_domain(url: str) -> str:
    """Extract and normalize root network domain from a URL."""
    if not isinstance(url, str):
        return "unknown"
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return "unknown"


def add_derived_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Enrich a DataFrame with derived quality, length, and linguistic metrics in place.
    Provides dual aliases (char_len/char_count, word_len/word_count, punct_digit_ratio/punct_ratio+digit_ratio)
    to eliminate vocabulary fragmentation across modules.
    """
    col = text_column if text_column in df.columns else "text"
    text_series = df[col].astype(str)

    # Document length (dual vocabulary: char_len & char_count)
    if "char_len" not in df.columns:
        df["char_len"] = text_series.str.len()
    if "char_count" not in df.columns:
        df["char_count"] = df["char_len"]

    # Word count (dual vocabulary: word_len & word_count)
    if "word_len" not in df.columns:
        df["word_len"] = text_series.str.split().str.len()
    if "word_count" not in df.columns:
        df["word_count"] = df["word_len"]

    # Script purity
    if "deva_ratio" not in df.columns:
        df["deva_ratio"] = text_series.apply(devanagari_ratio)

    # Symbol and digit noise (both individual and combined metrics)
    if "punct_ratio" not in df.columns:
        df["punct_ratio"] = text_series.apply(punct_ratio)
    if "digit_ratio" not in df.columns:
        df["digit_ratio"] = text_series.apply(digit_ratio)
    if "punct_digit_ratio" not in df.columns:
        df["punct_digit_ratio"] = df["punct_ratio"] + df["digit_ratio"]

    # Repetition
    if "max_line_repeat" not in df.columns:
        df["max_line_repeat"] = text_series.apply(max_line_repetition)

    # FineWeb-2 metadata
    if "npi_Deva_score" not in df.columns:
        if "top_langs" in df.columns:
            df["npi_Deva_score"] = df["top_langs"].apply(parse_npi_score)
        elif "language_score" in df.columns:
            df["npi_Deva_score"] = pd.to_numeric(df["language_score"], errors="coerce")

    # Domain
    if "domain" not in df.columns and "url" in df.columns:
        df["domain"] = df["url"].apply(extract_domain)

    return df

