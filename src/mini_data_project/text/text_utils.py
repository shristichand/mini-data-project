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

# Extended punctuation set covering ASCII and standard Nepali/Devanagari marks
PUNCT_SET = set(string.punctuation) | {"।", "॥", "‘", "’", "“", "”", "—", "–", "…", "·", "•"}


def devanagari_ratio(text: str) -> float:
    """Compute the ratio of Devanagari characters (Unicode U+0900 to U+097F)."""
    if not text:
        return 0.0
    deva_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return deva_chars / len(text)


def punct_digit_ratio(text: str) -> float:
    """Compute the ratio of numeric digits and punctuation marks."""
    if not text:
        return 0.0
    pd_chars = sum(1 for c in text if c.isdigit() or c in PUNCT_SET)
    return pd_chars / len(text)


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


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a DataFrame with derived quality, length, and linguistic metrics in place.
    Avoids re-computing features multiple times across different modules.
    """
    if "char_len" not in df.columns:
        df["char_len"] = df["text"].astype(str).str.len()
    if "word_len" not in df.columns:
        df["word_len"] = df["text"].astype(str).str.split().str.len()
    if "deva_ratio" not in df.columns:
        df["deva_ratio"] = df["text"].apply(devanagari_ratio)
    if "punct_digit_ratio" not in df.columns:
        df["punct_digit_ratio"] = df["text"].apply(punct_digit_ratio)
    if "max_line_repeat" not in df.columns:
        df["max_line_repeat"] = df["text"].apply(max_line_repetition)
    if "npi_Deva_score" not in df.columns and "top_langs" in df.columns:
        df["npi_Deva_score"] = df["top_langs"].apply(parse_npi_score)
    if "domain" not in df.columns and "url" in df.columns:
        df["domain"] = df["url"].apply(extract_domain)
    return df

