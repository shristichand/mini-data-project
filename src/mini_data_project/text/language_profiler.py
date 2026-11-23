"""
src/language_profiler.py

Linguistic profiling, script characterization, lexical diversity metrics,
and heuristic quality scoring for text datasets (with specialized Nepali Devanagari support).
"""

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Set

# Extended punctuation set covering ASCII and standard Nepali/Devanagari marks
EXTENDED_PUNCT_SET: Set[str] = set(string.punctuation) | {
    "।", "॥", "‘", "’", "“", "”", "—", "–", "…", "·", "•", "«", "»", "।", "॥"
}

# Curated list of high-frequency Nepali stopwords
NEPALI_STOPWORDS: Set[str] = {
    "र", "को", "मा", "का", "ले", "लाई", "छ", "पनि", "भने", "हुने",
    "गर्न", "एक", "हो", "छन्", "यो", "त्यो", "भएको", "गरी", "थियो",
    "बाट", "यस", "लागि", "हुन्", "गरेको", "थिए", "रहेको", "दिन",
    "आफ्नो", "अरू", "वा", "भनी", "भनेर", "उनले", "उनको", "गरेका",
    "भएका", "साथै", "सबै", "कसरी", "किन", "जब", "तब", "जहाँ", "त्यहाँ",
    "के", "कुनै", "हुनसक्छ", "गरे", "हुन", "गर्ने", "हुन्छ", "गर्नुपर्छ",
    "गरेर", "भएर", "भन्दा", "समेत", "दुई", "तीन", "चार", "पाँच", "तर",
    "न", "अथवा", "कृपया", "धेरै", "थोरै", "अहिले", "पहिले", "पछि",
    "अगाडि", "भित्र", "बाहिर", "माथि", "तल", "बीच", "प्रति", "बारे",
    "विरुद्ध", "द्वारा", "मार्फत", "यसरी", "त्यसरी", "कसैले", "कोही",
    "यति", "त्यति", "केही", "धेरैजसो", "अधिकांश", "यद्यपि", "तथापि",
}


def devanagari_ratio(text: str) -> float:
    """Compute the ratio of Devanagari characters (U+0900 to U+097F)."""
    if not text:
        return 0.0
    deva_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return deva_count / len(text)


def latin_ratio(text: str) -> float:
    """Compute the ratio of Latin alphabet characters (a-z, A-Z)."""
    if not text:
        return 0.0
    latin_count = sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    return latin_count / len(text)


def digit_ratio(text: str) -> float:
    """Compute ratio of digits (both ASCII 0-9 and Devanagari ०-९)."""
    if not text:
        return 0.0
    digit_count = sum(
        1 for c in text if c.isdigit() or ("\u0966" <= c <= "\u096F")
    )
    return digit_count / len(text)


def punct_ratio(text: str) -> float:
    """Compute ratio of punctuation marks."""
    if not text:
        return 0.0
    p_count = sum(1 for c in text if c in EXTENDED_PUNCT_SET)
    return p_count / len(text)


punctuation_ratio = punct_ratio


def whitespace_ratio(text: str) -> float:
    """Compute ratio of whitespace characters."""
    if not text:
        return 0.0
    ws_count = sum(1 for c in text if c.isspace())
    return ws_count / len(text)


def other_script_ratio(text: str) -> float:
    """Compute ratio of non-Devanagari, non-Latin, non-digit, non-punct characters."""
    if not text:
        return 0.0
    other_count = sum(
        1
        for c in text
        if not ("\u0900" <= c <= "\u097F")
        and not (("a" <= c <= "z") or ("A" <= c <= "Z"))
        and not (c.isdigit() or "\u0966" <= c <= "\u096F")
        and (c not in EXTENDED_PUNCT_SET)
        and not c.isspace()
    )
    return other_count / len(text)


def get_words(text: str) -> List[str]:
    """Tokenize words by whitespace and strip boundary punctuation."""
    raw_tokens = text.split()
    words = []
    punct_strip = "".join(EXTENDED_PUNCT_SET)
    for t in raw_tokens:
        clean = t.strip(punct_strip)
        if clean:
            words.append(clean)
    return words


def type_token_ratio(text: str) -> float:
    """
    Calculate Type-Token Ratio (TTR = unique words / total words).
    Measures lexical diversity; values closer to 1.0 indicate high vocabulary variety.
    """
    words = get_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def stopword_ratio(text: str, custom_stopwords: Set[str] = None) -> float:
    """
    Compute ratio of recognized Nepali stopwords in the tokenized words.
    High stopword ratio (>0.15) typically indicates natural grammatical prose.
    """
    words = get_words(text)
    if not words:
        return 0.0
    stopwords = custom_stopwords if custom_stopwords is not None else NEPALI_STOPWORDS
    sw_count = sum(1 for w in words if w in stopwords)
    return sw_count / len(words)


def max_line_repetition(text: str, min_line_len: int = 4) -> int:
    """Compute maximum frequency of any non-trivial line in the text."""
    if not text:
        return 0
    lines = [
        line.strip() for line in text.split("\n") if len(line.strip()) >= min_line_len
    ]
    if not lines:
        return 0
    return Counter(lines).most_common(1)[0][1]


def ngram_repetition_ratio(text: str, n: int = 3) -> float:
    """
    Compute n-gram repetition ratio (1 - unique n-grams / total n-grams).
    High values (>0.3) often indicate repeated boilerplate, templated text, or loops.
    """
    words = get_words(text)
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_count = len(set(ngrams))
    total_count = len(ngrams)
    return (total_count - unique_count) / total_count


def calculate_quality_score(
    text: str,
    target_script: str = "Devanagari",
    min_char_len: int = 40,
    max_char_len: int = 25000,
) -> float:
    """
    Heuristic text quality scoring on a 0.0 to 1.0 scale based on:
    - Target script purity
    - Punctuation / digit noise penalty
    - Stopword density (grammatical coherence)
    - Repetition penalty
    - Document length sanity
    """
    if not text:
        return 0.0

    length = len(text)
    # Length sanity
    if length < 10:
        return 0.0
    length_penalty = 1.0
    if length < min_char_len:
        length_penalty = length / max(1, min_char_len)
    elif length > max_char_len:
        length_penalty = max(0.2, 1.0 - (length - max_char_len) / max_char_len)

    # Script purity
    deva_rat = devanagari_ratio(text) if target_script == "Devanagari" else 1.0
    if deva_rat < 0.4:
        return 0.1 * deva_rat

    # Noise ratio
    pd_ratio = punct_ratio(text) + digit_ratio(text)
    noise_penalty = max(0.0, 1.0 - (pd_ratio * 1.5))

    # Stopword presence
    sw_rat = stopword_ratio(text)
    # Natural Nepali text usually has 10% - 35% stopwords
    if sw_rat < 0.03:
        sw_score = max(0.2, sw_rat / 0.05)
    elif sw_rat > 0.6:
        sw_score = 0.5  # abnormal repetition of stopwords
    else:
        sw_score = min(1.0, 0.5 + (sw_rat / 0.3))

    # Repetition penalty
    rep_ratio = ngram_repetition_ratio(text, n=3)
    rep_penalty = max(0.0, 1.0 - (rep_ratio * 2.0))

    # Composite weighted score
    raw_score = (
        0.35 * deva_rat
        + 0.25 * noise_penalty
        + 0.20 * sw_score
        + 0.20 * rep_penalty
    ) * length_penalty

    return round(float(max(0.0, min(1.0, raw_score))), 4)


def profile_text(text: str, min_char_len: int = 40) -> Dict[str, Any]:
    """
    Generate an exhaustive linguistic, statistical, and quality profile for a document.
    """
    if not text:
        return {
            "char_count": 0,
            "byte_count": 0,
            "word_count": 0,
            "line_count": 0,
            "deva_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
            "punct_ratio": 0.0,
            "whitespace_ratio": 0.0,
            "other_script_ratio": 0.0,
            "ttr": 0.0,
            "stopword_ratio": 0.0,
            "max_line_repeat": 0,
            "ngram_3_repetition": 0.0,
            "quality_score": 0.0,
        }

    words = get_words(text)
    return {
        "char_count": len(text),
        "byte_count": len(text.encode("utf-8")),
        "word_count": len(words),
        "line_count": len(text.split("\n")),
        "deva_ratio": round(devanagari_ratio(text), 4),
        "latin_ratio": round(latin_ratio(text), 4),
        "digit_ratio": round(digit_ratio(text), 4),
        "punct_ratio": round(punct_ratio(text), 4),
        "whitespace_ratio": round(whitespace_ratio(text), 4),
        "other_script_ratio": round(other_script_ratio(text), 4),
        "ttr": round(type_token_ratio(text), 4),
        "stopword_ratio": round(stopword_ratio(text), 4),
        "max_line_repeat": max_line_repetition(text),
        "ngram_3_repetition": round(ngram_repetition_ratio(text, n=3), 4),
        "quality_score": calculate_quality_score(text, min_char_len=min_char_len),
    }
