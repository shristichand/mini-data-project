"""
src/unicode_utils.py

Unicode normalization, character classification, and script sanitation
utilities for multilingual text with specialized support for Devanagari (npi_Deva).
"""

import re
import unicodedata
from typing import Dict, Tuple

# Unicode range constants
DEVANAGARI_RANGE = ("\u0900", "\u097F")
DEVANAGARI_EXT_RANGE = ("\uA8E0", "\uA8FF")
VEDIC_EXT_RANGE = ("\u1CD0", "\u1CFF")

# Combining signs and marks in Devanagari
DEVANAGARI_VIRAMA = "\u094D"       # ् (Halant / Virama)
DEVANAGARI_NUKTA = "\u093C"        # ़ (Nukta)
DEVANAGARI_DANDA = "\u0964"        # । (Purna Viram)
DEVANAGARI_DOUBLE_DANDA = "\u0965" # ॥ (Dirgha Viram)

# Zero-width formatting characters
ZWJ = "\u200D"   # Zero-Width Joiner
ZWNJ = "\u200C"  # Zero-Width Non-Joiner
BOM = "\uFEFF"   # Byte Order Mark / Zero-Width No-Break Space
ZWSP = "\u200B"  # Zero-Width Space
SOFT_HYPHEN = "\u00AD"

# Invisible and directional formatting characters to remove
INVISIBLE_CHARS_REGEX = re.compile(
    r"[\u00AD\u200B\u200E\u200F\u202A-\u202E\u2060-\u206F\uFEFF]"
)

# Control characters (excluding \n, \t, \r)
CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Apply standard Unicode normalization (NFC, NFD, NFKC, NFKD).
    
    Default is NFC (Canonical Decomposition followed by Canonical Composition),
    which produces stable single-codepoint representations for precomposed glyphs.
    """
    if not text:
        return ""
    if form.upper() not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"Invalid Unicode normalization form: {form}. Choose NFC, NFD, NFKC, or NFKD.")
    return unicodedata.normalize(form.upper(), text)


def remove_control_characters(
    text: str,
    preserve_newlines: bool = True,
    preserve_tabs: bool = True,
) -> str:
    """
    Strip non-printable C0 and C1 control characters.
    Optionally preserves standard whitespace controls (newline, tab).
    """
    if not text:
        return ""
    if preserve_newlines and preserve_tabs:
        return CONTROL_CHARS_REGEX.sub("", text)
    
    # Custom filter if tab or newline shouldn't be preserved
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            if preserve_newlines and ch in ("\n", "\r"):
                cleaned.append(ch)
            elif preserve_tabs and ch == "\t":
                cleaned.append(ch)
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def remove_invisible_characters(text: str) -> str:
    """
    Remove soft hyphens, zero-width spaces, BOMs, and bidirectional control marks.
    Preserves ZWJ and ZWNJ (handled specifically via sanitize_zwj_zwnj).
    """
    if not text:
        return ""
    return INVISIBLE_CHARS_REGEX.sub("", text)


def sanitize_zwj_zwnj(text: str) -> str:
    """
    Sanitize Zero-Width Joiner (ZWJ U+200D) and Zero-Width Non-Joiner (ZWNJ U+200C).
    
    In Devanagari, ZWJ and ZWNJ are valid only when directly following a Virama (Halant)
    to enforce explicit half-letter forms (e.g., र्‍) or prevent conjunct ligature formation.
    Any stray, isolated, or repeated ZWJ/ZWNJ (e.g. at word boundaries, after spaces, or
    adjacent to non-Virama characters) is stripped.
    """
    if not text or (ZWJ not in text and ZWNJ not in text):
        return text

    # Remove consecutive duplicate joiners (e.g., ZWJ+ZWJ -> ZWJ)
    text = re.sub(rf"[{ZWJ}{ZWNJ}]+", lambda m: m.group(0)[0], text)
    
    # Strip leading joiners
    text = text.lstrip(f"{ZWJ}{ZWNJ}")

    # Remove ZWJ/ZWNJ adjacent to whitespace
    text = re.sub(rf"\s+[{ZWJ}{ZWNJ}]+", " ", text)
    text = re.sub(rf"[{ZWJ}{ZWNJ}]+\s+", " ", text)
    
    # Strip ZWJ/ZWNJ when not preceded by Devanagari Virama (Halant \u094D)
    # Valid pattern: [\u094D][\u200D\u200C]
    def _filter_joiner(match):
        preceding_char = match.group(1)
        joiner = match.group(2)
        if preceding_char == DEVANAGARI_VIRAMA:
            return preceding_char + joiner
        return preceding_char

    text = re.sub(rf"(.)([{ZWJ}{ZWNJ}])", _filter_joiner, text)
    # Strip trailing joiners if not preceded by Virama
    text = re.sub(rf"(?<!{DEVANAGARI_VIRAMA})[{ZWJ}{ZWNJ}]+$", "", text)
    return text


def canonicalize_devanagari_nukta(text: str) -> str:
    """
    Ensure Devanagari Nukta (U+093C) is placed immediately after the base consonant.
    If a matra or vowel sign appears before the nukta due to faulty encoding/typing,
    reorder so the nukta immediately attaches to the base consonant.
    """
    if not text or DEVANAGARI_NUKTA not in text:
        return text

    # Swap Matra + Nukta -> Nukta + Matra
    # Devanagari dependent vowel signs: \u093E to \u094C, \u094E, \u094F, \u0955-\u0957, \u0962, \u0963
    matra_nukta_pattern = re.compile(r"([\u093E-\u094C\u0962\u0963])(\u093C)")
    text = matra_nukta_pattern.sub(r"\2\1", text)
    
    # Collapse multiple consecutive nuktas
    text = re.sub(r"\u093C+", "\u093C", text)
    return text


def clean_unicode_pipeline(
    text: str,
    form: str = "NFC",
    strip_controls: bool = True,
    clean_invisibles: bool = True,
    sanitize_joiners: bool = True,
    canonical_nukta: bool = True,
) -> str:
    """
    Run the end-to-end Unicode normalization and cleaning pipeline.
    """
    if not text:
        return ""

    # 1. Base Unicode normalization
    text = normalize_unicode(text, form=form)

    # 2. Control characters
    if strip_controls:
        text = remove_control_characters(text, preserve_newlines=True, preserve_tabs=True)

    # 3. Invisible formatting & bidi controls
    if clean_invisibles:
        text = remove_invisible_characters(text)

    # 4. Devanagari nukta canonicalization
    if canonical_nukta:
        text = canonicalize_devanagari_nukta(text)

    # 5. ZWJ / ZWNJ sanitization
    if sanitize_joiners:
        text = sanitize_zwj_zwnj(text)

    return text


def get_unicode_block_distribution(text: str) -> Dict[str, int]:
    """
    Count characters belonging to major Unicode script blocks in text.
    """
    counts = {
        "Devanagari": 0,
        "Devanagari_Extended": 0,
        "Basic_Latin": 0,
        "Latin_Extended": 0,
        "Digits_ASCII": 0,
        "Digits_Devanagari": 0,
        "General_Punctuation": 0,
        "Whitespace": 0,
        "Other": 0,
    }
    
    for ch in text:
        cp = ord(ch)
        if ch.isspace():
            counts["Whitespace"] += 1
        elif 0x0900 <= cp <= 0x097F:
            if 0x0966 <= cp <= 0x096F:
                counts["Digits_Devanagari"] += 1
            else:
                counts["Devanagari"] += 1
        elif 0xA8E0 <= cp <= 0xA8FF or 0x1CD0 <= cp <= 0x1CFF:
            counts["Devanagari_Extended"] += 1
        elif 0x0030 <= cp <= 0x0039:
            counts["Digits_ASCII"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["Basic_Latin"] += 1
        elif (0x0080 <= cp <= 0x024F):
            counts["Latin_Extended"] += 1
        elif (0x2000 <= cp <= 0x206F) or ch in "।॥.,!?;:\"'-–—()[]{}":
            counts["General_Punctuation"] += 1
        else:
            counts["Other"] += 1

    return counts
