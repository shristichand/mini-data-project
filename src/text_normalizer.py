"""
src/text_normalizer.py

Text cleaning, sanitation, and normalization transformations.
Designed for preprocessing multilingual and Nepali Devanagari web-scale text.
"""

import html
import re
from typing import Optional

# Regex patterns
HTML_TAGS_REGEX = re.compile(r"<[^>]+>")
URL_REGEX = re.compile(
    r"(?:https?:\/\/|www\.)[^\s/$.?#].[^\s]*",
    re.IGNORECASE,
)
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    re.IGNORECASE,
)
MULTIPLE_SPACES_REGEX = re.compile(r"[^\S\r\n]+")
MULTIPLE_NEWLINES_REGEX = re.compile(r"\n{3,}")

# Quotes and dashes mapping
QUOTE_MAP = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "«": '"',
    "»": '"',
}

DASH_MAP = {
    "—": "-",  # em dash
    "–": "-",  # en dash
    "―": "-",  # horizontal bar
    "‒": "-",  # figure dash
    "−": "-",  # minus sign
}

# Devanagari to ASCII digit mappings
DEVA_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ASCII_TO_DEVA_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def strip_html(text: str, unescape_entities: bool = True) -> str:
    """
    Remove HTML tags and optionally unescape HTML entities (&amp; -> &, etc.).
    """
    if not text:
        return ""
    if unescape_entities:
        text = html.unescape(text)
    return HTML_TAGS_REGEX.sub(" ", text)


def mask_urls_and_emails(
    text: str,
    replace_url: str = "",
    replace_email: str = "",
) -> str:
    """
    Mask or remove web URLs and email addresses.
    """
    if not text:
        return ""
    if replace_url is not None:
        text = URL_REGEX.sub(replace_url, text)
    if replace_email is not None:
        text = EMAIL_REGEX.sub(replace_email, text)
    return text


def normalize_whitespace(text: str, max_consecutive_newlines: int = 2) -> str:
    """
    Normalize whitespace:
    - Replace Windows/Mac carriage returns with standard newline \n
    - Collapse multiple horizontal spaces/tabs into a single space
    - Strip leading/trailing spaces on each line
    - Limit consecutive newlines to max_consecutive_newlines
    """
    if not text:
        return ""
    # Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Process line by line to preserve line structure
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = MULTIPLE_SPACES_REGEX.sub(" ", line).strip()
        cleaned_lines.append(cleaned_line)
    
    rebuilt = "\n".join(cleaned_lines)
    
    # Limit consecutive newlines
    if max_consecutive_newlines > 0:
        replacement = "\n" * max_consecutive_newlines
        rebuilt = re.sub(rf"\n{{{max_consecutive_newlines + 1},}}", replacement, rebuilt)
    
    return rebuilt.strip()


def normalize_punctuation(
    text: str,
    standardize_quotes: bool = True,
    standardize_dashes: bool = True,
    standardize_danda: bool = True,
    normalize_ellipses: bool = True,
) -> str:
    """
    Standardize typography:
    - Map curly/typographic quotes to standard ASCII quotes
    - Map typographic dashes (em, en) to standard hyphen
    - Standardize Nepali punctuation: convert pipe '|' to Devanagari Danda '।',
      and double pipes or repeated dandas '।।' to double danda '॥'
    - Normalize horizontal ellipses ('…', '.....') to '...'
    """
    if not text:
        return ""

    if standardize_quotes:
        for k, v in QUOTE_MAP.items():
            text = text.replace(k, v)

    if standardize_dashes:
        for k, v in DASH_MAP.items():
            text = text.replace(k, v)

    if normalize_ellipses:
        text = text.replace("…", "...")
        text = re.sub(r"\.{4,}", "...", text)

    if standardize_danda:
        # Normalize double danda: two dandas or two pipes
        text = re.sub(r"[।|]{2,}", "॥", text)
        # Normalize single pipe used as sentence terminator to Devanagari Purna Viram (danda)
        text = re.sub(r"(?<=\S)\s*\|\s*", " । ", text)
        # Ensure single space after danda if followed by non-whitespace
        text = re.sub(r"([।॥])([^\s।॥0-9])", r"\1 \2", text)

    return text


def convert_digits(text: str, to_ascii: bool = True) -> str:
    """
    Convert numerals between Devanagari (०-९) and ASCII (0-9).
    """
    if not text:
        return ""
    if to_ascii:
        return text.translate(DEVA_TO_ASCII_DIGITS)
    return text.translate(ASCII_TO_DEVA_DIGITS)


def case_fold(text: str, lower: bool = True) -> str:
    """
    Case-fold text. Only affects scripts with casing (Latin, Cyrillic, etc.);
    Devanagari remains completely unchanged.
    """
    if not text:
        return ""
    return text.lower() if lower else text


def clean_text_pipeline(
    text: str,
    strip_html_tags: bool = True,
    clean_urls: bool = False,
    clean_emails: bool = False,
    normalize_spaces: bool = True,
    normalize_punct: bool = True,
    ascii_digits: bool = False,
    lowercase: bool = False,
) -> str:
    """
    Unified text normalization pipeline combining all text cleaning steps.
    """
    if not text:
        return ""

    if strip_html_tags:
        text = strip_html(text)

    if clean_urls or clean_emails:
        text = mask_urls_and_emails(
            text,
            replace_url="" if clean_urls else None,
            replace_email="" if clean_emails else None,
        )

    if normalize_punct:
        text = normalize_punctuation(text)

    if ascii_digits:
        text = convert_digits(text, to_ascii=True)

    if lowercase:
        text = case_fold(text, lower=True)

    if normalize_spaces:
        text = normalize_whitespace(text)

    return text
