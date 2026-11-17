"""
src/tokenizers.py

Comprehensive tokenization suite for multilingual and Nepali Devanagari datasets:
1. SentenceTokenizer (Devanagari danda, Latin punctuation, abbreviation preservation)
2. WordTokenizer (regex script-aware word boundary tokenization)
3. AksharaTokenizer (linguistic Devanagari syllable / grapheme cluster tokenization)
4. BPETokenizer (trainable subword Byte-Pair Encoding with serialization)
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

# Common Nepali and English abbreviations that end with a period
COMMON_ABBREVIATIONS = {
    "डा", "प्रा", "वि.सं", "इ.सं", "नं", "रु", "लि", "श्री", "प", "मे",
    "dr", "mr", "mrs", "prof", "inc", "ltd", "dept", "co", "etc", "vs", "eg", "ie",
}

# Regex for sentence splitting
# Split on Devanagari danda (।), double danda (॥), question mark (?), exclamation (!),
# or full stop (.) followed by whitespace or end of string.
SENTENCE_SPLIT_REGEX = re.compile(r"([।॥!?\n]+|(?<=[^\s\d])\.(?=\s+|$))")

# Devanagari Unicode character classifications
DEVA_INDEPENDENT_VOWELS = r"[\u0904-\u0914\u0960\u0961\u0972-\u0977]"
DEVA_CONSONANTS = r"[\u0915-\u0939\u0958-\u095F\u0979-\u097F]"
DEVA_NUKTA = r"\u093C"
DEVA_VIRAMA = r"\u094D"
DEVA_MATRAS = r"[\u093E-\u094C\u094E\u094F\u0955-\u0957\u0962\u0963]"
DEVA_MODIFIERS = r"[\u0901-\u0903\u093B]"  # Candrabindu, Anusvara, Visarga
JOINERS = r"[\u200C\u200D]"

# Orthographic syllable (Akshara) regex pattern
AKSHARA_PATTERN = re.compile(
    # 1. Consonant cluster: (C + [Nukta] + Virama + [Joiner])* + C + [Nukta] + [Matra] + [Modifiers]
    rf"(?:{DEVA_CONSONANTS}{DEVA_NUKTA}?{DEVA_VIRAMA}{JOINERS}?)*"
    rf"{DEVA_CONSONANTS}{DEVA_NUKTA}?{DEVA_MATRAS}*{DEVA_MODIFIERS}*"
    r"|"
    # 2. Independent vowel + [Modifiers]
    rf"{DEVA_INDEPENDENT_VOWELS}{DEVA_NUKTA}?{DEVA_MODIFIERS}*"
    r"|"
    # 3. Devanagari numbers
    r"[\u0966-\u096F]+"
    r"|"
    # 4. English / Latin alphanumeric tokens
    r"[a-zA-Z0-9]+"
    r"|"
    # 5. Non-whitespace punctuation or symbols
    r"\S"
)

# Word tokenizer regex: groups Devanagari words, Latin words, numbers, and individual punctuation marks
WORD_TOKEN_REGEX = re.compile(
    r"[\u0900-\u097F]+|[a-zA-Z]+|\d+|[^\w\s]"
)


class SentenceTokenizer:
    """
    Sentence segmenter for Nepali Devanagari and multilingual text.
    Respects Devanagari danda (।), double danda (॥), punctuation, and abbreviations.
    """

    def __init__(self, abbreviations: Optional[Set[str]] = None):
        self.abbreviations = abbreviations if abbreviations is not None else COMMON_ABBREVIATIONS

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        # Replace known abbreviations temporarily to avoid splitting
        masked_text = text
        substitutions = {}
        for idx, abbr in enumerate(self.abbreviations):
            pattern = re.compile(rf"\b{re.escape(abbr)}\.", re.IGNORECASE)
            placeholder = f"__ABBR_{idx}__"
            if pattern.search(masked_text):
                substitutions[placeholder] = abbr + "."
                masked_text = pattern.sub(placeholder, masked_text)

        # Split on sentence boundaries
        parts = SENTENCE_SPLIT_REGEX.split(masked_text)
        sentences = []
        current = ""

        for part in parts:
            if not part:
                continue
            if SENTENCE_SPLIT_REGEX.fullmatch(part):
                current += part
                sent = current.strip()
                if sent:
                    sentences.append(sent)
                current = ""
            else:
                current += part

        if current.strip():
            sentences.append(current.strip())

        # Restore abbreviations
        final_sentences = []
        for s in sentences:
            for placeholder, original in substitutions.items():
                s = s.replace(placeholder, original)
            if s.strip():
                final_sentences.append(s.strip())

        return final_sentences


class WordTokenizer:
    """
    Script-aware word and punctuation tokenizer for Devanagari and Latin text.
    """

    def __init__(self, preserve_punctuation: bool = True):
        self.preserve_punctuation = preserve_punctuation

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = WORD_TOKEN_REGEX.findall(text)
        if not self.preserve_punctuation:
            tokens = [t for t in tokens if re.search(r"[\w\u0900-\u097F]", t)]
        return tokens


class AksharaTokenizer:
    """
    Devanagari orthographic syllable (Akshara) and grapheme cluster tokenizer.
    Decomposes text into linguistically sound Indic syllables rather than arbitrary Unicode code points.
    """

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [match.group(0) for match in AKSHARA_PATTERN.finditer(text)]


class BPETokenizer:
    """
    Lightweight, self-contained Byte-Pair Encoding (BPE) subword tokenizer.
    Supports training from corpus, subword encoding, decoding, and JSON persistence.
    """

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._cache: Dict[str, List[str]] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self._cache.clear()

    def _get_stats(self, splits: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Compute frequency of adjacent symbol pairs across all split words."""
        pairs = defaultdict(int)
        for word_tuple, freq in splits.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self, pair: Tuple[str, str], splits: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of the specified pair in current vocabulary splits."""
        first, second = pair
        new_splits = {}
        merged_symbol = first + second

        for word_tuple, freq in splits.items():
            if first not in word_tuple or second not in word_tuple:
                new_splits[word_tuple] = freq
                continue

            new_word = []
            i = 0
            n = len(word_tuple)
            while i < n:
                if i < n - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_word.append(merged_symbol)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_splits[tuple(new_word)] = freq
        return new_splits

    def train(
        self,
        texts: Iterable[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        max_corpus_words: int = 20000,
    ):
        """
        Train BPE merge rules on an iterable of raw or preprocessed strings.
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        self._init_special_tokens()
        self.merges = []

        # 1. Collect word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.strip().split()
            word_counts.update(words)

        # 2. Represent words as tuples of characters with boundary marker </w>
        # Limit to top most frequent words for fast and effective convergence
        splits: Dict[Tuple[str, ...], int] = {}
        for word, count in word_counts.most_common(max_corpus_words):
            if count < min_frequency:
                break
            splits[tuple(list(word) + ["</w>"])] = count

        if not splits:
            return

        # 3. Add initial character vocabulary
        char_set = set()
        for word_tuple in splits.keys():
            char_set.update(word_tuple)
        for ch in sorted(char_set):
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inverse_vocab[idx] = ch

        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            return

        # 4. Iterative BPE merges
        for _ in range(num_merges):
            pairs = self._get_stats(splits)
            if not pairs:
                break
            best_pair, best_freq = max(pairs.items(), key=lambda x: x[1])
            if best_freq < min_frequency:
                break

            splits = self._merge_vocab(best_pair, splits)
            self.merges.append(best_pair)
            new_token = "".join(best_pair)
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.inverse_vocab[idx] = new_token

    def tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word into learned subwords using the merge table."""
        if word in self._cache:
            return self._cache[word]

        symbols = list(word) + ["</w>"]
        if len(symbols) <= 1:
            self._cache[word] = symbols
            return symbols

        for pair in self.merges:
            first, second = pair
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break

        self._cache[word] = symbols
        return symbols

    def encode(self, text: str) -> List[str]:
        """Convert input text into subword token strings."""
        if not text:
            return []
        tokens = []
        for word in text.strip().split():
            tokens.extend(self.tokenize_word(word))
        return tokens

    def encode_to_ids(self, text: str) -> List[int]:
        """Convert input text into token integer IDs."""
        tokens = self.encode(text)
        unk_id = self.vocab.get("<unk>", 1)
        return [self.vocab.get(tok, unk_id) for tok in tokens]

    def decode(self, tokens: Union[List[str], List[int]]) -> str:
        """Decode subword tokens or token IDs back into normalized text."""
        if not tokens:
            return ""

        # Convert IDs to string tokens if needed
        if tokens and isinstance(tokens[0], int):
            str_tokens = [self.inverse_vocab.get(tid, "<unk>") for tid in tokens]
        else:
            str_tokens = list(tokens)

        # Filter special tokens
        filtered = [t for t in str_tokens if t not in self.SPECIAL_TOKENS]
        joined = "".join(filtered)
        text = joined.replace("</w>", " ")
        return text.strip()

    def save(self, filepath: str):
        """Save BPE vocabulary and merge table to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [list(p) for p in self.merges],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load BPE tokenizer model from a saved JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data.get("vocab_size", 4000))
        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(p) for p in data["merges"]]
        return tokenizer
