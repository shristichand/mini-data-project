"""
tests/test_text_pipeline.py

Comprehensive unit and integration test suite for:
- Unicode Normalization, ZWJ/ZWNJ sanitization, and Nukta canonicalization
- Text sanitization, HTML stripping, punctuation, and digit conversion
- Language profiling, script distributions, TTR, and quality scoring
- Sentence, Word, Akshara, and BPE subword tokenizers
- End-to-end TextDatasetPipeline execution and DataFrame filtering
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add project root, src, and scripts to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from unicode_utils import (
    clean_unicode_pipeline,
    normalize_unicode,
    remove_control_characters,
    remove_invisible_characters,
    sanitize_zwj_zwnj,
    canonicalize_devanagari_nukta,
    get_unicode_block_distribution,
    ZWJ,
    ZWNJ,
    BOM,
    ZWSP,
)
from text_normalizer import (
    clean_text_pipeline,
    strip_html,
    normalize_whitespace,
    normalize_punctuation,
    convert_digits,
    mask_urls_and_emails,
)
from language_profiler import (
    devanagari_ratio,
    latin_ratio,
    digit_ratio,
    punct_ratio,
    type_token_ratio,
    stopword_ratio,
    ngram_repetition_ratio,
    calculate_quality_score,
    profile_text,
    NEPALI_STOPWORDS,
)
from tokenizers import (
    SentenceTokenizer,
    WordTokenizer,
    AksharaTokenizer,
    BPETokenizer,
)
from text_pipeline import (
    PipelineConfig,
    PipelineResult,
    TextDatasetPipeline,
)


class TestUnicodeUtils(unittest.TestCase):
    """Test suite for Unicode normalization and script sanitization."""

    def test_unicode_normalization_forms(self):
        # Decomposed vs Composed form
        # 'é' as e + combining acute accent (\u0065\u0301) vs \u00E9
        decomposed = "e\u0301"
        composed = "\u00e9"
        self.assertEqual(normalize_unicode(decomposed, form="NFC"), composed)
        self.assertEqual(normalize_unicode(composed, form="NFD"), decomposed)

    def test_remove_control_characters(self):
        # Text containing null bytes, bell, escape characters
        dirty = "नमस्ते\x00 संसार\x07!\nनयाँ\tलाइन\x1b"
        cleaned = remove_control_characters(dirty, preserve_newlines=True, preserve_tabs=True)
        self.assertEqual(cleaned, "नमस्ते संसार!\nनयाँ\tलाइन")

    def test_remove_invisible_characters(self):
        # Text containing zero-width spaces, BOM, and soft hyphens
        dirty = f"{BOM}नेपाल{ZWSP}सुन्दर{chr(0x00AD)}छ"
        cleaned = remove_invisible_characters(dirty)
        self.assertEqual(cleaned, "नेपालसुन्दरछ")

    def test_sanitize_zwj_zwnj(self):
        # Stray ZWJ at edges or around spaces should be removed
        stray = f"{ZWJ}काठमाडौँ{ZWNJ} र {ZWJ}पोखरा"
        cleaned = sanitize_zwj_zwnj(stray)
        self.assertNotIn(ZWJ, cleaned)
        self.assertNotIn(ZWNJ, cleaned)

        # Valid halant + ZWJ for eyelash-Reph (र्‍) should be preserved
        eyelash_reph = f"र\u094D{ZWJ}"
        preserved = sanitize_zwj_zwnj(eyelash_reph)
        self.assertEqual(preserved, eyelash_reph)

    def test_canonicalize_devanagari_nukta(self):
        # Matra placed before nukta erroneously should be reordered: Nukta before Matra
        # \u0915 (क) + \u093E (ा) + \u093C (़) -> \u0915 (क) + \u093C (़) + \u093E (ा)
        malformed = "\u0915\u093E\u093C"
        corrected = canonicalize_devanagari_nukta(malformed)
        self.assertEqual(corrected, "\u0915\u093C\u093E")

    def test_unicode_block_distribution(self):
        text = "नेपाल Nepal 123 ।"
        dist = get_unicode_block_distribution(text)
        self.assertGreater(dist["Devanagari"], 0)
        self.assertGreater(dist["Basic_Latin"], 0)
        self.assertGreater(dist["Digits_ASCII"], 0)
        self.assertGreater(dist["Whitespace"], 0)


class TestTextNormalizer(unittest.TestCase):
    """Test suite for text cleaning, stripping, and normalization."""

    def test_strip_html(self):
        html_doc = "<p>काठमाडौँ &amp; ललितपुर</p><br/><div>नेपाल</div>"
        cleaned = strip_html(html_doc)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<div>", cleaned)
        self.assertIn("काठमाडौँ & ललितपुर", cleaned)

    def test_mask_urls_and_emails(self):
        text = "हाम्रो वेबसाइट https://example.com/nepal र ईमेल info@nepal.org.np हो।"
        cleaned = mask_urls_and_emails(text, replace_url="", replace_email="")
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("info@nepal.org.np", cleaned)

    def test_normalize_whitespace(self):
        text = "   काठमाडौँ    नेपालको   \n\n\n\n   राजधानी   हो।   "
        normalized = normalize_whitespace(text, max_consecutive_newlines=2)
        self.assertEqual(normalized, "काठमाडौँ नेपालको\n\nराजधानी हो।")

    def test_normalize_punctuation(self):
        text = "‘नेपाल’ — “सुन्दर देश” ... यो राम्रो छ |"
        normalized = normalize_punctuation(text)
        self.assertIn("'नेपाल'", normalized)
        self.assertIn('"सुन्दर देश"', normalized)
        self.assertIn("-", normalized)
        self.assertIn("।", normalized)
        self.assertNotIn("|", normalized)

    def test_convert_digits(self):
        deva_nums = "०१२३४५६७८९"
        ascii_nums = convert_digits(deva_nums, to_ascii=True)
        self.assertEqual(ascii_nums, "0123456789")
        back_to_deva = convert_digits(ascii_nums, to_ascii=False)
        self.assertEqual(back_to_deva, deva_nums)


class TestLanguageProfiler(unittest.TestCase):
    """Test suite for language characteristics, TTR, and quality scoring."""

    def test_script_ratios(self):
        nepali_text = "नेपाल एक सुन्दर देश हो।"
        self.assertGreater(devanagari_ratio(nepali_text), 0.70)
        self.assertEqual(latin_ratio(nepali_text), 0.0)

        mixed_text = "Nepal नेपाल 123"
        self.assertGreater(latin_ratio(mixed_text), 0.20)
        self.assertGreater(devanagari_ratio(mixed_text), 0.20)
        self.assertGreater(digit_ratio(mixed_text), 0.10)

    def test_type_token_ratio(self):
        diverse_text = "नेपाल एक सुन्दर हिमाल पहाड तराई देश हो"
        repetitive_text = "नेपाल नेपाल नेपाल नेपाल नेपाल"
        self.assertEqual(type_token_ratio(diverse_text), 1.0)
        self.assertEqual(type_token_ratio(repetitive_text), 0.2)

    def test_stopword_ratio(self):
        grammatical = "यो नेपालको संविधान हो र यसमा धेरै कुरा छन्।"
        sw_ratio = stopword_ratio(grammatical)
        self.assertGreater(sw_ratio, 0.20)

    def test_ngram_repetition_ratio(self):
        normal = "नेपाल एक दक्षिण एसियाली भूपरिवेष्ठित राष्ट्र हो।"
        looping = "सुन्दर देश सुन्दर देश सुन्दर देश सुन्दर देश सुन्दर देश"
        self.assertEqual(ngram_repetition_ratio(normal, n=2), 0.0)
        self.assertGreater(ngram_repetition_ratio(looping, n=2), 0.5)

    def test_quality_score(self):
        high_quality = (
            "नेपाल एक दक्षिण एसियाली भूपरिवेष्ठित राष्ट्र हो। "
            "यसको उत्तरमा चीन र दक्षिण, पूर्व तथा पश्चिममा भारत पर्छन्। "
            "नेपालको राजधानी काठमाडौँ हो। यहाँ विश्वको सर्वोच्च शिखर सगरमाथा अवस्थित छ।"
        )
        score_high = calculate_quality_score(high_quality)
        self.assertGreater(score_high, 0.70)

        junk_text = "!@#$%^&*() 1234567890 ??? ???"
        score_low = calculate_quality_score(junk_text)
        self.assertLess(score_low, 0.20)


class TestTokenizers(unittest.TestCase):
    """Test suite for Sentence, Word, Akshara, and BPE tokenizers."""

    def test_sentence_tokenizer_with_danda_and_abbr(self):
        tokenizer = SentenceTokenizer()
        text = "डा. केसीले अनशन सुरु गरे। यो पहिलो पटक होइन! के यो समाधान होला? वि.सं. २०८० मा यस्तो भयो।"
        sentences = tokenizer.tokenize(text)
        self.assertEqual(len(sentences), 4)
        # Verify abbreviations were not broken as sentence splits
        self.assertIn("डा. केसीले अनशन सुरु गरे।", sentences[0])
        self.assertIn("वि.सं. २०८० मा यस्तो भयो।", sentences[3])

    def test_word_tokenizer(self):
        tokenizer = WordTokenizer()
        text = "नेपाल, विश्वको १ नम्बर सुन्दर देश हो!"
        tokens = tokenizer.tokenize(text)
        self.assertIn("नेपाल", tokens)
        self.assertIn(",", tokens)
        self.assertIn("१", tokens)
        self.assertIn("सुन्दर", tokens)
        self.assertIn("!", tokens)

    def test_akshara_tokenizer_grapheme_clusters(self):
        tokenizer = AksharaTokenizer()
        # Words containing conjuncts, matras, and modifiers
        # 'नेपाल' -> ['ने', 'पा', 'ल']
        # 'स्वास्थ्य' -> ['स्वा', 'स्थ्य']
        tokens_nepal = tokenizer.tokenize("नेपाल")
        self.assertEqual(tokens_nepal, ["ने", "पा", "ल"])

        tokens_swasthya = tokenizer.tokenize("स्वास्थ्य")
        self.assertEqual(tokens_swasthya, ["स्वा", "स्थ्य"])

    def test_bpe_tokenizer_train_and_inference(self):
        tokenizer = BPETokenizer(vocab_size=150)
        corpus = [
            "नेपाल एक सुन्दर देश हो",
            "नेपाली भाषा र नेपालको कला संस्कृति",
            "काठमाडौँ नेपालको ठूलो सहर हो",
            "नेपालका हिमाल र पहाड",
        ] * 10
        tokenizer.train(corpus, vocab_size=150, min_frequency=2)
        self.assertGreater(len(tokenizer.vocab), 20)

        # Test encoding and decoding
        test_sent = "नेपाल एक सुन्दर सहर हो"
        tokens = tokenizer.encode(test_sent)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        decoded = tokenizer.decode(tokens)
        self.assertEqual(decoded, test_sent)

        # Test persistence
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tokenizer.save(tmp_path)
            loaded = BPETokenizer.load(tmp_path)
            self.assertEqual(len(loaded.vocab), len(tokenizer.vocab))
            loaded_tokens = loaded.encode(test_sent)
            self.assertEqual(loaded_tokens, tokens)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestTextDatasetPipeline(unittest.TestCase):
    """Test suite for integrated TextDatasetPipeline."""

    def setUp(self):
        self.config = PipelineConfig(
            unicode_form="NFC",
            strip_controls=True,
            clean_invisibles=True,
            sanitize_joiners=True,
            strip_html_tags=True,
            normalize_whitespace=True,
            normalize_punct=True,
            min_char_len=20,
            max_char_len=1000,
            min_deva_ratio=0.50,
            tokenization_mode="word",
        )
        self.pipeline = TextDatasetPipeline(config=self.config)

    def test_pipeline_process_valid_text(self):
        text = "<p><b>नेपाल</b> एक सुन्दर देश हो।</p>"
        result = self.pipeline.process_text(text)
        self.assertTrue(result.passed)
        self.assertNotIn("<p>", result.cleaned_text)
        self.assertIn("नेपाल", result.cleaned_text)
        self.assertGreater(len(result.tokens), 0)
        self.assertEqual(len(result.rejection_reasons), 0)

    def test_pipeline_filter_rejections(self):
        # 1. Below min char len
        short_res = self.pipeline.process_text("छोटो")
        self.assertFalse(short_res.passed)
        self.assertTrue(any("below_min_char_len" in r for r in short_res.rejection_reasons))

        # 2. Low Devanagari ratio (English text)
        eng_res = self.pipeline.process_text("This is an english document without devanagari text.")
        self.assertFalse(eng_res.passed)
        self.assertTrue(any("low_devanagari_ratio" in r for r in eng_res.rejection_reasons))

    def test_pipeline_dataframe_processing(self):
        df = pd.DataFrame({
            "text": [
                "नेपाल एक दक्षिण एसियाली भूपरिवेष्ठित राष्ट्र हो। यहाँ विश्वको सर्वोच्च शिखर सगरमाथा छ।",
                "Hi",  # Too short
                "English text only that should be rejected by devanagari ratio filter.",
                "काठमाडौँ, ललितपुर र भक्तपुर उपत्यकाका तीन प्रमुख जिल्ला हुन्। यहाँ धेरै ऐतिहासिक सम्पदा छन्।",
            ]
        })
        processed_df, report = self.pipeline.process_dataframe(df, text_column="text", filter_rejected=True)
        self.assertEqual(report["initial_records"], 4)
        self.assertEqual(report["retained_records"], 2)
        self.assertEqual(report["dropped_records"], 2)
        self.assertEqual(len(processed_df), 2)
        self.assertIn("tokens", processed_df.columns)
        self.assertIn("quality_score", processed_df.columns)


class TestFineWeb2PipelineIntegration(unittest.TestCase):
    """Test suite for direct connection between FineWeb-2 dataset and TextDatasetPipeline."""

    def test_fineweb2_npi_score_filter(self):
        config = PipelineConfig(
            min_char_len=20,
            min_deva_ratio=0.50,
            min_npi_score=0.95,
            tokenization_mode="word",
        )
        pipeline = TextDatasetPipeline(config=config)

        # DataFrame simulating FineWeb-2 schema
        df = pd.DataFrame({
            "text": [
                "नेपाल एक दक्षिण एसियाली भूपरिवेष्ठित राष्ट्र हो। यहाँ धेरै हिमाल छन्।",
                "नेपालको राजधानी काठमाडौँ हो। यहाँ थुप्रै ऐतिहासिक सम्पदाहरू छन्।",
            ],
            "id": ["urn:uuid:1", "urn:uuid:2"],
            "language": ["npi", "npi"],
            "language_score": [0.999, 0.850],  # 2nd is below 0.95 threshold
            "top_langs": [
                '{"npi_Deva_score": 0.999}',
                '{"npi_Deva_score": 0.850}',
            ],
        })

        processed_df, report = pipeline.process_dataframe(df, text_column="text", filter_rejected=True)
        self.assertEqual(report["initial_records"], 2)
        self.assertEqual(report["retained_records"], 1)
        self.assertEqual(report["dropped_records"], 1)
        self.assertIn("low_npi_score", report["rejection_breakdown"])
        self.assertEqual(processed_df["id"].iloc[0], "urn:uuid:1")

    def test_download_and_sample_pipeline_bridge(self):
        from download_and_sample import (
            SAMPLE_PARQUET_PATH,
            get_or_create_fineweb2_sample,
            inspect_schema,
            process_fineweb2_sample_with_pipeline,
        )

        sample_path = get_or_create_fineweb2_sample()
        self.assertTrue(os.path.exists(sample_path))

        schema, num_rows = inspect_schema(sample_path)
        self.assertGreater(num_rows, 0)
        self.assertIn("text", schema.names)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_parquet = os.path.join(tmpdir, "test_clean.parquet")
            out_report = os.path.join(tmpdir, "test_report.json")

            processed_df, report = process_fineweb2_sample_with_pipeline(
                sample_path=sample_path,
                output_path=out_parquet,
                report_path=out_report,
                tokenizer_mode="word",
                limit=50,
            )

            self.assertTrue(os.path.exists(out_parquet))
            self.assertTrue(os.path.exists(out_report))
            self.assertGreater(len(processed_df), 0)
            self.assertIn("tokens", processed_df.columns)
            self.assertIn("cleaned_text", processed_df.columns)
            self.assertIn("quality_score", processed_df.columns)


if __name__ == "__main__":
    unittest.main()
