# 📘 Beginner & Learner Guide to Mini Data Project

Welcome! If you are learning data engineering and NLP (Natural Language Processing), this guide is written especially for you.

---

## 🎯 1. What does this project do? (In 30 Seconds)

Imagine downloading 50,000 pages of text from Nepali websites.
Web data is **very messy**:
- It contains broken HTML tags (`<div>`, `<p>`).
- It has English ads, gibberish numbers, and emojis.
- Characters from different keyboards can look identical on screen but be completely different symbols to a computer.
- It's too big to open in Excel without crashing your computer!

This project solves all of that in **4 simple steps**:
1. **Download**: Pulls raw Nepali articles from the web (FineWeb-2 dataset).
2. **Profile & Clean**: Finds junk records, cleans weird characters, and saves only good Nepali sentences.
3. **Sample**: Takes smart, representative samples so you can work on a normal laptop without running out of RAM memory.
4. **Tokenize**: Cuts text into "tokens" (words, letters, or subwords) so AI models can read them.

---

## 📂 2. Folder Structure Made Simple

Everything is organized into **3 clear concepts**:

```
mini-data-project/
│
├── src/mini_data_project/          ← The actual Python functions (the "engine")
│   ├── download/                   ← Functions to download data from the web
│   │   └── download_and_sample.py
│   ├── sampling/                   ← Functions to take smart samples from big files
│   │   └── sampler.py
│   └── text/                       ← Functions to clean, profile, and tokenize text
│       ├── unicode_utils.py        ← Fixes Devanagari characters & invisible symbols
│       ├── text_normalizer.py      ← Strips HTML tags, trims whitespace, fixes punctuation
│       ├── language_profiler.py    ← Scores text quality (% Devanagari vs English)
│       ├── tokenizers.py           ← Splits text into words, sentences, or subwords (BPE)
│       ├── text_pipeline.py        ← Runs all cleaning and tokenizing in sequence
│       └── text_utils.py           ← Helper tools (gets website domain names, word counts)
│
├── scripts/                        ← Click-to-run scripts (the "steering wheel")
│   ├── download/
│   │   └── run_download.py         ← Run this to download the dataset
│   ├── sampling/
│   │   └── eval_sampling.py        ← Run this to test sampling methods
│   └── text/
│       ├── profile.py              ← Run this to check how dirty the data is
│       ├── visualize.py            ← Run this to generate nice visual charts
│       ├── clean.py                ← Run this to remove junk records
│       └── run_pipeline.py         ← Run this to tokenize and prepare data for AI
│
├── raw/                            ← Where raw downloaded files are kept
├── processed/                      ← Where cleaned & tokenized output files are saved
├── reports/                        ← Where charts (.png) and audit summaries are saved
└── tests/                          ← Automated checks ensuring math & code work correctly
```

---

## 🚀 3. How to Run the Project (Step-by-Step)

Open your terminal in the project folder and follow these steps in order:

### Step 1: Download the Raw Data
```bash
python scripts/download/run_download.py
```
*What happens:* Downloads a sample of 10,000 raw Nepali articles and saves it in `raw/sample_10k.parquet`.

### Step 2: Audit & Profile the Data
```bash
python scripts/text/profile.py
```
*What happens:* Scans the articles and prints a report on your screen showing:
- Missing values or blank rows
- Duplicate articles
- How much text is Nepali Devanagari vs Latin English
- Which websites the data came from (e.g. news sites, blogs)

### Step 3: Draw Helpful Charts
```bash
python scripts/text/visualize.py
```
*What happens:* Creates 4 visual charts in `reports/figures/`:
- Document length distribution
- Language confidence scores
- Top 15 website domains
- Categories of suspicious records

### Step 4: Clean the Data
```bash
python scripts/text/clean.py
```
*What happens:* Removes bad rows (empty texts, machine-translated junk, extreme duplicates) and saves clean data to `processed/clean.parquet`.

### Step 5: Test Sampling Methods
```bash
python scripts/sampling/eval_sampling.py
```
*What happens:* Tests smart sampling algorithms (like **Reservoir Sampling**, which lets you sample a huge file without loading the whole thing into memory).

### Step 6: Tokenize for AI
```bash
python scripts/text/run_pipeline.py --tokenizer word
```
*What happens:* Converts sentences into word tokens and saves the final AI-ready dataset in `processed/pipeline_clean_tokenized.parquet`.

---

## 🧠 4. Beginner Glossary (Common Questions)

### What is a "Parquet" file? (`.parquet`)
Think of a Parquet file like a compressed, super-fast CSV or Excel spreadsheet. It stores columns separately, meaning your computer only reads the data it needs instead of loading everything.

### What is "Sampling"?
Instead of training a model on 500 Gigabytes of text, we take a smaller sample (e.g., 10,000 rows). Good sampling ensures the small sample has the same variety of topics as the huge dataset.

### What is "Tokenization"?
Computers don't read words like humans do. Tokenization breaks a sentence into pieces:
- Sentence: `"नेपाल एक सुन्दर देश हो ।"`
- Word Tokens: `["नेपाल", "एक", "सुन्दर", "देश", "हो", "।"]`

### What is "Unicode Normalization"?
In Devanagari, characters like **'कि'** can be written in two ways in computer bytes:
1. Letter **क** + vowel sign **ि**
2. A single precomposed character
Unicode normalization converts all representations into one standard format so searches and AI models don't get confused.

---

## 🧪 5. How to Check If Everything Is Working

Run the test suite at any time:
```bash
pytest -v
```
All **36 tests** will run and should show `PASSED` in green!
