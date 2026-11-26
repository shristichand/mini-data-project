# FineWeb-2 Nepali Devanagari (`npi_Deva`) Data Engineering & Sampling Suite

A modular, clean data engineering, profiling, sampling, and text preprocessing suite for the Nepali Devanagari (`npi_Deva`) subset of Hugging Face's [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) dataset.

---

## 1. Project Overview

This repository provides an end-to-end data pipeline designed for low-resource NLP research and model pretraining:
1. **Remote Shard Discovery**: Introspects Hugging Face dataset repositories using `HfFileSystem` without downloading large archives blindly.
2. **Low-Memory Inspection**: Reads Parquet footers and Arrow metadata with zero payload memory overhead.
3. **Data Quality Profiling**: Audits missingness, duplicate content, length distributions, linguistic anomalies, language classification confidence, and domain origins.
4. **Declarative Cleaning Pipeline**: Sequentially prunes low-quality records and logs per-step attrition metrics to `processed/cleaning_log.json`.
5. **Batch Processing & Sampling Suite**:
   - Generator-based low-memory batch reader with projection pushdown (`iter_batches`).
   - In-memory random sampling (`random_sample_inmemory`).
   - Bounded-memory single-pass reservoir sampling (`reservoir_sample`, $O(n + B)$ RAM).
   - Proportional stratified sampling with exact quota balancing (`stratified_sample`).
   - Equal-allocation stratified sampling with tiny-strata redistribution (`stratified_equal`).
   - Reproducibility manifests recording canonical row-level SHA-256 and physical file hashes (`<name>.manifest.json`).
6. **Linguistic Preprocessing & Tokenization Engine**:
   - Devanagari Unicode normalization (NFC/NFKC, ZWJ/ZWNJ sanitization, Nukta canonicalization).
   - Sentence, Word, Akshara (grapheme cluster), and Byte-Pair Encoding (BPE) subword tokenizers.
   - Quality scoring, script ratio analysis, and stopword density profiling.

---

## 2. Clean Repository Architecture

```
mini-data-project/
├── pyproject.toml                      # Modern Python packaging & dependencies (uv-compatible)
├── requirements.txt                    # Pinned dependencies for pip installations
├── uv.lock                             # Locked dependency graph
├── README.md                           # Master project guide & architecture documentation
│
├── src/                                # Core package source (Clean domain src-layout)
│   └── mini_data_project/
│       ├── __init__.py                 # Clean public package API & domain exports
│       ├── cli.py                      # Unified CLI entry point (`mini-data-project`)
│       ├── config.py                   # Canonical path & shard configuration
│       │
│       ├── download/                   # [Download Domain] Ingestion & remote shard access
│       │   ├── __init__.py
│       │   └── download_and_sample.py  # Shard discovery, download & initial 10k sampling
│       │
│       ├── sampling/                   # [Sampling Domain] Batch streaming, sampling & manifests
│       │   ├── __init__.py
│       │   └── sampler.py              # Batch reader, reservoir, stratified & manifests
│       │
│       └── text/                       # [Text Domain] Linguistic preprocessing, tokenizers & pipeline
│           ├── __init__.py
│           ├── unicode_utils.py        # NFC/NFKC, ZWJ/ZWNJ sanitization, Nukta canonicalization
│           ├── text_normalizer.py      # HTML stripping, whitespace, quotes, dashes, digits
│           ├── language_profiler.py    # Devanagari ratio, TTR, stopwords, quality score
│           ├── tokenizers.py           # Akshara, BPE, Word, and Sentence tokenizers
│           ├── text_pipeline.py        # End-to-end text dataset transformation pipeline
│           └── text_utils.py           # Feature extraction, URL parsing & domain extraction
│
├── scripts/                            # Domain-organized pipeline runners & benchmarks
│   ├── download/
│   │   └── run_download.py             # Shard discovery & 10k sample downloader
│   ├── sampling/
│   │   └── eval_sampling.py            # Uniformity tests, latency benchmarks & manifests
│   └── text/
│       ├── clean.py                    # Declarative quality filtering pipeline
│       ├── profile.py                  # 6-dimension data quality profiling suite
│       ├── visualize.py                # Publication-quality matplotlib figure generator
│       └── run_pipeline.py             # Text normalization, tokenization, & dataset export
│
├── tests/                              # Comprehensive test suite (36 tests)
│   ├── test_sampler.py                 # Sampling invariants, batch reader, and manifests
│   └── test_text_pipeline.py           # Unicode sanitization, tokenizers, quality profiling
│
├── examples/                           # Interactive demos & prototype scripts
│   └── stream_parquet.py               # Remote Parquet streaming prototype with HfFileSystem
│
├── notes/                              # Architectural Decision Records (ADRs) & mathematical proofs
│   ├── reservoir_sampling_proof.md     # Mathematical induction proof of reservoir sampling
│   └── manifest_hashing_decision.md    # Canonical row hash vs file byte hash rationale
│
├── reports/                            # Generated audit logs, figures, benchmarks & manifests
│   ├── quality_report.md               # Comprehensive data quality audit report
│   ├── text_pipeline_report.json       # Pipeline execution & attrition audit log
│   ├── figures/                        # Profiling histograms and distribution plots
│   └── sampling/                       # Benchmarks, uniformity plots, & sample manifests
│
├── raw/                                # Local raw data directory (gitignored)
└── processed/                          # Local cleaned and tokenized datasets (gitignored)
```

---

## 3. Quickstart & Installation

### Option A: Using `uv` (Recommended — Fastest)
```bash
# 1. Clone repository
git clone https://github.com/shristichand/mini-data-project.git
cd mini-data-project

# 2. Sync virtual environment and dependencies
uv sync

# 3. Verify environment
uv run mini-data-project info

# 4. Run full test suite (36 tests)
uv run pytest -v
```

### Option B: Using Standard Python & `pip`
```bash
# 1. Clone repository
git clone https://github.com/shristichand/mini-data-project.git
cd mini-data-project

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies and package in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Run full test suite
pytest -v
```

---

## 4. Pipeline Execution Reference

You can run each stage using either the unified `mini-data-project` CLI or the standalone scripts in `scripts/`:

### 1. Download Shard & Produce Initial Sample
Downloads the test shard to `raw/` and creates a 10,000-row sample:
```bash
# Via CLI:
mini-data-project download

# Via domain script:
python scripts/download/run_download.py
```

### 2. Run Full Data Profiling Suite
Audits missingness, duplicates, length percentiles, anomalies, language scores, and domains:
```bash
# Via CLI:
mini-data-project profile

# Via domain script:
python scripts/text/profile.py
```

### 3. Generate Visualizations
Generates all 4 publication-quality charts in `reports/figures/`:
```bash
# Via CLI:
mini-data-project visualize

# Via domain script:
python scripts/text/visualize.py
```

### 4. Run Data Cleaning Pipeline
Filters anomalous records and writes `processed/clean.parquet` + `processed/cleaning_log.json`:
```bash
# Via CLI:
mini-data-project clean

# Via domain script:
python scripts/text/clean.py
```

### 5. Run Sampling Benchmark & Verification Harness
Benchmarks batch sizes [512, 2048, 8192], verifies reservoir uniformity across 200 trials, compares stratified allocations, and writes sample manifests to `reports/sampling/`:
```bash
# Via CLI:
mini-data-project sample

# Via domain script:
python scripts/sampling/eval_sampling.py
```

### 6. Run Text Preprocessing & Tokenization Pipeline
Executes Unicode normalization, text cleaning, quality filtering, and tokenization (e.g. Word, Akshara, or BPE):
```bash
# Via CLI:
mini-data-project pipeline --tokenizer word --convert-digits

# Via domain script:
python scripts/text/run_pipeline.py --tokenizer word --convert-digits
```

### 7. Run Test Suite
Executes all 36 unit and integration tests across sampling, Unicode, tokenizers, and pipelines:
```bash
pytest -v
```

---

## 5. Domain-Driven Python Library Usage

The package `mini_data_project` can be imported by domain or from the package root:

```python
# 1. Text Preprocessing Domain
from mini_data_project.text import (
    clean_unicode_pipeline,
    profile_text,
    TextDatasetPipeline,
    PipelineConfig,
    BPETokenizer,
)

raw_text = "नेपाल\u200Bमा पर्यटन प्रवर्द्धन..."
cleaned = clean_unicode_pipeline(raw_text)
profile = profile_text(cleaned)
print(f"Devanagari Ratio: {profile['deva_ratio']:.3f}, Quality: {profile['quality_score']:.3f}")

# 2. Sampling Domain
from mini_data_project.sampling import (
    iter_batches,
    reservoir_sample,
    stratified_sample,
)

sample_df = reservoir_sample("raw/sample_10k.parquet", n=100, seed=42)

# 3. Download Domain
from mini_data_project.download import get_or_create_fineweb2_sample

sample_path = get_or_create_fineweb2_sample(sample_size=10000)
```
