# FineWeb-2 Nepali Devanagari (`npi_Deva`) Data Engineering & Sampling Suite

A modular data engineering, profiling, and sampling suite for the Nepali Devanagari (`npi_Deva`) subset of Hugging Face's [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) dataset.

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

---

## 2. Repository Structure

```
mini-data-project/
├── .gitignore                          # Standard Python, macOS, and data ignores
├── README.md                           # Master project guide & office setup instructions
├── pyproject.toml                      # Modern packaging & dependencies
├── uv.lock                             # Locked dependency graph (for uv)
├── requirements.txt                    # Pinned requirements (for standard pip)
├── notes/
│   ├── reservoir_sampling_proof.md     # Mathematical induction proof of reservoir sampling
│   └── manifest_hashing_decision.md    # Canonical row hash vs file byte hash rationale
├── src/
│   ├── config.py                       # Canonical path configuration (SHARD_PATH)
│   ├── text_utils.py                   # Centralized linguistic metrics & feature engineering
│   ├── download_and_sample.py          # Shard discovery, download & initial 10k sample
│   ├── stream_parquet.py               # Remote batch-streaming prototype
│   └── mini_data_project/              # Python package root
├── scripts/
│   ├── profile.py                      # 6-dimension data profiling suite (dynamic conclusions)
│   ├── visualize.py                    # Publication-quality figure generator
│   ├── clean.py                        # Declarative quality filtering pipeline
│   ├── sampler.py                      # Batch reader & 4 sampling implementations
│   └── eval_sampling.py                # Uniformity tests, latency benchmarks & manifests
├── tests/
│   └── test_sampler.py                 # Pytest/Unittest suite (11 test cases, 25+ asserts)
├── reports/
│   ├── quality_report.md               # Comprehensive data quality audit report
│   ├── figures/                        # High-resolution matplotlib profiling plots
│   └── sampling/                       # Benchmarks, uniformity plots, comparison reports & manifests
├── raw/                                # Local raw data directory (gitignored)
└── processed/                          # Local cleaned data directory (gitignored)
```

---

## 3. Office Mac Setup & Quickstart

Follow these steps to set up and run this project on any macOS machine:

### Option A: Using `uv` (Recommended — Fastest)
If your office Mac has `uv` installed (install via `curl -LsSf https://astral.sh/uv/install.sh` or `brew install uv`):

```bash
# 1. Clone repository
git clone https://github.com/shristichand/mini-data-project.git
cd mini-data-project

# 2. Sync virtual environment and dependencies
uv sync

# 3. Download the dataset shard (raw/ is gitignored)
uv run python src/download_and_sample.py

# 4. Run the test suite
uv run pytest tests/test_sampler.py -v
```

### Option B: Using Standard Python & `pip`
If `uv` is not available, standard Python (>= 3.12) and `venv` can be used:

```bash
# 1. Clone repository
git clone https://github.com/shristichand/mini-data-project.git
cd mini-data-project

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Download dataset shard
python src/download_and_sample.py

# 5. Run tests
pytest tests/test_sampler.py -v
```

---

## 4. Pipeline Execution Reference

### Step 1: Download Shard & Produce Initial Sample
Downloads the 66.24 MB test shard to `raw/000_00000.parquet` and creates a 10,000-row sample:
```bash
uv run python src/download_and_sample.py
```

### Step 2: Run Full Data Profiling Suite
Evaluates missingness, duplicates, length percentiles, anomalies, language scores, and domains:
```bash
uv run python scripts/profile.py
```

### Step 3: Generate Visualizations
Generates all 4 publication-quality charts in `reports/figures/`:
```bash
uv run python scripts/visualize.py
```

### Step 4: Run Data Cleaning Pipeline
Filters anomalous records and writes `processed/clean.parquet` + `processed/cleaning_log.json`:
```bash
uv run python scripts/clean.py
```

### Step 5: Run Batch Processing & Sampling Harness
Benchmarks batch sizes [512, 2048, 8192], verifies reservoir uniformity across 200 trials, compares stratified allocations, and writes sample manifests to `reports/sampling/`:
```bash
uv run python scripts/eval_sampling.py
```

### Step 6: Execute Test Suite
Runs the 11 unit and integration tests covering reproducibility and sampling invariants:
```bash
uv run pytest tests/test_sampler.py -v
```
