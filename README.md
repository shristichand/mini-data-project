# mini-data-project

Data pipeline for exploring, inspecting, and sampling the Nepali Devanagari (`npi_Deva`) subset from Hugging Face's [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) dataset.

## Dataset Shards Overview

Using `HfFileSystem().ls("datasets/HuggingFaceFW/fineweb-2/data/npi_Deva", detail=True)`, the repository structure contains `train` and `test` splits:

| Split | File Path | Size (Bytes) | Size (MB) |
|---|---|---|---|
| **test** | `datasets/HuggingFaceFW/fineweb-2/data/npi_Deva/test/000_00000.parquet` | 69,460,662 | 66.24 MB |
| **train** | `datasets/HuggingFaceFW/fineweb-2/data/npi_Deva/train/000_00000.parquet` | 4,843,391,525 | 4,619.02 MB |
| **train** | `datasets/HuggingFaceFW/fineweb-2/data/npi_Deva/train/001_00000.parquet` | 1,840,006,187 | 1,754.77 MB |

- **Test split check**: The test split is indeed a **single Parquet file** (`test/000_00000.parquet`, 66.24 MB).

## Schema Inspection

The schema and metadata were inspected using `pyarrow.parquet.ParquetFile` without loading the full shard into memory:
- **Total rows in shard**: 49,462

### Column Definitions

- `text` (string): Extracted raw text content of the web page in Nepali (Devanagari script).
- `id` (string): Unique document identifier formatted as a UUID URI (`<urn:uuid:...>`).
- `dump` (string): Common Crawl snapshot ID from which the document was extracted (e.g., `CC-MAIN-2014-23`).
- `url` (string): Original web page source URL crawled by Common Crawl.
- `date` (string): Timestamp of when the webpage was crawled and archived (ISO 8601 UTC).
- `file_path` (string): S3 path to the source Common Crawl WARC archive file.
- `language` (string): Primary detected language code of the text (`npi` for Nepali).
- `language_score` (double): Language identification confidence probability (between 0.0 and 1.0).
- `language_script` (string): Detected writing script of the text (`Deva` for Devanagari).
- `minhash_cluster_size` (int64): Size of the cluster containing near-duplicate documents identified during MinHash deduplication.
- `top_langs` (string): JSON dictionary of candidate language-script tags and their confidence scores.

## Sampling and Storage

From the downloaded shard:
1. Sampled **10,000 rows** using `df.sample(n=10000, random_state=42)`.
2. Stored the sample in two formats under `raw/`:
   - `raw/sample_10k.parquet`: Columnar format preserving Arrow data types.
   - `raw/sample_10k.jsonl`: Line-delimited JSON with UTF-8 encoding (`ensure_ascii=False` / `force_ascii=False`).
3. Reloaded both files to verify that each contains exactly 10,000 rows.

## Quickstart

Run the complete pipeline:
```bash
uv run python src/download_and_sample.py
```
