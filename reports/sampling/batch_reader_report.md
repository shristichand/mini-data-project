# Parquet Batch Processing & Reader Evaluation Report

**Target Dataset Shard**: `/Users/siru/mini-data-project/raw/000_00000.parquet`  
**Total Records**: 49,462 | **File Size**: 66.24 MB | **Row Groups**: 50  
**Evaluation Timestamp**: 2026-11-14T10:37:09.123894+00:00  

---

## 1. Batch Size Benchmark

Reading only projected columns (`['id', 'text', 'language_score']`):

| Batch Size | Batch Count | First Batch Rows | Last Batch Rows | Total Rows | Elapsed Time (s) | Throughput (rows/s) | Latency (ms/batch) |
|---|---|---|---|---|---|---|---|
| **512** | 97 | 512 | 310 | 49,462 | 0.1638 | 301,883 | 1.69 |
| **2,048** | 25 | 2,048 | 310 | 49,462 | 0.1574 | 314,224 | 6.30 |
| **8,192** | 7 | 8,192 | 310 | 49,462 | 0.1693 | 292,185 | 24.18 |

> **Verification**: All batch configurations stream exactly **49,462 rows**, matching Parquet metadata.

## 2. Column Projection Pushdown Speedup

Comparison at `batch_size = 2048`:

- **Full Schema (11 columns)**: 0.1841s (268,674 rows/s)
- **Projected Schema (3 columns: `id`, `text`, `language_score`)**: 0.1574s (314,224 rows/s)
- **Speedup**: **1.17x faster** by avoiding unused column deserialization.

## 3. Recommended Production Settings

- **Default Batch Size**: `2,048` rows strikes the optimal balance between streaming throughput (~300k rows/s) and memory consumption (~2-4 MB RAM footprint per batch).
- **Projection Pushdown**: Always pass explicit `columns` list to `iter_batches` to avoid deserializing unneeded metadata fields.
