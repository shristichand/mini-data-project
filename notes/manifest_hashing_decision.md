# Engineering Decision: Parquet Manifest Hashing Strategy

## 1. Context & Problem Statement
When generating reproducibility manifests for dataset samples (`<name>.manifest.json`), we must record a cryptographic checksum (SHA-256) verifying the output.

However, **Parquet binary outputs can produce differing byte-level SHA-256 hashes even when the logical data rows are 100% identical**. Specifically:
- **Timestamp Metadata**: The Parquet file footer often records the write timestamp in UTC. Two writes of the identical DataFrame 5 seconds apart will contain different byte representations in the footer.
- **Compression & Page Chunking**: Minor differences in underlying Snappy/ZSTD library versions or thread scheduling across PyArrow releases can cause page boundaries and dictionary tables to vary slightly.
- **Column Order & Categorical Encodings**: Writing identical data with differing in-memory column orders or dictionary indices alters binary serialization.

---

## 2. Decision: Dual Hashing Strategy

To solve this, we compute and store **both** a **Canonical Row Hash** and the **Physical File Bytes Hash**:

### A. Primary Semantic Hash: Sorted Canonical Row Hash (`data_hash_sha256`)
- **How it works**:
  1. Sort the DataFrame by a primary key (`id` or all columns if no unique key).
  2. Serialize records to a deterministic, canonical representation (e.g., newline-delimited JSON or canonical tuple sequence with UTF-8 encoding).
  3. Compute the SHA-256 over this canonical serialized data.
- **Why this is essential**:
  This guarantees that **if two sampling runs produce the exact same records in the exact same logical state, their `data_hash_sha256` values are 100% identical**, completely immune to Parquet footer metadata timestamps, OS differences, or PyArrow internal dictionary variations.

### B. Transport Integrity Hash: File Byte Hash (`file_bytes_sha256`)
- **How it works**:
  Reads the raw binary bytes of the output `.parquet` file in chunks using `hashlib.sha256()` and records the hexadecimal digest.
- **Why this is useful**:
  Provides standard transport verification to verify that a file downloaded over the network or copied between disks has not suffered bit rot or truncation.

---

## 3. Summary Manifest Schema Example
```json
{
  "method": "reservoir_sampling",
  "seed": 42,
  "parameters": {
    "n": 1000,
    "batch_size": 2048
  },
  "source_file": "raw/000_00000.parquet",
  "source_row_count": 49462,
  "sample_row_count": 1000,
  "data_hash_sha256": "4b9e12...",
  "file_bytes_sha256": "9a71fc...",
  "timestamp_utc": "2026-11-14T10:30:00Z"
}
```
This dual-hash approach provides both **logical reproducibility** and **physical file integrity**.

