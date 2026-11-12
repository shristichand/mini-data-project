
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()

parquet_path = (
    "datasets/HuggingFaceFW/fineweb-2/data/"
    "npi_Deva/train/000_00000.parquet"
)

with fs.open(parquet_path, "rb") as f:
    parquet_file = pq.ParquetFile(f)

    print("Total rows:", parquet_file.metadata.num_rows)
    print("Row groups:", parquet_file.num_row_groups)

    total_documents = 0
    empty_documents = 0
    missing_documents = 0
    duplicate_cluster_candidates = 0

    for batch in parquet_file.iter_batches(
        batch_size=1000,
        columns=["text", "minhash_cluster_size"]
    ):
        data = batch.to_pylist()

        for row in data:
            total_documents += 1

            text = row["text"]

            if text is None:
                missing_documents += 1
            elif not text.strip():
                empty_documents += 1

            if (
                row["minhash_cluster_size"] is not None
                and row["minhash_cluster_size"] > 1
            ):
                duplicate_cluster_candidates += 1

            if total_documents >= 10000:
                break

        if total_documents >= 10000:
            break

    print("\n--- Raw Data Profile ---")
    print("Documents inspected:", total_documents)
    print("Missing documents:", missing_documents)
    print("Empty documents:", empty_documents)
    print(
        "Documents in non-singleton MinHash clusters:",
        duplicate_cluster_candidates
    )
