# from datasets import load_dataset 

# dataset = load_dataset("samsum")
# print(dataset)

# from huggingface_hub import hf_hub_download
# import pandas as pd

# REPO_ID = "YOUR_REPO_ID"
# FILENAME = "data.csv"

# dataset = pd.read_csv(
#     hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
# )

from huggingface_hub import HfFileSystem

fs = HfFileSystem()

repo_id = "HuggingFaceFW/fineweb-2"
subset = "npi_Deva"

parquet_path = f"datasets/{repo_id}/data/{subset}/test/000_00000.parquet"
print(f"Parquet path: {parquet_path}")

# with fs.open(parquet_path, "rb") as f:
#     data = f.read()


repository_path = f"datasets/{repo_id}/data/{subset}"
print(f"Repository path: {repository_path}")

# List dataset repository contents
files = fs.ls(repository_path)

for file in files:
    print(file["name"], file["type"])

train_path = f"{repository_path}/train"

train_files = fs.ls(train_path)


print("Total files:", len(train_files))
print("First 10 entries:", train_files[:10])

for file in train_files[:10]:
    print(file["name"], file["type"], file["size"])