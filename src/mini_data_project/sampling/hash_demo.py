import json
from pathlib import Path
import pandas as pd

from mini_data_project.config import SHARD_PATH
from mini_data_project.sampling import (
    random_sample_inmemory,
    save_sample_with_manifest,
    compute_canonical_data_hash,
    compute_file_hash,
)

OUT = Path("reports/hash_demo")
OUT.mkdir(parents=True, exist_ok=True)

def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")

N, SEED = 1000, 42
df = random_sample_inmemory(SHARD_PATH, n=N, seed=SEED)
path, mpath = save_sample_with_manifest(
    df, str(OUT / "a.parquet"), "random_inmemory", SEED, {"n": N, "seed": SEED}
)
manifest = json.load(open(mpath))
print("Manifest data hash :", manifest["data_hash_sha256"][:16], "...")
print("Manifest file hash :", manifest["file_bytes_sha256"][:16], "...\n")

# 1. Same seed regenerates the same sample
again = random_sample_inmemory(SHARD_PATH, n=N, seed=SEED)
check("1. same seed -> same data hash", compute_canonical_data_hash(again, "sha256") == manifest["data_hash_sha256"])

# 2. Row order doesn't matter for the data hash, but changes the file bytes
shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
shuffled.to_parquet(OUT / "shuffled.parquet", index=False)
check("2a. shuffled rows -> same data hash", compute_canonical_data_hash(shuffled, "sha256") == compute_canonical_data_hash(df, "sha256"))
check("2b. shuffled rows -> different file hash",
      compute_file_hash(str(OUT / "shuffled.parquet"), "sha256") != manifest["file_bytes_sha256"])

# 3. Changing one character changes the data hash
edited = df.copy()
edited.loc[0, "text"] = edited.loc[0, "text"] + "x"
check("3. one char edited -> data hash changes", compute_canonical_data_hash(edited, "sha256") != compute_canonical_data_hash(df, "sha256"))

# 4. Corrupting one byte changes the file hash
raw = bytearray(Path(path).read_bytes())
raw[len(raw) // 2] ^= 0x01
Path(OUT / "corrupt.parquet").write_bytes(raw)
check("4. one byte flipped -> file hash changes",
      compute_file_hash(str(OUT / "corrupt.parquet"), "sha256") != manifest["file_bytes_sha256"])

# 5. Different seed gives a different sample and hash
other = random_sample_inmemory(SHARD_PATH, n=N, seed=7)
check("5. different seed -> different data hash", compute_canonical_data_hash(other, "sha256") != compute_canonical_data_hash(df, "sha256"))

# 6. Verify from disk against the manifest
reloaded = pd.read_parquet(path)
check("6a. reloaded file data hash == manifest",
      compute_canonical_data_hash(reloaded, "sha256") == manifest["data_hash_sha256"])
check("6b. file bytes hash == manifest",
      compute_file_hash(path, "sha256") == manifest["file_bytes_sha256"])