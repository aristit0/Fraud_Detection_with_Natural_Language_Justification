import pandas as pd
import numpy as np
import faiss
import json
import os
from glob import glob
import pyarrow.parquet as pq

print("📦 Loading batch Parquet files...")

batch_dir = "model/tmp_batches"
files = sorted(glob(os.path.join(batch_dir, "batch_*.parquet")))

if not files:
    raise FileNotFoundError(f"No parquet files found in {batch_dir}")

all_txn, all_text, all_vec = [], [], []

for f in files:
    try:
        # Check for valid parquet structure before loading
        pq.read_metadata(f)
        df = pd.read_parquet(f)

        if "transaction_id" not in df or "text" not in df or "embedding" not in df:
            print(f"⚠️ Skipping file with missing columns: {f}")
            continue

        all_txn.extend(df["transaction_id"].tolist())
        all_text.extend(df["text"].tolist())
        all_vec.extend(df["embedding"].tolist())

    except Exception as e:
        print(f"❌ Failed to load {f}: {e}")

print(f"🔢 Total vectors loaded: {len(all_vec)}")

if not all_vec:
    raise ValueError("No vectors found. Check input files.")

vectors = np.array(all_vec, dtype="float32")

# Build FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Save outputs
os.makedirs("model", exist_ok=True)
faiss.write_index(index, "model/embeddings_index.faiss")

with open("model/id_to_text.json", "w") as f:
    json.dump({str(k): v for k, v in zip(all_txn, all_text)}, f)

print("✅ Final FAISS index and ID-to-text mapping saved.")