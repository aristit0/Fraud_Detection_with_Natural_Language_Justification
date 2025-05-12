import pandas as pd
import numpy as np
import faiss
import json
import os
from glob import glob

print("📦 Loading batch Parquet files...")
files = sorted(glob("model/tmp_batches/batch_*.parquet"))
all_txn, all_text, all_vec = [], [], []

for f in files:
    df = pd.read_parquet(f)
    all_txn.extend(df["transaction_id"].tolist())
    all_text.extend(df["text"].tolist())
    all_vec.extend(df["embedding"].tolist())

print(f"🔢 Total vectors: {len(all_vec)}")

vectors = np.array(all_vec).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

print("💾 Saving FAISS index...")
os.makedirs("model", exist_ok=True)
faiss.write_index(index, "model/embeddings_index.faiss")

with open("model/id_to_text.json", "w") as f:
    json.dump({str(k): v for k, v in zip(all_txn, all_text)}, f)

print("✅ Final FAISS index and ID-to-text mapping saved.")