import ray
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import json
import cml.data_v1 as cmldata

# Step 1: Ray init
ray.init(ignore_reinit_error=True)

# Step 2: Load from Impala
conn = cmldata.get_connection("impala-virtual-warehouse")
query = """
SELECT transaction_id, user_id, amount, category, country, device_type
FROM datamart.fraud_transactions
"""
print("📥 Running SQL query...")
df = conn.get_pandas_dataframe(query)
conn.close()

# Step 3: Prepare text
df["text"] = df.apply(lambda row: f"{row['user_id']} {row['amount']} {row['category']} {row['country']} {row['device_type']}", axis=1)

# Step 4: Split into batches
chunks = np.array_split(df, 100)

@ray.remote(num_gpus=1)
def encode_chunk(chunk):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(chunk["text"].tolist(), batch_size=64, show_progress_bar=False)
    return {
        "transaction_id": chunk["transaction_id"].tolist(),
        "text": chunk["text"].tolist(),
        "embedding": embeddings.tolist()
    }

# Step 5: Distribute encoding
futures = [encode_chunk.remote(chunk) for chunk in chunks]
print("🚀 Embedding in parallel across GPUs...")
results = ray.get(futures)

# Step 6: Flatten results
all_txn, all_text, all_vec = [], [], []
for r in results:
    all_txn.extend(r["transaction_id"])
    all_text.extend(r["text"])
    all_vec.extend(r["embedding"])

# Step 7: Build FAISS
vectors = np.array(all_vec).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Step 8: Save outputs
os.makedirs("model", exist_ok=True)
faiss.write_index(index, "model/embeddings_index.faiss")

with open("model/id_to_text.json", "w") as f:
    json.dump({str(k): v for k, v in zip(all_txn, all_text)}, f)

print("✅ Selesai. Model dan FAISS index tersimpan di folder /model")