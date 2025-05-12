import os, json
import ray
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import cml.data_v1 as cmldata

# Init Ray
ray.init(ignore_reinit_error=True)

# Parameters
BATCH_SIZE = 1_000_000          # One large batch from Impala
CHUNK_SIZE = 20_000             # Chunk processed per Ray GPU worker
TOTAL_ROWS = 255_000_000
CONNECTION_NAME = "impala-virtual-warehouse"
os.makedirs("model/tmp_batches", exist_ok=True)

# GPU task: process small chunk of data
@ray.remote(num_gpus=1)
def encode_chunk(chunk_texts: list):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model.encode(chunk_texts, batch_size=64, show_progress_bar=False).tolist()

# Connect to Impala
conn = cmldata.get_connection(CONNECTION_NAME)
batch_id = 0

for offset in range(0, TOTAL_ROWS, BATCH_SIZE):
    print(f"🔄 Batch {batch_id+1} | transaction_id >= {offset} AND < {offset + BATCH_SIZE}")
    query = f"""
        SELECT transaction_id, user_id, amount, category, country, device_type
        FROM datamart.fraud_transactions
        WHERE transaction_id >= {offset} AND transaction_id < {offset + BATCH_SIZE}
    """
    df = conn.get_pandas_dataframe(query)
    if df.empty:
        break

    # Prepare text column
    df["text"] = df.apply(lambda r: f"{r['user_id']} {r['amount']} {r['category']} {r['country']} {r['device_type']}", axis=1)
    
    # Chunk the texts and encode in parallel
    chunk_refs = []
    chunks = [df["text"].iloc[i:i+CHUNK_SIZE].tolist() for i in range(0, len(df), CHUNK_SIZE)]
    for chunk in chunks:
        chunk_refs.append(encode_chunk.remote(chunk))

    # Gather all embeddings
    all_embeddings = ray.get(chunk_refs)
    embeddings_flat = [item for sublist in all_embeddings for item in sublist]

    # Save
    df["embedding"] = embeddings_flat
    df[["transaction_id", "text", "embedding"]].to_parquet(f"model/tmp_batches/batch_{batch_id}.parquet", index=False)
    batch_id += 1

conn.close()
print("✅ All batches saved to model/tmp_batches/")