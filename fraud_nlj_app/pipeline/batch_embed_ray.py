import os, json
import ray
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import cml.data_v1 as cmldata

# Init Ray
ray.init(ignore_reinit_error=True)

BATCH_SIZE = 1_000_000
TOTAL_ROWS = 255_000_000
NUM_BATCHES = (TOTAL_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
CONNECTION_NAME = "impala-virtual-warehouse"

os.makedirs("model/tmp_batches", exist_ok=True)

@ray.remote(num_gpus=1)
def encode_texts(batch_df: pd.DataFrame):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(batch_df["text"].tolist(), batch_size=64, show_progress_bar=False)
    return embeddings.tolist()

conn = cmldata.get_connection(CONNECTION_NAME)

for batch_id in range(NUM_BATCHES):
    offset = batch_id * BATCH_SIZE
    print(f"🔄 Batch {batch_id+1}/{NUM_BATCHES} - OFFSET {offset}")

    query = f"""
        SELECT transaction_id, user_id, amount, category, country, device_type
        FROM datamart.fraud_transactions
        LIMIT {BATCH_SIZE} OFFSET {offset}
    """
    df = conn.get_pandas_dataframe(query)
    if df.empty:
        break

    df["text"] = df.apply(lambda r: f"{r['user_id']} {r['amount']} {r['category']} {r['country']} {r['device_type']}", axis=1)
    embeddings = ray.get(encode_texts.remote(df))

    df["embedding"] = embeddings
    df[["transaction_id", "text", "embedding"]].to_parquet(f"model/tmp_batches/batch_{batch_id}.parquet", index=False)

conn.close()
print("✅ All batches saved to model/tmp_batches/")