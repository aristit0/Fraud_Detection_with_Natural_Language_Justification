import os, json
import ray
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import cml.data_v1 as cmldata

# Init Ray
ray.init(ignore_reinit_error=True)

# Configuration
BATCH_SIZE = 1_000_000
START_ID = 0
END_ID = 255_000_000
CONNECTION_NAME = "impala-virtual-warehouse"

# Output directory
os.makedirs("model/tmp_batches", exist_ok=True)

# Remote function to use GPU
@ray.remote(num_gpus=1)
def encode_texts(batch_df: pd.DataFrame):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(batch_df["text"].tolist(), batch_size=64, show_progress_bar=False)
    return embeddings.tolist()

# Connect to Impala
conn = cmldata.get_connection(CONNECTION_NAME)

# Loop through transaction_id ranges instead of OFFSET
batch_id = 0
for batch_start in range(START_ID, END_ID, BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    print(f"🔄 Batch {batch_id+1} | transaction_id >= {batch_start} AND < {batch_end}")

    query = f"""
        SELECT transaction_id, user_id, amount, category, country, device_type
        FROM datamart.fraud_transactions
        WHERE transaction_id >= {batch_start} AND transaction_id < {batch_end}
    """
    df = conn.get_pandas_dataframe(query)
    if df.empty:
        print("🚫 No more data.")
        break

    df["text"] = df.apply(lambda r: f"{r['user_id']} {r['amount']} {r['category']} {r['country']} {r['device_type']}", axis=1)
    embeddings = ray.get(encode_texts.remote(df))

    df["embedding"] = embeddings
    df[["transaction_id", "text", "embedding"]].to_parquet(f"model/tmp_batches/batch_{batch_id}.parquet", index=False)
    batch_id += 1

# Close connection
conn.close()
print("✅ All batches saved to model/tmp_batches/")