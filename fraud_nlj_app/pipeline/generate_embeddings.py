# pipeline/generate_embeddings.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws
from pyspark.sql.types import StringType, LongType, StructType, StructField, ArrayType, FloatType
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import json
import os
import torch

# Step 1: Spark setup
spark = SparkSession.builder \
    .appName("GenerateEmbeddings") \
    .enableHiveSupport() \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

# Step 2: Load Hive data
print("🔍 Loading data from Hive...")
df = spark.sql("""
    SELECT transaction_id, user_id, amount, category, country, device_type
    FROM datamart.fraud_transactions
    LIMIT 1000
""")
df = df.withColumn("text", concat_ws(" ", "user_id", "amount", "category", "country", "device_type"))
df = df.cache()

# Step 3: Define embedding function (GPU-aware)
def embed_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if torch.cuda.is_available():
        model = model.to("cuda")
    embeddings = model.encode(
        pdf["text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return pd.DataFrame({
        "transaction_id": pdf["transaction_id"],
        "text": pdf["text"],
        "embedding": embeddings.tolist()
    })

# Step 4: Distributed embedding using mapInPandas
schema = StructType([
    StructField("transaction_id", LongType(), True),
    StructField("text", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True)
])
embedding_df = df.select("transaction_id", "text") \
    .repartition(10) \
    .mapInPandas(embed_partition, schema=schema)

# Step 5: Write intermediate output
print("📤 Writing intermediate embeddings to disk...")
embedding_df.write.mode("overwrite").parquet("model/embeddings_parquet")

# Step 6: Load for FAISS indexing
print("📥 Loading embeddings back for FAISS index...")
reloaded_df = pd.read_parquet("model/embeddings_parquet")

# Step 7: FAISS Index
print("💾 Building and writing FAISS index...")
vectors = np.vstack(reloaded_df["embedding"].values).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
os.makedirs("model", exist_ok=True)
faiss.write_index(index, "model/embeddings_index.faiss")

# Step 8: Save ID-to-text mapping
print("📝 Writing ID-to-text mapping...")
id_to_text = {str(row.transaction_id): row.text for row in reloaded_df.itertuples(index=False)}
with open("model/id_to_text.json", "w") as f:
    json.dump(id_to_text, f)

print("✅ All done. Files saved to /model")