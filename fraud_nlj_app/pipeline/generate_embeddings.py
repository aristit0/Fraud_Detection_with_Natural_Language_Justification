# pipeline/generate_embeddings.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws
from sentence_transformers import SentenceTransformer
from pyspark.sql.types import StringType, LongType, StructType, StructField, ArrayType, FloatType
import pandas as pd
import numpy as np
import faiss
import json
import os

# Step 1: Spark setup
spark = SparkSession.builder \
    .appName("GenerateEmbeddings") \
    .enableHiveSupport() \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# Step 2: Load Hive data (LIMIT for test)
print("🔍 Loading data from Hive...")
df = spark.sql("""
    SELECT transaction_id, user_id, amount, category, country, device_type
    FROM datamart.fraud_transactions
    LIMIT 1000
""")
df = df.withColumn("text", concat_ws(" ", "user_id", "amount", "category", "country", "device_type"))
df = df.cache()
df.count()  # Force materialization

# Step 3: Define embedding function
def embed_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(pdf["text"].tolist(), batch_size=16, show_progress_bar=False)
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
    .repartition(4) \
    .mapInPandas(embed_partition, schema=schema)

# Step 5: Collect back results to driver
print("📥 Collecting embeddings to driver...")
collected = embedding_df.toPandas()

# Step 6: Save FAISS index
print("💾 Writing FAISS index and id-to-text mapping...")
vectors = np.vstack(collected["embedding"].values).astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
os.makedirs("model", exist_ok=True)
faiss.write_index(index, "model/embeddings_index.faiss")

# Save mapping
id_to_text = {str(row.transaction_id): row.text for row in collected.itertuples(index=False)}
with open("model/id_to_text.json", "w") as f:
    json.dump(id_to_text, f)

print("✅ All done. FAISS index and text mapping saved to /model")