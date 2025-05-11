# pipeline/generate_embeddings.py
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws
import faiss
import numpy as np
import os
import json

# Spark setup
spark = SparkSession.builder \
    .appName("GenerateEmbeddings") \
    .enableHiveSupport() \
    .getOrCreate()

# Load data
print("🔍 Loading data from Hive table...")
df = spark.sql("SELECT transaction_id, user_id, amount, category, country FROM datamart.fraud_transactions")
df = df.withColumn("text", concat_ws(" ", "user_id", "amount", "category", "country"))

# Collect data to driver
rows = df.select("transaction_id", "text").collect()
texts = [row["text"] for row in rows]
ids = [row["transaction_id"] for row in rows]

# Embed
print("🔢 Generating embeddings...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

# Save FAISS index
print("💾 Saving FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, "model/embeddings_index.faiss")

# Save text mapping
os.makedirs("model", exist_ok=True)
with open("model/id_to_text.json", "w") as f:
    json.dump({str(i): text for i, text in zip(ids, texts)}, f)

print("✅ Embeddings and index saved.")
