import mlflow
import time
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import faiss
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws

# MLflow experiment setup
mlflow.set_experiment("Fraud_Embedding_Generation")

with mlflow.start_run():
    # Set up Spark
    spark = SparkSession.builder \
        .appName("EmbeddingExperiment") \
        .enableHiveSupport() \
        .getOrCreate()

    mlflow.log_param("source_table", "datamart.fraud_transactions")

    # Load Hive table
    df = spark.sql("SELECT transaction_id, user_id, amount, category, country FROM datamart.fraud_transactions")
    df = df.withColumn("text", concat_ws(" ", "user_id", "amount", "category", "country"))

    # Convert to pandas
    df_pd = df.select("transaction_id", "text").limit(10000).toPandas()  # for demo
    texts = df_pd["text"].tolist()

    # Load model
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    duration = time.time() - start

    mlflow.log_metric("embedding_time_sec", duration)
    mlflow.log_metric("embedding_count", len(embeddings))

    # Save artifacts
    os.makedirs("mlruns_output", exist_ok=True)
    np.save("mlruns_output/embeddings.npy", embeddings)
    df_pd.to_csv("mlruns_output/texts.csv", index=False)

    id_to_text = dict(zip(df_pd["transaction_id"].tolist(), texts))
    with open("mlruns_output/id_to_text.json", "w") as f:
        json.dump(id_to_text, f)

    # Index with FAISS
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, "mlruns_output/embeddings_index.faiss")

    # Log artifacts to MLflow
    mlflow.log_artifacts("mlruns_output")
    mlflow.set_tag("use_case", "fraud_detection")
    mlflow.end_run()