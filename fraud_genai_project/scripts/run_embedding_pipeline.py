import sys
from pyspark.sql.functions import concat_ws
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
df = spark.sql("SELECT * FROM datamart.fraud_transactions")
df = df.withColumn("text", concat_ws(" ", "user_id", "amount", "category", "country"))
texts = df.select("text").rdd.map(lambda r: r[0]).collect()
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)