from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import random
import os

# Spark session
spark = SparkSession.builder \
    .appName("PrepareLabeledFraudPairs") \
    .enableHiveSupport() \
    .getOrCreate()

# Read fraud and non-fraud samples from Hive
fraud_df = spark.sql("SELECT transaction_id, user_id, amount, category, country, device_type FROM datamart.fraud_transactions WHERE is_fraud = 1 LIMIT 10000")
nonfraud_df = spark.sql("SELECT transaction_id, user_id, amount, category, country, device_type FROM datamart.fraud_transactions WHERE is_fraud = 0 LIMIT 10000")

fraud_pd = fraud_df.toPandas()
nonfraud_pd = nonfraud_df.toPandas()

# Generate positive pairs (same class = fraud)
positive_pairs = [
    {
        "text1": f"{row1['user_id']} {row1['amount']} {row1['category']} {row1['country']} {row1['device_type']}",
        "text2": f"{row2['user_id']} {row2['amount']} {row2['category']} {row2['country']} {row2['device_type']}",
        "label": 1.0
    }
    for row1, row2 in zip(fraud_pd.sample(frac=1).head(5000), fraud_pd.sample(frac=1).head(5000))
]

# Generate negative pairs (different class)
negative_pairs = [
    {
        "text1": f"{row1['user_id']} {row1['amount']} {row1['category']} {row1['country']} {row1['device_type']}",
        "text2": f"{row2['user_id']} {row2['amount']} {row2['category']} {row2['country']} {row2['device_type']}",
        "label": 0.0
    }
    for row1, row2 in zip(fraud_pd.head(5000), nonfraud_pd.head(5000))
]

# Combine and save
pairs = pd.DataFrame(positive_pairs + negative_pairs)
os.makedirs("mlruns_output", exist_ok=True)
pairs.to_parquet("mlruns_output/labeled_fraud_pairs.parquet", index=False)
print("✅ Labeled pairs written to mlruns_output/labeled_fraud_pairs.parquet")
