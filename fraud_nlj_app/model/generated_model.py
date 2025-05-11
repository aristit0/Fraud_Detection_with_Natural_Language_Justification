from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col
import os

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# Join on random pairs with different transaction_id
df = spark.sql("""
    SELECT
        CONCAT_WS(' ', A.user_id, A.amount, A.category, A.country) AS text1,
        CONCAT_WS(' ', B.user_id, B.amount, B.category, B.country) AS text2,
        CASE WHEN A.is_fraud = B.is_fraud THEN 1 ELSE 0 END AS label
    FROM datamart.fraud_transactions A
    JOIN datamart.fraud_transactions B
        ON A.transaction_id != B.transaction_id
    WHERE RAND() < 0.00001
    LIMIT 5000
""")

# Save to CSV
os.makedirs("model", exist_ok=True)
df.write.mode("overwrite").option("header", True).csv("model/labeled_fraud_pairs_csv")

# Convert to single file
df.coalesce(1).write.mode("overwrite").option("header", True).csv("model/labeled_fraud_pairs_csv")

print("✅ Saved labeled pairs to model/labeled_fraud_pairs_csv/")