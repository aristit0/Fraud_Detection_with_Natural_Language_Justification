from pyspark.sql import SparkSession
from pyspark.sql.types import *
from faker import Faker
import random
import datetime

fake = Faker()
spark = SparkSession.builder.appName("FraudDataGen").getOrCreate()

schema = StructType([
    StructField("transaction_id", LongType(), False),
    StructField("user_id", IntegerType(), False),
    StructField("amount", DoubleType(), False),
    StructField("category", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("country", StringType(), False),
    StructField("device_type", StringType(), False),
    StructField("is_fraud", IntegerType(), False),
])

def generate_fake_row(start_id):
    return (
        start_id,
        random.randint(1000, 9999),
        round(random.uniform(10, 10000), 2),
        random.choice(["electronics", "travel", "food", "clothing", "jewelry", "crypto"]),
        fake.date_time_between(start_date='-1y', end_date='now'),
        random.choice(["US", "UK", "JP", "ID", "DE", "SG", "NG"]),
        random.choice(["mobile", "desktop", "tablet"]),
        random.choices([0, 1], weights=[0.995, 0.005])[0]
    )

batch_size = 10_000_000
total_records = 1_000_000_000
output_path = "hdfs:///tmp/fraud_data"

for i in range(0, total_records, batch_size):
    data = [generate_fake_row(i + j) for j in range(batch_size)]
    df = spark.createDataFrame(data, schema)
    df.write.mode("append").csv(output_path)
    print(f"Written {i + batch_size} records")