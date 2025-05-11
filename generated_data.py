from pyspark.sql import SparkSession
from pyspark.sql.types import *
from faker import Faker
import random
import datetime

# Initialize
spark = SparkSession.builder \
    .appName("FraudDataGenerator") \
    .enableHiveSupport() \
    .getOrCreate()

fake = Faker()
Faker.seed(42)
random.seed(42)

# Define schema
schema = StructType([
    StructField("transaction_id", LongType(), False),
    StructField("user_id", IntegerType(), False),
    StructField("amount", DoubleType(), False),
    StructField("category", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("country", StringType(), False),
    StructField("device_type", StringType(), False),
    StructField("is_fraud", IntegerType(), False)
])

# Sample domains
categories = ["electronics", "travel", "food", "clothing", "jewelry", "crypto"]
countries = ["US", "UK", "JP", "ID", "DE", "SG", "NG"]
devices = ["mobile", "desktop", "tablet"]

# Generator function
def generate_batch(start_id, batch_size):
    data = []
    for i in range(batch_size):
        data.append((
            start_id + i,
            random.randint(1000, 9999),
            round(random.uniform(10, 10000), 2),
            random.choice(categories),
            fake.date_time_between(start_date='-1y', end_date='now'),
            random.choice(countries),
            random.choice(devices),
            random.choices([0, 1], weights=[0.995, 0.005])[0]
        ))
    return data

# Batch write loop
total_records = 1_000_000_000
batch_size = 1_000_000

for start_id in range(0, total_records, batch_size):
    print(f"Generating records {start_id} to {start_id + batch_size}")
    batch_data = generate_batch(start_id, batch_size)
    df = spark.createDataFrame(batch_data, schema=schema)
    
    df.write.mode("append").insertInto("datamart.fraud_transactions")  # Hive insert
    
    print(f"✅ Inserted batch up to: {start_id + batch_size}")