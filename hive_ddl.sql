CREATE EXTERNAL TABLE datamart.fraud_transactions (
  transaction_id BIGINT,
  user_id INT,
  amount DOUBLE,
  category STRING,
  `timestamp` TIMESTAMP,
  country STRING,
  device_type STRING,
  is_fraud INT
)
STORED AS PARQUET
TBLPROPERTIES ('external.table.purge'='true');