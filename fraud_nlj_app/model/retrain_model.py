import os
import mlflow
import pandas as pd
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Load training data from Parquet or Hive-converted file
DATA_PATH = "mlruns_output/labeled_fraud_pairs.parquet"
df = pd.read_parquet(DATA_PATH)

# Expected format: anchor, positive, negative
# You can modify this logic based on your labeled format
data = [
    InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
    for _, row in df.iterrows()
]

# Set experiment tracking
mlflow.set_experiment("Fraud_Justification_Retrain")

with mlflow.start_run():
    mlflow.log_param("training_samples", len(data))

    # Load base model (GPU will be used if available)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Create DataLoader
    train_dataloader = DataLoader(data, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model=model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100
    )

    # Save model locally and log as artifact
    model_save_path = "trained_fraud_model"
    model.save(model_save_path)
    mlflow.log_artifacts(model_save_path, artifact_path="model")

    mlflow.set_tag("use_case", "fraud_detection_with_justification")
    mlflow.end_run()
