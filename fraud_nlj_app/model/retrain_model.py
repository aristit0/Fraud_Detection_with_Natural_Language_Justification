# model/retrain_model.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os

# Load dataset
print("📦 Loading training data...")
df = pd.read_csv("model/labeled_fraud_pairs.csv")  # CSV with columns: text1, text2, label (1=fraud match, 0=not)
df = df.dropna()

train_examples = [
    InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
    for _, row in df.iterrows()
]

# Load base model
print("🧠 Loading base model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Prepare dataloader and loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)

# Train model
print("🚀 Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100
)

# Save model
save_path = "model/retrained_embedding_model"
os.makedirs(save_path, exist_ok=True)
model.save(save_path)
print(f"✅ Model saved to: {save_path}")
