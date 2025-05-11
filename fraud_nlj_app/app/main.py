# app/main.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os
import json

app = Flask(__name__)

# Load embedding model (GPU-enabled if available)
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load LLM for explanation
gen_pipeline = pipeline("text-generation", model="gpt2", device=0)

# Load FAISS index and id mapping
faiss_index = faiss.read_index("model/embeddings_index.faiss")
with open("model/id_to_text.json") as f:
    id_to_text = json.load(f)

# Justification prompt template
prompt_template = (
    "A user just did the following transaction: \"{transaction}\". "
    "Here are similar past fraud examples: {examples}. "
    "Is this suspicious? Why?"
)

@app.route("/", methods=["POST"])
def root():
    data = request.json
    transaction_text = data.get("transaction")

    # Embed transaction
    query_vec = embedding_model.encode([transaction_text])

    # Find similar transactions
    D, I = faiss_index.search(query_vec, k=5)
    similar_examples = [id_to_text[str(i)] for i in I[0] if str(i) in id_to_text]

    # Generate explanation
    prompt = prompt_template.format(transaction=transaction_text, examples="; ".join(similar_examples))
    result = gen_pipeline(prompt, max_length=100, do_sample=True, temperature=0.7)

    return jsonify({
        "transaction": transaction_text,
        "similar_frauds": similar_examples,
        "justification": result[0]["generated_text"]
    })

# Start the app in CML environment
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))