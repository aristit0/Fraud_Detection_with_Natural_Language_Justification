# app/main.py
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os
import json

app = Flask(__name__, template_folder="templates")

# Load embedding model (GPU-enabled if available)
embedding_model = SentenceTransformer("model/retrained_embedding_model")

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

@app.route("/", methods=["GET", "POST"])
def root():
    if request.method == "POST":
        transaction_text = request.form.get("transaction")
        query_vec = embedding_model.encode([transaction_text])
        D, I = faiss_index.search(query_vec, k=5)
        similar_examples = [id_to_text[str(i)] for i in I[0] if str(i) in id_to_text]

        prompt = prompt_template.format(transaction=transaction_text, examples="; ".join(similar_examples))
        result = gen_pipeline(prompt, max_length=100, do_sample=True, temperature=0.7)

        return render_template("index.html", transaction=transaction_text, examples=similar_examples, response=result[0]["generated_text"])

    return render_template("index.html")

# Start the app in CML environment
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))