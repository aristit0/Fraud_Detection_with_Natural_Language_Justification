from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os
import json
import torch

# Get absolute project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Flask app with correct template folder
app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, "templates"))

# Load embedding model (SentenceTransformer)
embedding_model_path = os.path.join(PROJECT_ROOT, "trained_fraud_model")
embedding_model = SentenceTransformer(embedding_model_path)

# Detect device (GPU or CPU)
device_id = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load text-generation model (Falcon-7B or local)
try:
    falcon_model_path = os.path.join(PROJECT_ROOT, "trained_falcon_model")  # LOCAL directory (optional)
    gen_pipeline = pipeline(
        "text-generation",
        model=falcon_model_path if os.path.exists(falcon_model_path) else "tiiuae/falcon-7b-instruct",
        device=device_id,
        torch_dtype=torch_dtype
    )
except Exception as e:
    print(f"⚠️ Failed to load Falcon model: {e}")
    raise

# Load FAISS index and ID-to-text mapping
faiss_index_path = os.path.join(PROJECT_ROOT, "model/embeddings_index.faiss")
id_to_text_path = os.path.join(PROJECT_ROOT, "model/id_to_text.json")

faiss_index = faiss.read_index(faiss_index_path)
with open(id_to_text_path) as f:
    id_to_text = json.load(f)

# Prompt template
prompt_template = (
    "A user just did the following transaction: \"{transaction}\". "
    "Here are similar past fraud examples: {examples}. "
    "Is this suspicious? Why?"
)

@app.route("/", methods=["GET", "POST"])
def root():
    if request.method == "POST":
        transaction_text = request.form.get("transaction")

        # Vectorize query
        query_vec = embedding_model.encode([transaction_text])
        D, I = faiss_index.search(np.array(query_vec).astype(np.float32), k=5)

        # Lookup similar examples
        similar_examples = [id_to_text[str(i)] for i in I[0] if str(i) in id_to_text]

        # Generate response
        prompt = prompt_template.format(transaction=transaction_text, examples="; ".join(similar_examples))
        result = gen_pipeline(prompt, max_length=100, do_sample=True, temperature=0.7)

        return render_template(
            "index.html",
            transaction=transaction_text,
            examples=similar_examples,
            response=result[0]["generated_text"]
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))