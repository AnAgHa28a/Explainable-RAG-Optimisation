import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data_loader import load_sst2
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("Loading data...")
sentences, labels = load_sst2()

sentences = sentences[:1000]   # keep it fast

print("Loading model...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    cache_folder="./model_cache",
    trust_remote_code=False
)

print("Encoding corpus...")
embeddings = model.encode(
    sentences,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

print("Building FAISS index...")
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Index size:", index.ntotal)

# ---------------- QUERY ----------------

query = "The movie was amazing and very enjoyable"

q_emb = model.encode([query])
q_emb = np.array(q_emb).astype("float32")

distances, indices = index.search(q_emb, 5)

print("\nQUERY:", query)
print("\nTOP MATCHES:\n")

for i in indices[0]:
    print("-", sentences[i])