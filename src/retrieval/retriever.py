import faiss
import numpy as np

def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index, model, sentences, query, k=5):
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, k)

    return [sentences[i] for i in indices[0]]