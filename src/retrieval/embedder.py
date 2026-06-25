from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentence = "The movie was fantastic"

embedding = model.encode(sentence)

print("Embedding Dimension:", len(embedding))