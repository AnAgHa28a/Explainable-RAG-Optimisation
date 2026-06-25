from transformers import pipeline
from src.retrieval.faiss_index import model, index, sentences
from src.prompting.prompt_generator import build_basic_prompt

clf = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

query = "The movie was fantastic"

# use retrieved context (IMPORTANT)
context = "\n".join([
    "The movie is powerful and provocative.",
    "I loved this film",
    "Amazing acting and great story"
])

prompt = build_basic_prompt(query, context)

print("\nPROMPT:\n", prompt)

# IMPORTANT FIX: feed prompt, not raw query
result = clf(prompt)

print("\nPREDICTION:\n", result)