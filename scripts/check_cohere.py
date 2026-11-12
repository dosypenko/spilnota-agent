"""Smoke test: verify the Cohere API key works and preview multilingual embeddings."""

import math
import os

import cohere
from dotenv import load_dotenv

load_dotenv()  # read .env into os.environ

co = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])

MODEL = "embed-v4.0"

documents = [
    "Barbershop in Toronto, men's haircuts and beard trims",
    "Манікюр і педикюр у Калгарі, українська майстриня",
    "Psychologist offering therapy sessions in Ukrainian",
]
query = "барбер торонто"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: 1.0 = same direction (same meaning), ~0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


doc_vectors = co.embed(
    model=MODEL,
    texts=documents,
    input_type="search_document",
    embedding_types=["float"],
).embeddings.float_

query_vector = co.embed(
    model=MODEL,
    texts=[query],
    input_type="search_query",
    embedding_types=["float"],
).embeddings.float_[0]

print(f"Model: {MODEL}, vector size: {len(query_vector)}")
print(f"Query: {query!r}\n")

scores = [(cosine(query_vector, vec), doc) for doc, vec in zip(documents, doc_vectors)]
for score, doc in sorted(scores, reverse=True):
    print(f"{score:.3f}  {doc}")
