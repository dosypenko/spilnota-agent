"""Embed retrieval + Cohere Rerank. Usage: uv run scripts/eval_rerank.py [embed_model] [rerank_model]"""

import sys

from spilnota_agent.evals import evaluate, load_cases, load_listings
from spilnota_agent.search_embed import EmbedSearch
from spilnota_agent.search_rerank import RerankSearch

embed_model = sys.argv[1] if len(sys.argv) > 1 else "embed-v4.0"
rerank_model = sys.argv[2] if len(sys.argv) > 2 else "rerank-v3.5"

listings = load_listings()
cases = load_cases(listings)
retriever = EmbedSearch(listings, model=embed_model).search
search = RerankSearch(listings, retriever=retriever, model=rerank_model).search

print(f"Retriever: {embed_model}, reranker: {rerank_model}\n")
evaluate(search, cases)
