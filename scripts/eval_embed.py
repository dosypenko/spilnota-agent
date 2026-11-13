"""Run the eval set against Cohere Embed search. Usage: uv run scripts/eval_embed.py [model]"""

import sys

from spilnota_agent.evals import evaluate, load_cases, load_listings
from spilnota_agent.search_embed import EmbedSearch

model = sys.argv[1] if len(sys.argv) > 1 else "embed-v4.0"
listings = load_listings()
cases = load_cases(listings)
print(f"Model: {model}\n")
evaluate(EmbedSearch(listings, model=model).search, cases)
