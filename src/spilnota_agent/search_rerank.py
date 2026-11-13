"""Two-stage search: a retriever picks candidates, Cohere Rerank reorders them."""

import hashlib
import json
import os
import time
from pathlib import Path

import cohere
from dotenv import load_dotenv

from spilnota_agent.evals import SearchFn
from spilnota_agent.search_embed import document_text

CACHE_DIR = Path(".cache/rerank")
TRIAL_DELAY_SECONDS = 6.5  # stay under the low per-minute rerank limit of trial keys


class RerankSearch:
    def __init__(
        self,
        listings: list[dict],
        retriever: SearchFn,
        model: str = "rerank-v3.5",
        candidates: int = 20,
    ):
        load_dotenv()
        self.client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
        self.retriever = retriever
        self.model = model
        self.candidates = candidates
        self.texts = {listing["id"]: document_text(listing) for listing in listings}
        self.cache_path = CACHE_DIR / f"{model}.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def search(self, query: str, city: str | None) -> list[str]:
        """Rerank the retriever's top candidates; keep the rest in the retriever's order."""
        ranked = self.retriever(query, city)  # city filter is already applied here
        head, tail = ranked[: self.candidates], ranked[self.candidates :]

        key = hashlib.sha256(json.dumps([query, head]).encode()).hexdigest()
        if key not in self.cache:
            time.sleep(TRIAL_DELAY_SECONDS)
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=[self.texts[listing_id] for listing_id in head],
                top_n=len(head),
            )
            self.cache[key] = [result.index for result in response.results]
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))

        return [head[i] for i in self.cache[key]] + tail
