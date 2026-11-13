"""Semantic search with Cohere Embed: listings and queries share one vector space."""

import hashlib
import json
import os
from pathlib import Path

import cohere
import numpy as np
from dotenv import load_dotenv

CACHE_DIR = Path(".cache/embeddings")
BATCH_SIZE = 96  # max texts per Cohere embed request


def document_text(listing: dict) -> str:
    """What the model sees for each listing: name, category, place and both descriptions."""
    parts = [
        listing["name"],
        f"Category: {listing['category']}",
        f"City: {listing['city'] or 'all of Canada'}",
        listing.get("description") or "",
        listing.get("description_uk") or "",
    ]
    return "\n".join(part for part in parts if part)


def cache_key(input_type: str, text: str) -> str:
    return hashlib.sha256(f"{input_type}\n{text}".encode()).hexdigest()


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors to length 1, so a dot product equals cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


class EmbedSearch:
    def __init__(self, listings: list[dict], model: str = "embed-v4.0"):
        load_dotenv()
        self.client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
        self.model = model
        self.listings = listings
        self.cache_path = CACHE_DIR / f"{model}.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

        vectors = self.embed([document_text(listing) for listing in listings], "search_document")
        self.doc_matrix = normalize(np.array(vectors))  # shape: (46, vector_size)

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Return one vector per text; call the API only for texts not in the cache."""
        keys = [cache_key(input_type, text) for text in texts]
        missing = [text for text, key in zip(texts, keys) if key not in self.cache]

        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start : start + BATCH_SIZE]
            response = self.client.embed(
                model=self.model,
                texts=batch,
                input_type=input_type,
                embedding_types=["float"],
            )
            for text, vector in zip(batch, response.embeddings.float_):
                self.cache[cache_key(input_type, text)] = vector

        if missing:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        return [self.cache[key] for key in keys]

    def search(self, query: str, city: str | None) -> list[str]:
        """Rank all listings by cosine similarity to the query, then apply the city filter."""
        query_vector = normalize(np.array(self.embed([query], "search_query")[0]))
        scores = self.doc_matrix @ query_vector  # 46 cosine similarities at once
        ranked = np.argsort(-scores)  # indices, highest score first
        return [
            self.listings[i]["id"]
            for i in ranked
            # Same rule as the keyword baseline. Duplicated on purpose: extract it once a third search needs it.
            if not city or self.listings[i]["city"] == city or self.listings[i]["scope"] == "national"
        ]
