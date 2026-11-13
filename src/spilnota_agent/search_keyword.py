"""Baseline: naive keyword search over listing text (no dictionary, no typo handling)."""

import re

TEXT_FIELDS = ("name", "description", "description_uk", "category", "city", "link")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words; drop 1-letter tokens."""
    return [token for token in re.findall(r"\w+", text.casefold()) if len(token) > 1]


def listing_text(listing: dict) -> str:
    """All searchable fields of a listing joined into one lowercase string."""
    return " ".join(str(listing.get(field) or "") for field in TEXT_FIELDS).casefold()


class KeywordSearch:
    def __init__(self, listings: list[dict]):
        self.listings = listings
        self.texts = {listing["id"]: listing_text(listing) for listing in listings}

    def search(self, query: str, city: str | None) -> list[str]:
        """Score = number of query tokens found as substrings; national listings pass any city filter."""
        tokens = tokenize(query)
        scored = []
        for listing in self.listings:
            if city and listing["city"] != city and listing["scope"] != "national":
                continue
            text = self.texts[listing["id"]]
            score = sum(1 for token in tokens if token in text)
            if score > 0:
                scored.append((score, listing["id"]))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [listing_id for _, listing_id in scored]
