"""Export approved Spilnota listings from Supabase to fixtures/listings.json."""

import json
import os
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

# Only the fields search needs. `meta` is left out on purpose: it may hold
# contact details or internal data that should not land in a public repo.
COLUMNS = [
    "id", "type", "category", "name", "description", "description_uk",
    "scope", "province", "city", "district", "link", "languages", "speaks_ukrainian",
]
OUTPUT = Path("fixtures/listings.json")


def fetch_approved_listings() -> list[dict]:
    """Query the Supabase REST API (PostgREST) for approved items."""
    response = httpx.get(
        f"{SUPABASE_URL}/rest/v1/items",
        params={
            "select": ",".join(COLUMNS),
            "status": "eq.approved",  # PostgREST filter syntax: column=operator.value
            "order": "id.asc",        # stable order -> clean git diffs on re-export
        },
        headers={"apikey": SUPABASE_KEY},
        timeout=30,
    )
    response.raise_for_status()  # turn 4xx/5xx into an exception
    return response.json()


def main() -> None:
    listings = fetch_approved_listings()
    if not listings:
        raise SystemExit("0 listings returned: check the key and RLS policies on `items`.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(listings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(listings)} listings to {OUTPUT}")
    print("By type:", dict(Counter(item["type"] for item in listings)))
    print("By city:", dict(Counter(item["city"] for item in listings)))


if __name__ == "__main__":
    main()
