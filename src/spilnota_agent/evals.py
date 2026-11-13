"""Evaluation harness: run a search function over eval cases, compute recall@k and MRR."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LISTINGS_PATH = Path("fixtures/listings.json")
CASES_PATH = Path("evals/cases.jsonl")

# Any search implementation: (query, city filter) -> listing ids, best match first.
SearchFn = Callable[[str, str | None], list[str]]


@dataclass
class Case:
    id: str
    query: str
    city: str | None
    relevant: set[str]  # full listing ids
    source: str  # "real" or "synthetic"


def load_listings(path: Path = LISTINGS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(listings: list[dict], path: Path = CASES_PATH) -> list[Case]:
    """Read cases.jsonl and expand short id prefixes to full listing ids."""
    all_ids = [item["id"] for item in listings]
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        relevant = set()
        for prefix in raw["relevant"]:
            matches = [i for i in all_ids if i.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"{raw['id']}: prefix {prefix!r} matched {len(matches)} listings")
            relevant.add(matches[0])
        cases.append(Case(raw["id"], raw["query"], raw["city"], relevant, raw["source"]))
    return cases


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of relevant listings that appear in the top k results."""
    found = relevant & set(ranked[:k])
    return len(found) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant result; 0 if none is returned."""
    for position, listing_id in enumerate(ranked, start=1):
        if listing_id in relevant:
            return 1 / position
    return 0.0


def evaluate(search: SearchFn, cases: list[Case], k: int = 5, verbose: bool = True) -> dict:
    """Run every case through `search`; print and return averages per source."""
    rows = []
    for case in cases:
        ranked = search(case.query, case.city)
        rows.append((case, recall_at_k(ranked, case.relevant, k), reciprocal_rank(ranked, case.relevant)))

    if verbose:
        for case, recall, rr in rows:
            print(f"{case.id:8} recall@{k}={recall:.2f}  rr={rr:.2f}  {case.query}")
        print()

    summary = {}
    for source in ("real", "synthetic", "all"):
        subset = [row for row in rows if source == "all" or row[0].source == source]
        mean_recall = sum(row[1] for row in subset) / len(subset)
        mrr = sum(row[2] for row in subset) / len(subset)
        summary[source] = {"n": len(subset), f"recall@{k}": mean_recall, "mrr": mrr}
        print(f"{source:10} n={len(subset):2}  recall@{k}={mean_recall:.3f}  MRR={mrr:.3f}")
    return summary
