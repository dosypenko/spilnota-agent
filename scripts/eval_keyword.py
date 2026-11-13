"""Run the eval set against the keyword baseline."""

from spilnota_agent.evals import evaluate, load_cases, load_listings
from spilnota_agent.search_keyword import KeywordSearch

listings = load_listings()
cases = load_cases(listings)
evaluate(KeywordSearch(listings).search, cases)
