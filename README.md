# spilnota-agent

Learning project: practical LLM/RAG engineering with [Cohere](https://cohere.com), applied to
[Spilnota](https://spilnota.ca) — a directory of Ukrainian-speaking businesses, groups and events in Canada.

Users search in Ukrainian, English and mixed/transliterated text ("барбер", "nails toronto",
"психолог калгарі"). The goal is to measure how far multilingual embeddings and reranking improve
on the current keyword search — with evals at every step.

## Roadmap

1. [x] Setup: Cohere SDK, API key, smoke test
2. [x] Fixtures: export approved listings to `fixtures/listings.json`
3. [ ] Evals: test cases from real search queries; recall@5, MRR
4. [ ] Baseline: keyword search over fixtures
5. [ ] Cohere Embed → vector search
6. [ ] Cohere Rerank on top
7. [ ] Tool-use agent with Command; port to AWS Bedrock

## Setup

Requires [uv](https://docs.astral.sh/uv/).

~~~bash
uv sync                      # create .venv and install dependencies
cp .env.example .env         # then add your Cohere API key
uv run scripts/check_cohere.py
~~~

## Step 1 — Smoke test

`scripts/check_cohere.py` embeds three listings and one query with `embed-v4.0` and ranks them
by cosine similarity. A Ukrainian query matches an English listing with no shared words:

~~~
Query: 'барбер торонто'

0.507  Barbershop in Toronto, men's haircuts and beard trims
0.252  Манікюр і педикюр у Калгарі, українська майстриня
0.095  Psychologist offering therapy sessions in Ukrainian
~~~

## Step 2 — Fixtures

`scripts/export_fixtures.py` pulls approved listings from the Supabase REST API (PostgREST) using the
publishable key, so Row Level Security applies exactly as for a site visitor. The `meta` column is
excluded on purpose. Output is committed to `fixtures/listings.json` so evals run without any keys.

~~~
Saved 46 listings to fixtures/listings.json
By type: {'social': 6, 'business': 31, 'event': 2, 'group': 5, 'tool': 2}
By city: {None: 8, 'Toronto': 29, 'Edmonton': 2, 'Vancouver': 2, 'Calgary': 5}
~~~

`city: None` are Canada-wide listings (`scope: "national"`). The dataset is Toronto-heavy (29/46),
which the eval set needs to account for.
