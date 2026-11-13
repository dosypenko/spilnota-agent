# spilnota-agent

Learning project: practical LLM/RAG engineering with [Cohere](https://cohere.com), applied to
[Spilnota](https://spilnota.ca) — a directory of Ukrainian-speaking businesses, groups and events in Canada.

Users search in Ukrainian, English and mixed/transliterated text ("барбер", "nails toronto",
"психолог калгарі"). The goal is to measure how far multilingual embeddings and reranking improve
on the current keyword search — with evals at every step.

## Roadmap

1. [x] Setup: Cohere SDK, API key, smoke test
2. [x] Fixtures: export approved listings to `fixtures/listings.json`
3. [x] Evals: test cases from real search queries; recall@5, MRR
4. [x] Baseline: keyword search over fixtures
5. [x] Cohere Embed → vector search
6. [x] Cohere Rerank on top
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

## Step 3 — Evals

`evals/cases.jsonl` holds 27 cases: 14 **real** queries from the production `search_events` log and
13 **synthetic** ones that cover cities and phrasings the log doesn't have yet. Metrics are reported
separately for each source, so synthetic cases can't inflate the real-world number.

Each case is a query, an optional city filter, and the listings that count as a correct answer.
Real queries include the messy things users actually type:

| Pattern | Example | Meaning |
|---|---|---|
| Wrong keyboard layout | `іещьф` | "stoma" typed on a Ukrainian layout |
| Truncated query | `дах торо` | "дах торонто" (roof Toronto) |
| Cross-lingual | `realtor` | listing name is in Ukrainian |
| Service, not name | `balloons` | balloon decor studio named MILAIRÉ |
| Known prod failure | `uamedia` | 0 results today: the term is only in the link |

`src/spilnota_agent/evals.py` runs any search function `(query, city) -> ranked ids` over the cases:

- **recall@5**: share of correct listings in the top 5 (about one phone screen)
- **MRR**: mean of 1 / position of the first correct listing

The harness was checked against an oracle search (1.000 / 1.000) and an empty one (0.000 / 0.000).

## Step 4 — Keyword baseline

`src/spilnota_agent/search_keyword.py`: lowercase, split into words, count query words found as
substrings in name, both descriptions, category, city and link. No dictionary, no typo handling.

| Set | recall@5 | MRR |
|---|---|---|
| real (14) | **0.607** | **0.589** |
| synthetic (13) | 1.000 | 0.904 |
| all (27) | 0.796 | 0.741 |

**Synthetic cases turned out too easy.** Every listing has both an English and a Ukrainian
description, so a query in either language usually shares words with one of them. The real-query
number is the one to beat.

Where keyword search fails on real queries:

- **Synonyms / meaning**: `realtor`, `ріелтор`, `software developer` share no words with the listing
- **Transliteration**: `stoma` (Latin) vs `стоматологія` (Cyrillic)
- **Keyboard layout**: `іещьф` is "stoma" typed on a Ukrainian layout
- **Ranking**: `калгарі` finds the right groups but ranks them 4th–5th, because many listings
  mention Calgary and ties fall back to file order

Side finding: `uamedia` returns 0 results in production but is found here, because the baseline
also searches the `link` field.

## Step 5 — Cohere Embed

`src/spilnota_agent/search_embed.py` embeds each listing once (name, category, city and both
descriptions) with `input_type="search_document"`, embeds the query with `"search_query"`, and ranks
by cosine similarity. Vectors are cached on disk by text hash, so re-runs cost no API calls.

| Real queries (14) | recall@5 | MRR |
|---|---|---|
| Keyword baseline | 0.607 | 0.589 |
| Cohere `embed-v4.0` | **0.857** | **0.859** |

Synthetic set: 1.000 / 1.000.

**Fixed by embeddings:** synonyms (`realtor`, `ріелтор` → a Ukrainian real-estate agent),
meaning without shared words (`software developer` → a Ukrainian IT community group), ranking
(`калгарі` now puts the Calgary community groups first), and partly transliteration (`stoma`).

**Not fixed:** `іещьф` — "stoma" typed on a Ukrainian keyboard layout — lands around position 33.
To a model it is noise; a plain layout-conversion function before search is the right tool here.

Caveat: with 14 real cases, one case moves recall by ~0.07. The direction is clear; the decimals are not.

## Step 6 — Cohere Rerank

`src/spilnota_agent/search_rerank.py`: two-stage search. The embedding retriever picks the top 20
candidates; Cohere Rerank (a cross-encoder that reads query and listing together) reorders them.
The retriever is injected as any `(query, city) -> ids` function, so Rerank runs on top of any search.

### Results — real queries (14)

| Approach | recall@5 | MRR |
|---|---|---|
| Keyword baseline | 0.607 | 0.589 |
| `embed-v4.0` | 0.857 | 0.859 |
| `embed-v5.0-fast` | 0.857 | 0.834 |
| `embed-v4.0` + `rerank-v3.5` | 0.857 | 0.931 |
| **`embed-v5.0-fast` + `rerank-v3.5`** | **0.893** | **0.931** |

All embedding and rerank setups score 1.000 recall@5 on the synthetic set.

**What the numbers say**

- **Embed finds, Rerank orders.** Rerank never changes which listings are candidates, so on its own
  it lifts MRR, not recall.
- **Same average, different mistakes.** `embed-v4.0` and `embed-v5.0-fast` tie on recall but fail on
  different queries. v5 had the right roofing company for `дах торо` at position 7; Rerank moved it to 1.
- **Rerank can hurt.** For `імміграц`, v5 alone returned both immigration listings in the top 5;
  after reranking, the PR CUAET group dropped out. Per-case results catch this; averages hide it.
- **Models aren't the fix for everything.** `іещьф` ("stoma" on a Ukrainian keyboard layout) is never
  even a candidate. A layout-conversion step before search is the right tool.
- **Small sample.** 0.857 → 0.893 is one query. Treat the gap as a signal, not a proof.
