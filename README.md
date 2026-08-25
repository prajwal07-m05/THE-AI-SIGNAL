# Frontier Intelligence Pipeline

A fault-tolerant, highly-concurrent ingestion pipeline that builds an **AI /
venture intelligence graph** from real public sources: startups, products, AI
research papers (with live GitHub star metrics), fresh AI news, and fresh AI
jobs. Raw content is structured into a canonical JSON schema by a **multi-tier
LLM extraction engine** and normalized by a **deterministic entity resolver**.

> **Data integrity:** Every emitted record traces back to a real `source.url`.
> The LLM is instructed to return `null` rather than guess, and every record is
> validated against a Pydantic schema before it is written — records that fail
> are *quarantined, never emitted*. No fabricated data.

---

## Architecture at a glance

```
                 ┌────────────┐   raw dicts   ┌──────────────────────────────┐
  Sources ─────► │  Scrapers  │ ────────────► │          Pipeline            │
  (arXiv, YC,    │ (async gen)│               │  dedupe → LLM structure →    │
   RSS, job      └────────────┘               │  entity resolve → freshness  │
   APIs)              │                        │  → schema validate → emit    │
                      ▼                        └──────────────────────────────┘
             ┌──────────────────┐                          │
             │  AsyncFetcher    │  global + per-host        ▼
             │  semaphores,     │  token bucket,     ┌──────────────┐
             │  backoff+jitter, │  Retry-After       │ OutputWriter │
             │  429/5xx retries │                    │ Sheets / CSV │
             └──────────────────┘                    └──────────────┘
                      │  escalates to
                      ▼
             ┌──────────────────┐
             │ PlaywrightRenderer│  stealthed Chromium for Cloudflare /
             │  (anti-bot tier) │  Datadome / JS-heavy domains
             └──────────────────┘
```

Full design rationale (scale to 500k, 413/429 strategy, freshness across
distributed nodes, storage choices) is in **[`docs/architecture.md`](docs/architecture.md)**
— export it to `architecture.pdf` for submission.

---

## Mapping to the assignment

| Phase | Requirement | Where |
|------|-------------|-------|
| I | Concurrent bulk scraper, scales by pagination only | `core/http_client.py`, `scrapers/*` |
| I | Research papers + GitHub stars | `scrapers/arxiv_scraper.py`, `scrapers/github_metrics.py` |
| II | 5 news + 5 job sources, 24h freshness | `scrapers/news_scraper.py`, `scrapers/jobs_scraper.py` |
| II | Date normalization + relative dates + heuristic | `freshness/date_parser.py` |
| III | LLM fallback chain Gemini→Groq→DeepSeek | `llm/orchestrator.py`, `llm/providers.py` |
| III | 413 chunking + 429 backoff/jitter | `llm/chunking.py`, `llm/orchestrator.py` |
| IV | Deterministic entity resolution (50-entity seed) | `resolver/entity_resolver.py`, `resolver/seed_entities.py` |
| V | Async operation + anti-bot | `core/http_client.py`, `core/anti_bot.py` |
| VI | Architecture doc | `docs/architecture.md` |
| Out | 6-tab Google Sheet / CSV | `output/sheets_writer.py` |

---

## Setup

```bash
git clone <your-repo-url> && cd frontier-intel-pipeline
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium                            # only needed for anti-bot tier

cp .env.example .env                                   # then fill in keys
```

### Required / optional keys (`.env`)
- **LLM (≥1 required):** `GEMINI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`
- **GitHub (recommended):** `GITHUB_TOKEN` — raises limit 60 → 5,000 req/hr
- **Sheets (optional):** `GOOGLE_SERVICE_ACCOUNT_JSON`, `GOOGLE_SHEET_ID`
  (share the sheet with the service-account email). If unset → CSVs in `./out/`.
- **Redis (optional):** `REDIS_URL` for distributed dedupe; else local SQLite.

---

## Run

```bash
# Trial targets: 1k startups, 1k products, 1k papers + all fresh news/jobs
python -m src.main run --startups 1000 --products 1000 --papers 1000

# Fast smoke test, no LLM/keys needed (structured sources are already clean):
python -m src.main run --papers 25 --no-startups --no-llm

# Single vertical:
python -m src.main run --papers 200 --no-startups --no-news --no-jobs
```

Output → six CSVs in `./out/` (or the six tabs of your Google Sheet):
`Startups`, `Products`, `Research Papers`, `Jobs`, `News`, `Entity Mapping Log`.

---

## Tests

```bash
pytest            # deterministic core: resolver, freshness, chunking
```

The resolver test proves the assignment's canonical example:
`"OpenAI" / "OpenAI, Inc." / "Open AI" → "OpenAI"`.

---

## Scaling to 500k (no code changes)

Everything that governs volume is env/config-driven (`MAX_CONCURRENCY`,
`GLOBAL_RPS`, pagination limits, `REDIS_URL`). To go from 1k to 500k you scale
*infrastructure*: add crawler nodes sharing one Redis dedupe ledger, raise
concurrency, and let each vertical page further. See `docs/architecture.md`.
