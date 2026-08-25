# Frontier Intelligence Pipeline — Architecture & Production Design

*Export this file to `architecture.pdf` for submission (`Print → Save as PDF`, or
`pandoc docs/architecture.md -o architecture.pdf`). Kept ≤3 pages as required.*

---

## 0. System overview

A streaming producer/consumer pipeline. **Scrapers** are thin async generators
that yield raw dicts carrying provenance (`source_url`). A **processing layer**
runs each raw record through: distributed dedupe → LLM structuring → entity
resolution → freshness filtering → Pydantic schema validation → output. Memory
stays flat regardless of volume because records stream; nothing materializes the
full corpus. A single choke-point HTTP client enforces concurrency limits, rate
limiting, and retries uniformly.

**Stack:** `asyncio` + `httpx` (HTTP/2) for I/O; `aiolimiter` token bucket;
`tenacity` for backoff+jitter; Playwright (stealth) for the anti-bot tier;
Gemini/Groq/DeepSeek for extraction; `rapidfuzz` for resolution; Redis (or
SQLite) for the dedupe/freshness ledger; Pydantic for the schema contract.

---

## 1. Scale Strategy — collecting 500,000 records without manual intervention

**Principle: volume is an infrastructure parameter, never a code change.**

- **Source-appropriate acquisition.** Prefer official/structured endpoints
  (arXiv Atom API, Algolia-backed directories like YC, job-board JSON APIs,
  RSS) over HTML scraping. These paginate cheaply and don't trip anti-bot, so
  going from 1k → 500k is `page += 1` in a loop already written.
- **Horizontal fan-out.** The unit of work is a URL/record. Run N stateless
  crawler nodes (containers / K8s jobs / serverless workers) that pull work from
  a shared queue (Redis Streams / SQS) and share one dedupe ledger (§3). Adding
  throughput = adding replicas; no coordination code changes.
- **Bounded politeness at scale.** A global token bucket (`GLOBAL_RPS`) plus
  per-host semaphores mean we can be massively parallel *across* domains while
  never hammering *one* — the single biggest cause of bans at scale.
- **Backpressure.** Bounded `asyncio.Queue`s between producers and consumers
  keep memory constant and let slow stages (LLM) throttle fast stages (scrape)
  automatically.
- **Checkpointing.** The ledger records every claimed URL, so a killed run
  resumes without re-doing work — essential for multi-hour bulk jobs.

---

## 2. Handling 413s & 429s across thousands of concurrent extractions

**413 Payload Too Large (context overflow) — structural prevention.**
Before *every* LLM call we count tokens (`tiktoken`) and, if needed, apply
**salient head+tail truncation** to that provider's budget (`llm/chunking.py`),
so a 413 is normally impossible. Head+tail (not head-only) is used because
titles/authors/dates live at the top *and* bottom of documents. If a provider
still returns 413, the orchestrator **halves the budget and retries once**, then
falls through to the next provider.

**429 Too Many Requests — graceful degradation.**
Two layers: (1) the HTTP client and LLM orchestrator both use **exponential
backoff with full jitter** (`base 1s, cap 60s`) and **respect `Retry-After`**
when present. (2) After a bounded number of 429s on one provider, the
orchestrator **falls through the chain** (Gemini → Groq → DeepSeek) rather than
blocking — so a single provider's rate ceiling never stalls the fleet. Jitter is
critical: thousands of workers retrying in lockstep would create thundering-herd
429 storms; jitter de-synchronizes them.

**Provider fallback = availability.** Each provider is tried in priority order;
`ProviderUnavailable` (auth/quota/network) falls through immediately. If all
fail, the record is **quarantined** (counted, logged) — never emitted with
guessed values.

---

## 3. Freshness Tracking — never processing the same article/job twice

- **Content-addressed claim.** Identity = `sha1(normalized canonical URL)`.
  Claiming is an **atomic Redis `SET key NX`**: exactly one node across the
  entire distributed fleet wins the claim; everyone else sees "already seen" and
  skips. This is the deduplication guarantee across nodes. (SQLite `INSERT`
  with a PK constraint provides the same semantics for single-node runs.)
- **Freshness window.** All dates are normalized to timezone-aware UTC
  (`freshness/date_parser.py`), handling ISO/RFC, epoch seconds, and relative
  strings ("2 hours ago", "yesterday", "3d"). A record passes only if its
  publication date is within `FRESHNESS_WINDOW_HOURS` (24h).
- **Missing-date heuristic.** When a source exposes no reliable date, we fall
  back to the incremental-crawl heuristic: a URL **not previously in the ledger
  is treated as new since the last run**; anything already recorded is stale.
  The ledger's `first_seen` timestamp makes this decision deterministic and
  shareable across nodes.

---

## 4. Storage Strategy

- **Primary store — PostgreSQL.** Records are validated JSON with a stable
  schema and strong provenance requirements; Postgres gives ACID writes,
  `JSONB` for the flexible `content` payload, rich indexing on
  `recordType/collectedAt/source`, and easy upserts keyed on the canonical
  entity. It scales comfortably to the hundreds of millions of rows implied by
  "lakhs across many verticals" and is operationally boring (a virtue).
- **Dedupe / freshness ledger — Redis.** Sub-millisecond atomic `SET NX` is the
  right primitive for a hot, distributed "have we seen this?" check; TTLs can
  auto-expire the freshness window.
- **Relationship / graph layer — Neo4j (or Postgres + `pgvector` + a graph
  view).** The product is an *intelligence graph*: startups → founders →
  products → papers → repos → jobs are edges, not rows. A property graph makes
  "which papers by authors now at company X have >1k-star repos" a first-class
  traversal. `pgvector` embeddings power semantic entity resolution and
  similarity search that complements the deterministic resolver.
- **Object storage — S3/GCS** for raw HTML snapshots (audit trail: every record
  can be re-derived from its stored source, reinforcing the no-hallucination
  guarantee).

---

## 5. Anti-bot strategy (Phase V)

Tiered escalation, cheapest first: (0) `httpx` GET for APIs/directories — covers
~95%; (1) realistic headers + cookie warm-up; (2) **stealthed async Playwright**
(real Chromium, JS execution, browser-matching TLS/fingerprint, human-like
waits) for Cloudflare-managed-challenge / Datadome / heavy-JS pages
(`core/anti_bot.py`); (3) residential-proxy rotation + CAPTCHA-solver webhook
(seam via `PROXY_URL`, kept out of the trial run to respect ToS). We degrade
**politely** — no aggressive captcha defeat — because the goal is durable
intelligence, and burned IPs/domains cost more than they save.

---

## 6. Data-integrity guarantees (the disqualification clause)

1. `source.url` is injected by the scraper, never produced by the LLM.
2. The extraction system prompt forbids inference: absent field ⇒ `null`.
3. Every record is validated against a Pydantic schema pre-emission.
4. Failures are quarantined and counted, never emitted with fabricated values.
5. Raw source snapshots are retained so any record is re-derivable.
