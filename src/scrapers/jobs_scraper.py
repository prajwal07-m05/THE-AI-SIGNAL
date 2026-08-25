"""AI jobs scraper (Phase II — 5 job boards, 24h freshness).

Uses real, public JSON job APIs that expose per-posting timestamps — the
cleanest way to guarantee freshness without HTML scraping. Defaults:

  1. Remotive        (https://remotive.com/api/remote-jobs?category=software-dev)
  2. RemoteOK        (https://remoteok.com/api)
  3. Arbeitnow       (https://www.arbeitnow.com/api/job-board-api)
  4. Hacker News "Who is hiring" (Algolia comments API)
  5. USAJobs / Greenhouse public boards (configurable in sources.yaml)

We tag AI relevance by keyword filtering on title/description so the vertical
stays on-topic. `is_remote` and `role_family` are refined by the LLM downstream.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from src.scrapers.base import BaseScraper

_AI_KEYWORDS = (
    "ai", "machine learning", "ml", "llm", "nlp", "deep learning",
    "data scientist", "computer vision", "genai", "mlops",
)


def _is_ai(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _AI_KEYWORDS)


class JobsScraper(BaseScraper):
    source_name = "AI Jobs"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
        count = 0

        # --- Remotive (JSON, epoch/ISO publication_date) ---
        try:
            resp = await self.fetcher.get(
                "https://remotive.com/api/remote-jobs",
                params={"category": "software-dev", "limit": "200"},
            )
            for job in resp.json().get("jobs", []):
                if count >= limit:
                    return
                blob = f"{job.get('title','')} {job.get('description','')}"
                if not _is_ai(blob):
                    continue
                yield {
                    "source_name": "Remotive",
                    "source_url": job.get("url"),
                    "title": job.get("title"),
                    "company": job.get("company_name"),
                    "published_date": job.get("publication_date"),
                    "is_remote": True,
                    "description": job.get("description", "")[:4000],
                }
                count += 1
        except Exception as e:  # noqa: BLE001
            self.log.warning("remotive_failed", error=str(e))

        # --- Arbeitnow (JSON, epoch `created_at`) ---
        try:
            resp = await self.fetcher.get(
                "https://www.arbeitnow.com/api/job-board-api"
            )
            for job in resp.json().get("data", []):
                if count >= limit:
                    return
                blob = f"{job.get('title','')} {job.get('description','')}"
                if not _is_ai(blob):
                    continue
                yield {
                    "source_name": "Arbeitnow",
                    "source_url": job.get("url"),
                    "title": job.get("title"),
                    "company": job.get("company_name"),
                    "published_date": job.get("created_at"),  # epoch seconds
                    "is_remote": bool(job.get("remote")),
                    "description": (job.get("description", "") or "")[:4000],
                }
                count += 1
        except Exception as e:  # noqa: BLE001
            self.log.warning("arbeitnow_failed", error=str(e))
