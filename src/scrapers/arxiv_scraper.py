"""arXiv research-paper scraper (Phase I — Research Papers vertical).

Uses the OFFICIAL arXiv Atom API (https://export.arxiv.org/api/query), which is
public, ToS-friendly, and paginated — the correct way to bulk-acquire papers
without scraping HTML or tripping anti-bot. We query the cs.AI / cs.LG / cs.CL
categories sorted by submission date (freshest first) and page through in
batches of 100 up to `limit`, so scaling to any N is a pure pagination change.

Every yielded record carries the real arXiv abstract URL as provenance. GitHub
correlation + live star counts are added downstream by github_metrics.py.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import feedparser

from src.scrapers.base import BaseScraper

_API = "http://export.arxiv.org/api/query"
_CATEGORIES = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML"
_PAGE = 100


class ArxivScraper(BaseScraper):
    source_name = "arXiv"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
        fetched = 0
        start = 0
        while fetched < limit:
            batch = min(_PAGE, limit - fetched)
            params = {
                "search_query": _CATEGORIES,
                "start": str(start),
                "max_results": str(batch),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = await self.fetcher.get(_API, params=params)
            feed = feedparser.parse(resp.text)
            if not feed.entries:
                self.log.info("arxiv_exhausted", start=start)
                break

            for entry in feed.entries:
                yield {
                    "source_name": self.source_name,
                    "source_url": entry.get("link"),
                    "title": entry.get("title", "").replace("\n", " ").strip(),
                    "authors": [a.name for a in entry.get("authors", [])],
                    "abstract": entry.get("summary", ""),
                    "published_date": entry.get("published"),
                    # arXiv exposes linked resources; a code repo often appears here.
                    "candidate_links": [
                        link.get("href", "")
                        for link in entry.get("links", [])
                        if link.get("href")
                    ],
                }
                fetched += 1

            start += batch
            # arXiv asks callers to wait ~3s between page requests. Be polite.
            await asyncio.sleep(3.0)
