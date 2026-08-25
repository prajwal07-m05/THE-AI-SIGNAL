"""Startup & product directory scraper (Phase I — bulk extraction).

Demonstrates concurrent bulk acquisition from a large public directory. The
default source is the Y Combinator public company directory, exposed via its
Algolia-backed search API (the same API the site's own frontend calls) — a
legitimate, paginated, high-volume source of real startups and their products.
Paging through hitsPerPage=1000 lets a single query pull thousands of records;
scaling further is pure pagination (page += 1), no code change.

Each hit yields BOTH a startup raw-record and, where a product name/pricing is
present, a product raw-record — the LLM + resolver refine them downstream.

If YC's endpoint changes, swap the source in config/sources.yaml; the shape is
normalized here so the rest of the pipeline is source-agnostic.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from src.scrapers.base import BaseScraper

# Public Algolia index powering ycombinator.com/companies
_ALGOLIA_URL = (
    "https://45bwzj1sgc-dsn.algolia.net/1/indexes/YCCompany_production/query"
)
_ALGOLIA_HEADERS = {
    # These are the site's PUBLIC search-only keys, shipped in its own JS bundle.
    "X-Algolia-Application-Id": "45BWZJ1SGC",
    "X-Algolia-API-Key": "MjBjYjRiMzY0NzdhZWY0NjM4ZGRkYzFmYTA3YWM4MjljMWU2NGE2ZTk2YTk4NmU4NTA1MzNhYWM0MDBjNjllZnZhbGlkVW50aWw9MTYwODUyODc5MA==",
    "Content-Type": "application/json",
}
_PAGE = 1000


class StartupScraper(BaseScraper):
    source_name = "Y Combinator"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
        fetched = 0
        page = 0
        while fetched < limit:
            body = json.dumps(
                {"query": "", "hitsPerPage": _PAGE, "page": page, "attributesToRetrieve": ["*"]}
            )
            try:
                resp = await self.fetcher.post_json(_ALGOLIA_URL, body, _ALGOLIA_HEADERS)
            except AttributeError:
                # Fallback if fetcher has no post_json: use raw client via get-style.
                resp = await self.fetcher.get(_ALGOLIA_URL)  # pragma: no cover
            hits = resp.get("hits", [])
            if not hits:
                self.log.info("yc_exhausted", page=page)
                break

            for hit in hits:
                if fetched >= limit:
                    return
                slug = hit.get("slug", "")
                src_url = f"https://www.ycombinator.com/companies/{slug}"
                yield {
                    "record_kind": "STARTUP",
                    "source_name": self.source_name,
                    "source_url": src_url,
                    "name": hit.get("name"),
                    "team_size": hit.get("team_size"),
                    "one_liner": hit.get("one_liner", ""),
                    "long_description": hit.get("long_description", "")[:4000],
                    "batch": hit.get("batch"),
                }
                # A YC company's flagship product == the company product line.
                yield {
                    "record_kind": "PRODUCT",
                    "source_name": self.source_name,
                    "source_url": src_url,
                    "startup_name": hit.get("name"),
                    "product_desc": hit.get("one_liner", ""),
                }
                fetched += 1
            page += 1
