"""arXiv research-paper scraper.

Uses the official arXiv Atom API. Endpoint and category configuration are
loaded from the shared source registry so source configuration is not duplicated
inside scraper code.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import feedparser

from src.scrapers.base import BaseScraper

_PAGE = 100


class ArxivScraper(BaseScraper):
    source_name = "arXiv"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
        if limit <= 0:
            return

        source = self.required_source("research_papers")

        if not source.endpoint:
            raise RuntimeError("arXiv source has no configured endpoint")

        categories = source.categories or (
            "cs.AI",
            "cs.LG",
            "cs.CL",
            "cs.CV",
            "stat.ML",
        )

        search_query = " OR ".join(
            f"cat:{category}" for category in categories
        )

        fetched = 0
        start = 0

        while fetched < limit:
            batch = min(_PAGE, limit - fetched)

            params = {
                "search_query": search_query,
                "start": str(start),
                "max_results": str(batch),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            resp = await self.fetcher.get(
                source.endpoint,
                params=params,
            )

            feed = feedparser.parse(resp.text)

            if not feed.entries:
                self.log.info(
                    "arxiv_exhausted",
                    start=start,
                )
                break

            for entry in feed.entries:
                if fetched >= limit:
                    return

                source_url = entry.get("link")

                if not source_url:
                    self.log.warning(
                        "arxiv_record_missing_url",
                        title=entry.get("title", ""),
                    )
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": entry.get("title", "").replace(
                        "\n",
                        " ",
                    ).strip(),
                    "authors": [
                        author.name
                        for author in entry.get("authors", [])
                        if getattr(author, "name", None)
                    ],
                    "abstract": entry.get("summary", ""),
                    "published_date": entry.get("published"),
                    "candidate_links": [
                        link.get("href", "")
                        for link in entry.get("links", [])
                        if link.get("href")
                    ],
                }

                fetched += 1

            start += batch

            # arXiv requests a delay between API calls.
            if fetched < limit:
                await asyncio.sleep(3.0)
