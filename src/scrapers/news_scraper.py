"""AI news signal scraper.

All source endpoints are resolved from config/sources.yaml. The scraper supports
both JSON APIs and RSS/Atom feeds and preserves the original source URL for
provenance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import feedparser

from src.scrapers.base import BaseScraper
from src.core.source_registry import SourceDefinition


class NewsScraper(BaseScraper):
    source_name = "AI News"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
        if limit <= 0:
            return

        count = 0

        for source in self.registry.get_all("news"):
            if count >= limit:
                return

            if not source.endpoint:
                self.log.warning(
                    "news_source_missing_endpoint",
                    source=source.name,
                )
                continue

            if source.source_type == "api":
                async for record in self._scrape_api_source(source, limit - count):
                    yield record
                    count += 1
                    if count >= limit:
                        return

            elif source.source_type in {"rss", "atom"}:
                async for record in self._scrape_feed_source(source, limit - count):
                    yield record
                    count += 1
                    if count >= limit:
                        return

            else:
                self.log.warning(
                    "unsupported_news_source_type",
                    source=source.name,
                    source_type=source.source_type,
                )

    async def _scrape_api_source(
        self,
        source: SourceDefinition,
        limit: int,
    ) -> AsyncIterator[dict]:
        if source.name.casefold() != "hacker news":
            self.log.warning(
                "unsupported_news_api",
                source=source.name,
            )
            return

        try:
            resp = await self.fetcher.get(
                source.endpoint,
                params={
                    "query": "AI",
                    "tags": "story",
                    "hitsPerPage": "50",
                },
            )

            for hit in resp.json().get("hits", []):
                if limit <= 0:
                    return

                object_id = hit.get("objectID")
                url = hit.get("url")

                if not url and object_id:
                    url = (
                        "https://news.ycombinator.com/item"
                        f"?id={object_id}"
                    )

                if not url:
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": url,
                    "title": hit.get("title", ""),
                    "published_date": hit.get("created_at"),
                }

                limit -= 1

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "news_api_failed",
                source=source.name,
                error=str(exc),
            )

    async def _scrape_feed_source(
        self,
        source: SourceDefinition,
        limit: int,
    ) -> AsyncIterator[dict]:
        try:
            resp = await self.fetcher.get(source.endpoint)
            feed = feedparser.parse(resp.text)

            for entry in feed.entries:
                if limit <= 0:
                    return

                source_url = entry.get("link")

                if not source_url:
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": entry.get("title", ""),
                    "published_date": (
                        entry.get("published")
                        or entry.get("updated")
                    ),
                    "summary": entry.get("summary", ""),
                }

                limit -= 1

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "rss_failed",
                source=source.name,
                error=str(exc),
            )
