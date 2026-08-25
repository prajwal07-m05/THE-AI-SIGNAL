"""AI news signal scraper (Phase II — 5 distinct sources, 24h freshness).

Sources are real, high-signal AI feeds with stable RSS/Atom or public JSON APIs
(no anti-bot needed for these; the Playwright tier is reserved for protected
domains). Freshness is enforced downstream by the freshness module against each
item's real published date.

The five sources below are configurable in config/sources.yaml; defaults:
  1. Hacker News front-page AI stories (Algolia API, real timestamps)
  2. MIT Technology Review — AI (RSS)
  3. VentureBeat — AI (RSS)
  4. TechCrunch — AI (RSS)
  5. The Verge — AI (RSS)
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import feedparser

from src.scrapers.base import BaseScraper

_RSS_SOURCES: dict[str, str] = {
    "MIT Technology Review": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "VentureBeat": "https://venturebeat.com/category/ai/feed/",
    "TechCrunch": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
}
_HN_API = "https://hn.algolia.com/api/v1/search_by_date"


class NewsScraper(BaseScraper):
    source_name = "AI News"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
        count = 0

        # --- Source 1: Hacker News (real epoch timestamps, JSON API) ---
        try:
            resp = await self.fetcher.get(
                _HN_API, params={"query": "AI", "tags": "story", "hitsPerPage": "50"}
            )
            for hit in resp.json().get("hits", []):
                if count >= limit:
                    return
                url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
                yield {
                    "source_name": "Hacker News",
                    "source_url": url,
                    "title": hit.get("title", ""),
                    "published_date": hit.get("created_at"),  # ISO-8601
                }
                count += 1
        except Exception as e:  # noqa: BLE001
            self.log.warning("hn_failed", error=str(e))

        # --- Sources 2-5: RSS feeds ---
        for name, feed_url in _RSS_SOURCES.items():
            try:
                resp = await self.fetcher.get(feed_url)
                feed = feedparser.parse(resp.text)
                for entry in feed.entries:
                    if count >= limit:
                        return
                    yield {
                        "source_name": name,
                        "source_url": entry.get("link"),
                        "title": entry.get("title", ""),
                        "published_date": entry.get("published")
                        or entry.get("updated"),
                        "summary": entry.get("summary", ""),
                    }
                    count += 1
            except Exception as e:  # noqa: BLE001
                self.log.warning("rss_failed", source=name, error=str(e))
