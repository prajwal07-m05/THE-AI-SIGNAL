"""Behavior tests for the AI news scraper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.source_registry import SourceRegistry
from src.scrapers.news_scraper import NewsScraper


class FakeResponse:
    def __init__(self, *, json_data=None, text=""):
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class FakeFetcher:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get(self, url, *, params=None, headers=None):
        self.calls.append(
            {
                "url": url,
                "params": params or {},
            }
        )
        return self.responses.pop(0)


def _registry(
    tmp_path: Path,
    *,
    news_sources: list[dict],
) -> SourceRegistry:
    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "research_papers": [],
                "startups_products": [],
                "news": news_sources,
                "jobs": [],
                "freshness_window_hours": 24,
            }
        ),
        encoding="utf-8",
    )

    return SourceRegistry(path)


@pytest.mark.asyncio
async def test_hacker_news_api_uses_configured_endpoint(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                json_data={
                    "hits": [
                        {
                            "objectID": "123",
                            "url": "https://example.com/article",
                            "title": "AI changes everything",
                            "created_at": "2026-08-25T10:00:00Z",
                        }
                    ]
                }
            )
        ]
    )

    scraper = NewsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            news_sources=[
                {
                    "name": "Hacker News",
                    "type": "api",
                    "endpoint": "https://configured.example/hn",
                }
            ],
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    assert fetcher.calls[0]["url"] == (
        "https://configured.example/hn"
    )

    assert fetcher.calls[0]["params"] == {
        "query": "AI",
        "tags": "story",
        "hitsPerPage": "50",
    }


@pytest.mark.asyncio
async def test_hacker_news_preserves_provenance(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                json_data={
                    "hits": [
                        {
                            "objectID": "123",
                            "url": "https://example.com/article",
                            "title": "AI changes everything",
                            "created_at": "2026-08-25T10:00:00Z",
                        }
                    ]
                }
            )
        ]
    )

    scraper = NewsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            news_sources=[
                {
                    "name": "Hacker News",
                    "type": "api",
                    "endpoint": "https://configured.example/hn",
                }
            ],
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    record = records[0]

    assert record["source_name"] == "Hacker News"
    assert record["source_url"] == "https://example.com/article"
    assert record["title"] == "AI changes everything"
    assert (
        record["published_date"]
        == "2026-08-25T10:00:00Z"
    )


@pytest.mark.asyncio
async def test_hacker_news_falls_back_to_item_url(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                json_data={
                    "hits": [
                        {
                            "objectID": "987654",
                            "title": "AI story without URL",
                            "created_at": "2026-08-25T11:00:00Z",
                        }
                    ]
                }
            )
        ]
    )

    scraper = NewsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            news_sources=[
                {
                    "name": "Hacker News",
                    "type": "api",
                    "endpoint": "https://configured.example/hn",
                }
            ],
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1
    assert records[0]["source_url"] == (
        "https://news.ycombinator.com/item?id=987654"
    )


@pytest.mark.asyncio
async def test_rss_source_is_fetched_and_parsed(
    tmp_path: Path,
):
    rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example AI News</title>
    <item>
      <title>New AI breakthrough</title>
      <link>https://example.com/ai-breakthrough</link>
      <pubDate>Mon, 25 Aug 2026 10:00:00 GMT</pubDate>
      <description>Important AI news.</description>
    </item>
  </channel>
</rss>
"""

    fetcher = FakeFetcher(
        [
            FakeResponse(text=rss),
        ]
    )

    scraper = NewsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            news_sources=[
                {
                    "name": "Example Feed",
                    "type": "rss",
                    "endpoint": "https://configured.example/feed.xml",
                }
            ],
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    record = records[0]

    assert record["source_name"] == "Example Feed"
    assert (
        record["source_url"]
        == "https://example.com/ai-breakthrough"
    )
    assert record["title"] == "New AI breakthrough"
    assert record["published_date"]
    assert record["summary"] == "Important AI news."

    assert fetcher.calls[0]["url"] == (
        "https://configured.example/feed.xml"
    )


@pytest.mark.asyncio
async def test_news_limit_is_global_across_sources(
    tmp_path: Path,
):
    first_rss = """\
<rss version="2.0">
  <channel>
    <item>
      <title>First story</title>
      <link>https://example.com/first</link>
    </item>
    <item>
      <title>Second story</title>
      <link>https://example.com/second</link>
    </item>
  </channel>
</rss>
"""

    second_rss = """\
<rss version="2.0">
  <channel>
    <item>
      <title>Third story</title>
      <link>https://example.com/third</link>
    </item>
  </channel>
</rss>
"""

    fetcher = FakeFetcher(
        [
            FakeResponse(text=first_rss),
            FakeResponse(text=second_rss),
        ]
    )

    scraper = NewsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            news_sources=[
                {
                    "name": "Feed One",
                    "type": "rss",
                    "endpoint": "https://configured.example/one",
                },
                {
                    "name": "Feed Two",
                    "type": "rss",
                    "endpoint": "https://configured.example/two",
                },
            ],
        ),
    )

    records = [
        record
        async for record in scraper.scrape(2)
    ]

    assert len(records) == 2
    assert [
        record["title"]
        for record in records
    ] == [
        "First story",
        "Second story",
    ]

    assert len(fetcher.calls) == 1


@pytest.mark.asyncio
async def test_news_skips_entries_without_links(
    tmp_path: Path,
):
    rss = """\
<rss version="2.0">
  <channel>
    <item>
      <title>No link</title>
    </item>
    <item>
      <title>Valid story</title>
      <link>https://example.com/valid</link>
    </item>
  </channel>
</rss>
"""

    fetcher = FakeFetcher(
        [
            FakeResponse(text=rss),
        ]
    )

    scraper = NewsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            news_sources=[
                {
                    "name": "Example Feed",
                    "type": "rss",
                    "endpoint": "https://configured.example/feed.xml",
                }
            ],
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1
    assert records[0]["title"] == "Valid story"
