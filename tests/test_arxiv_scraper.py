"""Behavior tests for the arXiv scraper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.source_registry import SourceRegistry
from src.scrapers.arxiv_scraper import ArxivScraper


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeFetcher:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, *, params=None, headers=None):
        self.calls.append((url, params or {}))
        return self.response


def _registry(tmp_path: Path) -> SourceRegistry:
    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "research_papers": [
                    {
                        "name": "arXiv",
                        "type": "api",
                        "endpoint": "https://configured.example/arxiv",
                        "categories": ["cs.AI", "cs.LG"],
                    }
                ],
                "startups_products": [],
                "news": [],
                "jobs": [],
                "freshness_window_hours": 24,
            }
        ),
        encoding="utf-8",
    )

    return SourceRegistry(path)


_ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2608.12345</id>
    <updated>2026-08-25T10:00:00Z</updated>
    <published>2026-08-25T09:00:00Z</published>
    <title>Example AI Paper</title>
    <summary>
      A useful paper with implementation at
      https://github.com/example/research-repo
    </summary>
    <author>
      <name>Jane Doe</name>
    </author>
    <link
      href="https://arxiv.org/abs/2608.12345"
      rel="alternate"
      type="text/html"
    />
    <link
      href="https://github.com/example/research-repo"
      rel="related"
    />
  </entry>
</feed>
"""


@pytest.mark.asyncio
async def test_arxiv_scraper_uses_configured_endpoint_and_categories(
    tmp_path: Path,
    monkeypatch,
):
    fetcher = FakeFetcher(FakeResponse(_ARXIV_XML))

    monkeypatch.setattr(
        "src.scrapers.arxiv_scraper.asyncio.sleep",
        lambda *_args, **_kwargs: _no_sleep(),
    )

    scraper = ArxivScraper(
        fetcher,
        registry=_registry(tmp_path),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    assert fetcher.calls[0][0] == (
        "https://configured.example/arxiv"
    )

    params = fetcher.calls[0][1]

    assert params["search_query"] == "cat:cs.AI OR cat:cs.LG"
    assert params["start"] == "0"
    assert params["max_results"] == "1"
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"


@pytest.mark.asyncio
async def test_arxiv_scraper_preserves_provenance_and_links(
    tmp_path: Path,
    monkeypatch,
):
    fetcher = FakeFetcher(FakeResponse(_ARXIV_XML))

    monkeypatch.setattr(
        "src.scrapers.arxiv_scraper.asyncio.sleep",
        lambda *_args, **_kwargs: _no_sleep(),
    )

    scraper = ArxivScraper(
        fetcher,
        registry=_registry(tmp_path),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    record = records[0]

    assert record["source_name"] == "arXiv"
    assert (
        record["source_url"]
        == "https://arxiv.org/abs/2608.12345"
    )
    assert record["title"] == "Example AI Paper"
    assert record["authors"] == ["Jane Doe"]
    assert "implementation" in record["abstract"]
    assert (
        "https://github.com/example/research-repo"
        in record["candidate_links"]
    )


@pytest.mark.asyncio
async def test_arxiv_scraper_respects_limit(
    tmp_path: Path,
    monkeypatch,
):
    xml = _ARXIV_XML.replace(
        "</feed>",
        """
  <entry>
    <id>http://arxiv.org/abs/2608.54321</id>
    <published>2026-08-25T08:00:00Z</published>
    <title>Second Paper</title>
    <summary>Second abstract.</summary>
    <author><name>John Smith</name></author>
    <link href="https://arxiv.org/abs/2608.54321" />
  </entry>
</feed>
""",
    )

    fetcher = FakeFetcher(FakeResponse(xml))

    monkeypatch.setattr(
        "src.scrapers.arxiv_scraper.asyncio.sleep",
        lambda *_args, **_kwargs: _no_sleep(),
    )

    scraper = ArxivScraper(
        fetcher,
        registry=_registry(tmp_path),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1


async def _no_sleep():
    return None
