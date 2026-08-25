"""Tests proving scrapers consume the shared source registry."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.source_registry import SourceRegistry
from src.scrapers.arxiv_scraper import ArxivScraper
from src.scrapers.jobs_scraper import JobsScraper
from src.scrapers.news_scraper import NewsScraper
from src.scrapers.startups_scraper import StartupScraper


class DummyFetcher:
    async def get(self, *args, **kwargs):
        raise AssertionError("network should not be called by this test")

    async def post_json(self, *args, **kwargs):
        raise AssertionError("network should not be called by this test")


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
                "startups_products": [
                    {
                        "name": "Y Combinator",
                        "type": "algolia",
                        "index": "ConfiguredIndex",
                    }
                ],
                "news": [
                    {
                        "name": "Hacker News",
                        "type": "api",
                        "endpoint": "https://configured.example/hn",
                    },
                    {
                        "name": "Example Feed",
                        "type": "rss",
                        "endpoint": "https://configured.example/feed.xml",
                    },
                ],
                "jobs": [
                    {
                        "name": "Remotive",
                        "type": "api",
                        "endpoint": "https://configured.example/remotive",
                    },
                    {
                        "name": "RemoteOK",
                        "type": "api",
                        "endpoint": "https://configured.example/remoteok",
                    },
                ],
                "freshness_window_hours": 24,
            }
        ),
        encoding="utf-8",
    )

    return SourceRegistry(path)


def test_arxiv_uses_registry(tmp_path: Path):
    scraper = ArxivScraper(
        DummyFetcher(),
        registry=_registry(tmp_path),
    )

    source = scraper.required_source("research_papers")

    assert source.endpoint == "https://configured.example/arxiv"
    assert source.categories == ("cs.AI", "cs.LG")


def test_startups_use_registry_index(tmp_path: Path):
    scraper = StartupScraper(
        DummyFetcher(),
        registry=_registry(tmp_path),
    )

    source = scraper.source(
        "startups_products",
        "Y Combinator",
    )

    assert source.index == "ConfiguredIndex"
    assert source.source_type == "algolia"


def test_news_sources_are_registry_backed(tmp_path: Path):
    scraper = NewsScraper(
        DummyFetcher(),
        registry=_registry(tmp_path),
    )

    sources = scraper.registry.get_all("news")

    assert [source.name for source in sources] == [
        "Hacker News",
        "Example Feed",
    ]

    assert sources[0].endpoint == "https://configured.example/hn"
    assert sources[1].endpoint == "https://configured.example/feed.xml"


def test_jobs_sources_are_registry_backed(tmp_path: Path):
    scraper = JobsScraper(
        DummyFetcher(),
        registry=_registry(tmp_path),
    )

    sources = scraper.registry.get_all("jobs")

    assert [source.name for source in sources] == [
        "Remotive",
        "RemoteOK",
    ]

    assert sources[0].endpoint == "https://configured.example/remotive"
    assert sources[1].endpoint == "https://configured.example/remoteok"
