"""Behavior tests for the Y Combinator startup/product scraper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.source_registry import SourceRegistry
from src.scrapers.startups_scraper import StartupScraper


class FakeFetcher:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.calls: list[tuple[str, str, dict]] = []

    async def post_json(self, url, body, headers=None):
        self.calls.append(
            (
                url,
                body,
                headers or {},
            )
        )
        return self.responses.pop(0)


def _registry(tmp_path: Path) -> SourceRegistry:
    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "research_papers": [],
                "startups_products": [
                    {
                        "name": "Y Combinator",
                        "type": "algolia",
                        "index": "ConfiguredIndex",
                    }
                ],
                "news": [],
                "jobs": [],
                "freshness_window_hours": 24,
            }
        ),
        encoding="utf-8",
    )

    return SourceRegistry(path)


def _scraper(
    tmp_path: Path,
    responses: list[dict],
) -> tuple[StartupScraper, FakeFetcher]:
    fetcher = FakeFetcher(responses)

    scraper = StartupScraper(
        fetcher,
        registry=_registry(tmp_path),
        algolia_api_key="test-algolia-api-key",
    )

    return scraper, fetcher


def _hit(
    *,
    name: str = "Example AI",
    slug: str = "example-ai",
) -> dict:
    return {
        "name": name,
        "slug": slug,
        "team_size": 12,
        "one_liner": "AI infrastructure for developers",
        "long_description": "A longer description of Example AI.",
        "batch": "W24",
    }


@pytest.mark.asyncio
async def test_yc_scraper_uses_configured_algolia_index(
    tmp_path: Path,
):
    scraper, fetcher = _scraper(
        tmp_path,
        [
            {
                "hits": [
                    _hit(),
                ]
            },
        ],
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 2
    assert len(fetcher.calls) == 1

    url, body, headers = fetcher.calls[0]

    assert (
        url
        == "https://45bwzj1sgc-dsn.algolia.net/1/indexes/"
        "ConfiguredIndex/query"
    )

    assert '"query": ""' in body
    assert '"hitsPerPage": 1' in body
    assert '"page": 0' in body
    assert '"attributesToRetrieve": ["*"]' in body

    assert (
        headers["X-Algolia-Application-Id"]
        == "45BWZJ1SGC"
    )
    assert (
        headers["X-Algolia-API-Key"]
        == "test-algolia-api-key"
    )
    assert (
        headers["Content-Type"]
        == "application/json"
    )


@pytest.mark.asyncio
async def test_yc_scraper_emits_startup_and_product(
    tmp_path: Path,
):
    scraper, _ = _scraper(
        tmp_path,
        [
            {
                "hits": [
                    _hit(),
                ]
            },
        ],
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 2

    startup, product = records

    assert startup["record_kind"] == "STARTUP"
    assert startup["source_name"] == "Y Combinator"
    assert (
        startup["source_url"]
        == "https://www.ycombinator.com/companies/example-ai"
    )
    assert startup["name"] == "Example AI"
    assert startup["team_size"] == 12
    assert startup["batch"] == "W24"

    assert product["record_kind"] == "PRODUCT"
    assert product["source_name"] == "Y Combinator"
    assert (
        product["source_url"]
        == "https://www.ycombinator.com/companies/example-ai"
    )
    assert product["startup_name"] == "Example AI"
    assert (
        product["product_desc"]
        == "AI infrastructure for developers"
    )


@pytest.mark.asyncio
async def test_yc_scraper_respects_limit(
    tmp_path: Path,
):
    scraper, fetcher = _scraper(
        tmp_path,
        [
            {
                "hits": [
                    _hit(
                        name="First AI",
                        slug="first-ai",
                    ),
                    _hit(
                        name="Second AI",
                        slug="second-ai",
                    ),
                ]
            },
        ],
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 2
    assert records[0]["name"] == "First AI"
    assert records[1]["startup_name"] == "First AI"

    assert len(fetcher.calls) == 1

    _, body, _ = fetcher.calls[0]

    assert '"hitsPerPage": 1' in body


@pytest.mark.asyncio
async def test_yc_scraper_skips_hits_without_identity(
    tmp_path: Path,
):
    scraper, _ = _scraper(
        tmp_path,
        [
            {
                "hits": [
                    {
                        "name": "Missing Slug",
                        "team_size": 4,
                    },
                    _hit(),
                ]
            },
        ],
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 2
    assert records[0]["record_kind"] == "STARTUP"
    assert records[0]["name"] == "Example AI"


@pytest.mark.asyncio
async def test_yc_scraper_stops_when_algolia_is_exhausted(
    tmp_path: Path,
):
    scraper, fetcher = _scraper(
        tmp_path,
        [
            {
                "hits": [],
            },
        ],
    )

    records = [
        record
        async for record in scraper.scrape(10)
    ]

    assert records == []
    assert len(fetcher.calls) == 1


@pytest.mark.asyncio
async def test_yc_scraper_uses_injected_key_without_directory_request(
    tmp_path: Path,
):
    scraper, fetcher = _scraper(
        tmp_path,
        [
            {
                "hits": [
                    _hit(),
                ]
            },
        ],
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert records

    assert len(fetcher.calls) == 1

    url, _, headers = fetcher.calls[0]

    assert url.startswith(
        "https://45bwzj1sgc-dsn.algolia.net/"
    )

    assert (
        headers["X-Algolia-API-Key"]
        == "test-algolia-api-key"
    )


@pytest.mark.asyncio
async def test_yc_scraper_rejects_non_algolia_source(
    tmp_path: Path,
):
    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "research_papers": [],
                "startups_products": [
                    {
                        "name": "Y Combinator",
                        "type": "api",
                        "endpoint": "https://example.com",
                    }
                ],
                "news": [],
                "jobs": [],
            }
        ),
        encoding="utf-8",
    )

    registry = SourceRegistry(path)
    fetcher = FakeFetcher([])

    scraper = StartupScraper(
        fetcher,
        registry=registry,
        algolia_api_key="test-algolia-api-key",
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported startup source type",
    ):
        [
            record
            async for record in scraper.scrape(1)
        ]