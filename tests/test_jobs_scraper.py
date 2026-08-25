"""Behavior tests for the AI jobs scraper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.source_registry import SourceRegistry
from src.scrapers.jobs_scraper import JobsScraper


class FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data


class FakeFetcher:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get(
        self,
        url,
        *,
        params=None,
        headers=None,
    ):
        self.calls.append(
            {
                "url": url,
                "params": params or {},
            }
        )

        return self.responses.pop(0)


def _registry(
    tmp_path: Path,
    source: dict,
) -> SourceRegistry:
    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "research_papers": [],
                "startups_products": [],
                "news": [],
                "jobs": [source],
                "freshness_window_hours": 24,
            }
        ),
        encoding="utf-8",
    )

    return SourceRegistry(path)


@pytest.mark.asyncio
async def test_remotive_filters_for_ai_and_preserves_fields(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "jobs": [
                        {
                            "title": "Backend Developer",
                            "description": (
                                "Build ordinary web APIs."
                            ),
                            "url": (
                                "https://example.com/non-ai"
                            ),
                            "company_name": "Web Co",
                        },
                        {
                            "title": "LLM Engineer",
                            "description": (
                                "Build production LLM systems."
                            ),
                            "url": (
                                "https://example.com/llm"
                            ),
                            "company_name": "AI Co",
                            "publication_date": (
                                "2026-08-25T10:00:00Z"
                            ),
                        },
                    ]
                }
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "Remotive",
                "type": "api",
                "endpoint": (
                    "https://configured.example/remotive"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    record = records[0]

    assert record["source_name"] == "Remotive"
    assert record["source_url"] == (
        "https://example.com/llm"
    )
    assert record["title"] == "LLM Engineer"
    assert record["company"] == "AI Co"
    assert record["is_remote"] is True

    assert fetcher.calls[0]["url"] == (
        "https://configured.example/remotive"
    )

    assert fetcher.calls[0]["params"] == {
        "category": "software-dev",
        "limit": "200",
    }


@pytest.mark.asyncio
async def test_arbeitnow_filters_for_ai(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "data": [
                        {
                            "title": "Accountant",
                            "description": "Finance role.",
                            "url": (
                                "https://example.com/accountant"
                            ),
                        },
                        {
                            "title": (
                                "Machine Learning Engineer"
                            ),
                            "description": "Train ML models.",
                            "url": (
                                "https://example.com/ml"
                            ),
                            "company_name": "ML Co",
                            "created_at": (
                                "2026-08-25T09:00:00Z"
                            ),
                            "remote": True,
                        },
                    ]
                }
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "Arbeitnow",
                "type": "api",
                "endpoint": (
                    "https://configured.example/arbeitnow"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    record = records[0]

    assert record["source_name"] == "Arbeitnow"
    assert record["source_url"] == (
        "https://example.com/ml"
    )
    assert record["title"] == (
        "Machine Learning Engineer"
    )
    assert record["company"] == "ML Co"
    assert record["published_date"] == (
        "2026-08-25T09:00:00Z"
    )
    assert record["is_remote"] is True

    assert fetcher.calls[0]["url"] == (
        "https://configured.example/arbeitnow"
    )


@pytest.mark.asyncio
async def test_remoteok_filters_for_ai(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                [
                    {
                        "position": "Designer",
                        "description": "Design websites.",
                        "url": (
                            "https://example.com/design"
                        ),
                    },
                    {
                        "position": "GenAI Engineer",
                        "description": (
                            "Build generative AI products."
                        ),
                        "url": (
                            "https://example.com/genai"
                        ),
                        "company": "GenAI Co",
                        "date": "2026-08-25",
                    },
                ]
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "RemoteOK",
                "type": "api",
                "endpoint": (
                    "https://configured.example/remoteok"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    record = records[0]

    assert record["source_name"] == "RemoteOK"
    assert record["source_url"] == (
        "https://example.com/genai"
    )
    assert record["title"] == "GenAI Engineer"
    assert record["company"] == "GenAI Co"
    assert record["published_date"] == (
        "2026-08-25"
    )
    assert record["is_remote"] is True

    assert fetcher.calls[0]["url"] == (
        "https://configured.example/remoteok"
    )


@pytest.mark.asyncio
async def test_hacker_news_hiring_uses_configured_query(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "hits": [
                        {
                            "objectID": "12345",
                            "title": (
                                "Who is hiring? AI engineer"
                            ),
                            "story_text": (
                                "Looking for an ML engineer."
                            ),
                            "created_at": (
                                "2026-08-25T08:00:00Z"
                            ),
                        }
                    ]
                }
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "Hacker News Hiring",
                "type": "api",
                "endpoint": (
                    "https://configured.example/hiring"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    assert fetcher.calls[0]["url"] == (
        "https://configured.example/hiring"
    )

    assert fetcher.calls[0]["params"] == {
        "query": "Who is hiring",
        "tags": "story",
        "hitsPerPage": "50",
    }


@pytest.mark.asyncio
async def test_hacker_news_hiring_builds_item_url(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "hits": [
                        {
                            "objectID": "54321",
                            "title": "AI startup hiring",
                            "comment_text": (
                                "Hiring a computer vision engineer."
                            ),
                        }
                    ]
                }
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "Hacker News Hiring",
                "type": "api",
                "endpoint": (
                    "https://configured.example/hiring"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    record = records[0]

    assert record["source_name"] == (
        "Hacker News Hiring"
    )

    assert record["source_url"] == (
        "https://news.ycombinator.com/item?id=54321"
    )

    assert record["title"] == "AI startup hiring"
    assert record["company"] is None
    assert record["is_remote"] is False
    assert "computer vision" in record["description"]


@pytest.mark.asyncio
async def test_jobs_limit_is_global_across_sources(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "jobs": [
                        {
                            "title": "AI Engineer",
                            "description": "Build AI.",
                            "url": (
                                "https://example.com/first"
                            ),
                            "company_name": "First Co",
                        },
                        {
                            "title": "ML Engineer",
                            "description": "Build ML.",
                            "url": (
                                "https://example.com/second"
                            ),
                            "company_name": "Second Co",
                        },
                    ]
                }
            ),
            FakeResponse(
                {
                    "data": [],
                }
            ),
        ]
    )

    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "research_papers": [],
                "startups_products": [],
                "news": [],
                "jobs": [
                    {
                        "name": "Remotive",
                        "type": "api",
                        "endpoint": (
                            "https://configured.example/remotive"
                        ),
                    },
                    {
                        "name": "Arbeitnow",
                        "type": "api",
                        "endpoint": (
                            "https://configured.example/arbeitnow"
                        ),
                    },
                ],
                "freshness_window_hours": 24,
            }
        ),
        encoding="utf-8",
    )

    scraper = JobsScraper(
        fetcher,
        registry=SourceRegistry(path),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1
    assert records[0]["source_name"] == "Remotive"
    assert len(fetcher.calls) == 1


@pytest.mark.asyncio
async def test_jobs_skip_records_without_urls(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "jobs": [
                        {
                            "title": "AI Engineer",
                            "description": (
                                "Build AI systems."
                            ),
                            "company_name": "Missing URL Co",
                        },
                        {
                            "title": "LLM Engineer",
                            "description": (
                                "Build LLM systems."
                            ),
                            "url": (
                                "https://example.com/llm"
                            ),
                            "company_name": "Valid Co",
                        },
                    ]
                }
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "Remotive",
                "type": "api",
                "endpoint": (
                    "https://configured.example/remotive"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1
    assert records[0]["source_url"] == (
        "https://example.com/llm"
    )


@pytest.mark.asyncio
async def test_stale_jobs_do_not_consume_limit(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "jobs": [
                        {
                            "title": "Old AI Engineer",
                            "description": (
                                "Build old AI systems."
                            ),
                            "url": (
                                "https://example.com/old"
                            ),
                            "company_name": "Old Co",
                            "publication_date": (
                                "2026-08-20T10:00:00Z"
                            ),
                        },
                        {
                            "title": "Fresh LLM Engineer",
                            "description": (
                                "Build fresh LLM systems."
                            ),
                            "url": (
                                "https://example.com/fresh"
                            ),
                            "company_name": "Fresh Co",
                            "publication_date": (
                                "2026-08-25T18:00:00Z"
                            ),
                        },
                    ]
                }
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "Remotive",
                "type": "api",
                "endpoint": (
                    "https://configured.example/remotive"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    assert records[0]["title"] == (
        "Fresh LLM Engineer"
    )

    assert records[0]["source_url"] == (
        "https://example.com/fresh"
    )


@pytest.mark.asyncio
async def test_missing_timestamp_is_deferred_to_pipeline(
    tmp_path: Path,
):
    fetcher = FakeFetcher(
        [
            FakeResponse(
                {
                    "jobs": [
                        {
                            "title": "AI Engineer",
                            "description": (
                                "Build AI systems."
                            ),
                            "url": (
                                "https://example.com/unknown-date"
                            ),
                            "company_name": "Unknown Date Co",
                        }
                    ]
                }
            )
        ]
    )

    scraper = JobsScraper(
        fetcher,
        registry=_registry(
            tmp_path,
            {
                "name": "Remotive",
                "type": "api",
                "endpoint": (
                    "https://configured.example/remotive"
                ),
            },
        ),
    )

    records = [
        record
        async for record in scraper.scrape(1)
    ]

    assert len(records) == 1

    assert records[0]["published_date"] is None