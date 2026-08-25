"""Scraper base class.

Every scraper is an async generator that yields RAW dicts (never validated
records). The pipeline layer is responsible for LLM structuring, entity
resolution, freshness filtering, dedupe, and schema validation — keeping
scrapers thin and single-responsibility. Each raw dict MUST carry `source_name`
and `source_url` so provenance is preserved end-to-end (anti-hallucination).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.core.http_client import AsyncFetcher
from src.core.logging import get_logger


class BaseScraper(ABC):
    #: human-readable source label written to `source.name`
    source_name: str

    def __init__(self, fetcher: AsyncFetcher) -> None:
        self.fetcher = fetcher
        self.log = get_logger(self.__class__.__name__)

    @abstractmethod
    def scrape(self, limit: int) -> AsyncIterator[dict]:
        """Yield up to `limit` raw records as dicts."""
        ...
