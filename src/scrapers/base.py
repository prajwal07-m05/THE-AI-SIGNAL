"""Base scraper abstraction.

Scrapers are intentionally thin. They are responsible for acquiring raw
source data and preserving provenance. Configuration comes from the shared
source registry so endpoints can be changed without modifying scraper code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.core.http_client import AsyncFetcher
from src.core.logging import get_logger
from src.core.source_registry import SourceDefinition, SourceRegistry, get_source_registry


class BaseScraper(ABC):
    """Common scraper functionality."""

    source_name: str

    def __init__(
        self,
        fetcher: AsyncFetcher,
        registry: SourceRegistry | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.registry = registry or get_source_registry()
        self.log = get_logger(self.__class__.__name__)

    def source(
        self,
        vertical: str,
        name: str | None = None,
    ) -> SourceDefinition:
        """Resolve a configured source for this scraper."""
        return self.registry.get(
            vertical,
            name or self.source_name,
        )

    def required_source(
        self,
        vertical: str,
        name: str | None = None,
    ) -> SourceDefinition:
        """Resolve a source and require a configured endpoint."""
        return self.registry.require_endpoint(
            vertical,
            name or self.source_name,
        )

    @abstractmethod
    def scrape(self, limit: int) -> AsyncIterator[dict]:
        """Yield up to ``limit`` raw records."""
        ...
