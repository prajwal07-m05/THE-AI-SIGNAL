"""Startup and product directory scraper.

The Y Combinator source configuration comes from config/sources.yaml.

YC's public companies directory exposes the configuration required by its
public Algolia search endpoint. Production runs discover the current public
Algolia API key from the directory instead of relying on an expired
hard-coded key.

Tests can inject the public API key directly so they remain deterministic
and network-free.

No YC login or authentication is required.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from src.scrapers.base import BaseScraper


_ALGOLIA_APP_ID = "45BWZJ1SGC"

_ALGOLIA_HOST = (
    "https://45bwzj1sgc-dsn.algolia.net"
)

_YC_DIRECTORY_URL = (
    "https://www.ycombinator.com/companies"
)

_PAGE = 1000


_API_KEY_PATTERNS = (
    re.compile(
        r'"apiKey"\s*:\s*"([^"]+)"',
        re.IGNORECASE,
    ),
    re.compile(
        r'"api_key"\s*:\s*"([^"]+)"',
        re.IGNORECASE,
    ),
    re.compile(
        r'"algoliaApiKey"\s*:\s*"([^"]+)"',
        re.IGNORECASE,
    ),
    re.compile(
        r'"algolia_api_key"\s*:\s*"([^"]+)"',
        re.IGNORECASE,
    ),
    re.compile(
        r'"ALGOLIA_API_KEY"\s*:\s*"([^"]+)"',
        re.IGNORECASE,
    ),
)


def _extract_algolia_api_key(
    html: str,
) -> str | None:
    """Extract YC's public Algolia API key from directory HTML."""

    for pattern in _API_KEY_PATTERNS:
        match = pattern.search(html)

        if not match:
            continue

        api_key = match.group(1).strip()

        if api_key:
            return api_key

    return None


class StartupScraper(BaseScraper):
    source_name = "Y Combinator"

    def __init__(
        self,
        fetcher,
        registry=None,
        algolia_api_key: str | None = None,
    ) -> None:
        super().__init__(
            fetcher,
            registry=registry,
        )

        self._algolia_api_key = (
            algolia_api_key.strip()
            if algolia_api_key
            else None
        )

    async def _discover_algolia_api_key(self) -> str:
        """Discover YC's current public Algolia key."""

        response = await self.fetcher.get(
            _YC_DIRECTORY_URL,
        )

        api_key = _extract_algolia_api_key(
            response.text
        )

        if not api_key:
            raise RuntimeError(
                "Could not discover YC's public Algolia API key "
                "from the public companies directory."
            )

        return api_key

    async def _get_algolia_api_key(self) -> str:
        """Return an injected key or discover the live public key."""

        if self._algolia_api_key:
            return self._algolia_api_key

        return await self._discover_algolia_api_key()

    async def scrape(
        self,
        limit: int,
    ) -> AsyncIterator[dict[str, Any]]:
        if limit <= 0:
            return

        source = self.source(
            "startups_products",
            self.source_name,
        )

        if source.source_type != "algolia":
            raise RuntimeError(
                f"unsupported startup source type: {source.source_type}"
            )

        if not source.index:
            raise RuntimeError(
                f"startup source '{source.name}' has no configured index"
            )

        api_key = await self._get_algolia_api_key()

        endpoint = (
            f"{_ALGOLIA_HOST}/1/indexes/"
            f"{source.index}/query"
        )

        headers = {
            "X-Algolia-Application-Id": _ALGOLIA_APP_ID,
            "X-Algolia-API-Key": api_key,
            "Content-Type": "application/json",
        }

        fetched = 0
        page = 0

        while fetched < limit:
            body = json.dumps(
                {
                    "query": "",
                    "hitsPerPage": min(
                        _PAGE,
                        limit - fetched,
                    ),
                    "page": page,
                    "attributesToRetrieve": ["*"],
                }
            )

            try:
                response = await self.fetcher.post_json(
                    endpoint,
                    body,
                    headers,
                )
            except AttributeError:
                self.log.error(
                    "algolia_fetcher_missing_post_json",
                    source=source.name,
                )
                raise

            hits = response.get(
                "hits",
                [],
            )

            if not hits:
                self.log.info(
                    "yc_exhausted",
                    page=page,
                )
                return

            for hit in hits:
                if fetched >= limit:
                    return

                name = hit.get("name")
                slug = hit.get("slug")

                if not name or not slug:
                    self.log.warning(
                        "yc_record_missing_identity",
                        page=page,
                    )
                    continue

                source_url = (
                    "https://www.ycombinator.com/"
                    f"companies/{slug}"
                )

                yield {
                    "record_kind": "STARTUP",
                    "source_name": source.name,
                    "source_url": source_url,
                    "name": name,
                    "team_size": hit.get("team_size"),
                    "one_liner": hit.get(
                        "one_liner",
                        "",
                    ),
                    "long_description": (
                        hit.get(
                            "long_description",
                            "",
                        )
                        or ""
                    )[:4000],
                    "batch": hit.get("batch"),
                }

                yield {
                    "record_kind": "PRODUCT",
                    "source_name": source.name,
                    "source_url": source_url,
                    "startup_name": name,
                    "product_desc": hit.get(
                        "one_liner",
                        "",
                    ),
                }

                fetched += 1

            page += 1