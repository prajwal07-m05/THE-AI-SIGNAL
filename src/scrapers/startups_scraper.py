"""Startup and product directory scraper.

The Y Combinator source configuration comes from config/sources.yaml.
The public Algolia credentials remain implementation configuration because
the registry stores the source/index identity rather than secrets.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from src.core.source_registry import SourceDefinition
from src.scrapers.base import BaseScraper


_ALGOLIA_APP_ID = "45BWZJ1SGC"

_ALGOLIA_API_KEY = (
    "MjBjYjRiMzY0NzdhZWY0NjM4ZGRkYzFmYTA3YWM4MjljMWU2NGE2ZTk2YTk4NmU4NTA1MzNhYWM0MDBjNjllZnZhbGlkVW50aWw9MTYwODUyODc5MA=="
)

_PAGE = 1000


class StartupScraper(BaseScraper):
    source_name = "Y Combinator"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
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

        endpoint = (
            "https://45bwzj1sgc-dsn.algolia.net/1/indexes/"
            f"{source.index}/query"
        )

        headers = {
            "X-Algolia-Application-Id": _ALGOLIA_APP_ID,
            "X-Algolia-API-Key": _ALGOLIA_API_KEY,
            "Content-Type": "application/json",
        }

        fetched = 0
        page = 0

        while fetched < limit:
            body = json.dumps(
                {
                    "query": "",
                    "hitsPerPage": _PAGE,
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

            hits = response.get("hits", [])

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
                    f"https://www.ycombinator.com/companies/{slug}"
                )

                yield {
                    "record_kind": "STARTUP",
                    "source_name": source.name,
                    "source_url": source_url,
                    "name": name,
                    "team_size": hit.get("team_size"),
                    "one_liner": hit.get("one_liner", ""),
                    "long_description": (
                        hit.get("long_description", "") or ""
                    )[:4000],
                    "batch": hit.get("batch"),
                }

                yield {
                    "record_kind": "PRODUCT",
                    "source_name": source.name,
                    "source_url": source_url,
                    "startup_name": name,
                    "product_desc": hit.get("one_liner", ""),
                }

                fetched += 1

            page += 1
