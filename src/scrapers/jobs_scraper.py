"""AI jobs scraper.

Job-board endpoints are loaded from config/sources.yaml. Each configured
source is normalized into the common raw-record shape and passed downstream
for freshness, deduplication, resolution and schema validation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from src.core.source_registry import SourceDefinition
from src.scrapers.base import BaseScraper


_AI_KEYWORDS = (
    "ai",
    "machine learning",
    "ml",
    "llm",
    "nlp",
    "deep learning",
    "data scientist",
    "computer vision",
    "genai",
    "mlops",
)


def _is_ai(text: str) -> bool:
    low = text.lower()
    return any(keyword in low for keyword in _AI_KEYWORDS)


class JobsScraper(BaseScraper):
    source_name = "AI Jobs"

    async def scrape(self, limit: int) -> AsyncIterator[dict]:
        if limit <= 0:
            return

        count = 0

        for source in self.registry.get_all("jobs"):
            if count >= limit:
                return

            if not source.endpoint:
                self.log.warning(
                    "job_source_missing_endpoint",
                    source=source.name,
                )
                continue

            if source.name.casefold() == "remotive":
                async for record in self._remotive(source, limit - count):
                    yield record
                    count += 1

            elif source.name.casefold() == "arbeitnow":
                async for record in self._arbeitnow(source, limit - count):
                    yield record
                    count += 1

            elif source.name.casefold() == "hacker news hiring":
                async for record in self._hacker_news_hiring(
                    source,
                    limit - count,
                ):
                    yield record
                    count += 1

            elif source.name.casefold() == "remoteok":
                async for record in self._remoteok(source, limit - count):
                    yield record
                    count += 1

            elif source.name.casefold() == "greenhouse boards":
                self.log.info(
                    "greenhouse_source_requires_board_configuration",
                    source=source.name,
                )

            else:
                self.log.warning(
                    "unsupported_job_source",
                    source=source.name,
                )

            if count >= limit:
                return

    async def _remotive(
        self,
        source: SourceDefinition,
        limit: int,
    ) -> AsyncIterator[dict]:
        try:
            resp = await self.fetcher.get(
                source.endpoint,
                params={
                    "category": "software-dev",
                    "limit": "200",
                },
            )

            for job in resp.json().get("jobs", []):
                if limit <= 0:
                    return

                blob = (
                    f"{job.get('title', '')} "
                    f"{job.get('description', '')}"
                )

                if not _is_ai(blob):
                    continue

                source_url = job.get("url")

                if not source_url:
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": job.get("title"),
                    "company": job.get("company_name"),
                    "published_date": job.get("publication_date"),
                    "is_remote": True,
                    "description": (
                        job.get("description", "") or ""
                    )[:4000],
                }

                limit -= 1

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "remotive_failed",
                error=str(exc),
            )

    async def _arbeitnow(
        self,
        source: SourceDefinition,
        limit: int,
    ) -> AsyncIterator[dict]:
        try:
            resp = await self.fetcher.get(source.endpoint)

            for job in resp.json().get("data", []):
                if limit <= 0:
                    return

                blob = (
                    f"{job.get('title', '')} "
                    f"{job.get('description', '')}"
                )

                if not _is_ai(blob):
                    continue

                source_url = job.get("url")

                if not source_url:
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": job.get("title"),
                    "company": job.get("company_name"),
                    "published_date": job.get("created_at"),
                    "is_remote": bool(job.get("remote")),
                    "description": (
                        job.get("description", "") or ""
                    )[:4000],
                }

                limit -= 1

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "arbeitnow_failed",
                error=str(exc),
            )

    async def _remoteok(
        self,
        source: SourceDefinition,
        limit: int,
    ) -> AsyncIterator[dict]:
        try:
            resp = await self.fetcher.get(source.endpoint)

            payload = resp.json()

            if not isinstance(payload, list):
                return

            for job in payload:
                if limit <= 0:
                    return

                if not isinstance(job, dict):
                    continue

                blob = (
                    f"{job.get('position', '')} "
                    f"{job.get('description', '')}"
                )

                if not _is_ai(blob):
                    continue

                source_url = job.get("url")

                if not source_url:
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": job.get("position"),
                    "company": job.get("company"),
                    "published_date": job.get("date"),
                    "is_remote": True,
                    "description": (
                        job.get("description", "") or ""
                    )[:4000],
                }

                limit -= 1

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "remoteok_failed",
                error=str(exc),
            )

    async def _hacker_news_hiring(
        self,
        source: SourceDefinition,
        limit: int,
    ) -> AsyncIterator[dict]:
        try:
            resp = await self.fetcher.get(
                source.endpoint,
                params={
                    "query": "Who is hiring",
                    "tags": "story",
                    "hitsPerPage": "50",
                },
            )

            for hit in resp.json().get("hits", []):
                if limit <= 0:
                    return

                title = hit.get("title", "")
                text = hit.get("story_text", "") or hit.get("comment_text", "")

                blob = f"{title} {text}"

                if not _is_ai(blob):
                    continue

                object_id = hit.get("objectID")

                if not object_id:
                    continue

                source_url = (
                    f"https://news.ycombinator.com/item?id={object_id}"
                )

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": title,
                    "company": None,
                    "published_date": hit.get("created_at"),
                    "is_remote": False,
                    "description": text[:4000],
                }

                limit -= 1

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "hacker_news_hiring_failed",
                error=str(exc),
            )
