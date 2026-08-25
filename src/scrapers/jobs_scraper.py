"""AI jobs scraper.

Job-board endpoints are loaded from config/sources.yaml. Each configured
source is normalized into the common raw-record shape and passed downstream
for freshness, deduplication, resolution and schema validation.

The scraper performs an early freshness check for records with a parseable
publication timestamp so stale candidates do not consume the requested job
limit. The pipeline performs the authoritative freshness check downstream as
a second safety layer.

Records without a reliable timestamp are still emitted so the pipeline can
apply its last-seen heuristic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date, datetime, timezone
from typing import Any

from src.core.source_registry import SourceDefinition
from src.scrapers.base import BaseScraper


_AI_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "ml",
    "llm",
    "large language model",
    "nlp",
    "natural language processing",
    "deep learning",
    "data scientist",
    "computer vision",
    "genai",
    "generative ai",
    "mlops",
)

_FRESHNESS_WINDOW_HOURS = 24


def _is_ai(text: str) -> bool:
    low = text.lower()
    return any(keyword in low for keyword in _AI_KEYWORDS)


def _parse_datetime(value: Any) -> datetime | None:
    """Parse common ISO/date representations.

    Unknown formats intentionally return None. Those records are allowed
    through so the downstream pipeline can apply its own date parser and
    heuristic.
    """

    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        dt = value

    elif isinstance(value, date):
        dt = datetime(
            value.year,
            value.month,
            value.day,
        )

    elif isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        normalized = text.replace(
            "Z",
            "+00:00",
        )

        try:
            dt = datetime.fromisoformat(
                normalized,
            )
        except ValueError:
            return None

    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc,
        )

    return dt.astimezone(
        timezone.utc,
    )


def _is_fresh_timestamp(
    value: Any,
    *,
    now: datetime | None = None,
) -> bool:
    """Return whether a parseable timestamp is within 24 hours.

    Missing or unparseable timestamps return True intentionally. Those records
    must reach the pipeline because the pipeline owns the authoritative
    freshness/last-seen heuristic.

    Future timestamps are not rejected here; the downstream freshness logic
    remains authoritative.
    """

    if value is None or value == "":
        return True

    published = _parse_datetime(value)

    if published is None:
        return True

    current = now or datetime.now(
        timezone.utc,
    )

    age_hours = (
        current - published
    ).total_seconds() / 3600

    return age_hours <= _FRESHNESS_WINDOW_HOURS


class JobsScraper(BaseScraper):
    source_name = "AI Jobs"

    def _fresh_enough(
        self,
        published_date: Any,
        *,
        source: SourceDefinition,
    ) -> bool:
        """Early-filter definitely stale records.

        This prevents stale records from consuming the global requested-job
        limit while leaving unknown/missing timestamps for the pipeline's
        authoritative freshness heuristic.
        """

        fresh = _is_fresh_timestamp(
            published_date,
        )

        if not fresh:
            self.log.info(
                "job_candidate_stale",
                source=source.name,
                published_date=str(published_date),
            )

        return fresh

    async def scrape(
        self,
        limit: int,
    ) -> AsyncIterator[dict]:
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

            source_name = source.name.casefold()

            if source_name == "remotive":
                async for record in self._remotive(
                    source,
                    limit - count,
                ):
                    yield record
                    count += 1

                    if count >= limit:
                        return

            elif source_name == "arbeitnow":
                async for record in self._arbeitnow(
                    source,
                    limit - count,
                ):
                    yield record
                    count += 1

                    if count >= limit:
                        return

            elif source_name == "hacker news hiring":
                async for record in self._hacker_news_hiring(
                    source,
                    limit - count,
                ):
                    yield record
                    count += 1

                    if count >= limit:
                        return

            elif source_name == "remoteok":
                async for record in self._remoteok(
                    source,
                    limit - count,
                ):
                    yield record
                    count += 1

                    if count >= limit:
                        return

            elif source_name == "greenhouse boards":
                self.log.info(
                    "greenhouse_source_requires_board_configuration",
                    source=source.name,
                )

            else:
                self.log.warning(
                    "unsupported_job_source",
                    source=source.name,
                )

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

            for job in resp.json().get(
                "jobs",
                [],
            ):
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

                published_date = job.get(
                    "publication_date",
                )

                if not self._fresh_enough(
                    published_date,
                    source=source,
                ):
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": job.get("title"),
                    "company": job.get("company_name"),
                    "published_date": published_date,
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
            resp = await self.fetcher.get(
                source.endpoint,
            )

            for job in resp.json().get(
                "data",
                [],
            ):
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

                published_date = job.get(
                    "created_at",
                )

                if not self._fresh_enough(
                    published_date,
                    source=source,
                ):
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": job.get("title"),
                    "company": job.get("company_name"),
                    "published_date": published_date,
                    "is_remote": bool(
                        job.get("remote")
                    ),
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
            resp = await self.fetcher.get(
                source.endpoint,
            )

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

                published_date = job.get(
                    "date",
                )

                if not self._fresh_enough(
                    published_date,
                    source=source,
                ):
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": job.get("position"),
                    "company": job.get("company"),
                    "published_date": published_date,
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

            for hit in resp.json().get(
                "hits",
                [],
            ):
                if limit <= 0:
                    return

                title = hit.get(
                    "title",
                    "",
                )

                text = (
                    hit.get("story_text", "")
                    or hit.get("comment_text", "")
                )

                blob = f"{title} {text}"

                if not _is_ai(blob):
                    continue

                object_id = hit.get(
                    "objectID",
                )

                if not object_id:
                    continue

                source_url = (
                    "https://news.ycombinator.com/item"
                    f"?id={object_id}"
                )

                published_date = hit.get(
                    "created_at",
                )

                if not self._fresh_enough(
                    published_date,
                    source=source,
                ):
                    continue

                yield {
                    "source_name": source.name,
                    "source_url": source_url,
                    "title": title,
                    "company": None,
                    "published_date": published_date,
                    "is_remote": False,
                    "description": text[:4000],
                }

                limit -= 1

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "hacker_news_hiring_failed",
                error=str(exc),
            )