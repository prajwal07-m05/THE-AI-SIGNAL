"""Production pipeline orchestration.

The CLI owns the shared AsyncFetcher lifecycle and passes it into each
vertical. The pipeline owns the processing boundary:

    scraper
        -> provenance validation
        -> freshness filtering
        -> deduplication
        -> optional LLM enrichment
        -> entity resolution
        -> Pydantic validation
        -> output bucket

Scrapers remain source-specific and emit raw dictionaries. This module keeps
the rest of the system source-agnostic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from src.core.dedupe import DedupeLedger
from src.core.http_client import AsyncFetcher
from src.core.logging import get_logger
from src.freshness.date_parser import (
    heuristic_is_new,
    is_fresh,
    parse_date,
)
from src.llm.orchestrator import (
    AllProvidersFailed,
    LLMOrchestrator,
)
from src.models.schemas import (
    JobRecord,
    NewsRecord,
    ProductRecord,
    ResearchPaperRecord,
    StartupRecord,
)
from src.resolver.entity_resolver import EntityResolver
from src.scrapers.arxiv_scraper import ArxivScraper
from src.scrapers.jobs_scraper import JobsScraper
from src.scrapers.news_scraper import NewsScraper
from src.scrapers.startups_scraper import StartupScraper
from src.settings import get_settings

log = get_logger(__name__)


class Pipeline:
    """Coordinate ingestion, processing, validation and output."""

    OUTPUT_KEYS = (
        "startups",
        "products",
        "papers",
        "jobs",
        "news",
    )

    SCHEMAS = {
        "startups": StartupRecord,
        "products": ProductRecord,
        "papers": ResearchPaperRecord,
        "jobs": JobRecord,
        "news": NewsRecord,
    }

    def __init__(
        self,
        *,
        fetcher: AsyncFetcher | None = None,
        resolver: EntityResolver | None = None,
        dedupe: DedupeLedger | None = None,
        llm: LLMOrchestrator | None = None,
        use_llm: bool = True,
    ) -> None:
        self.settings = get_settings()

        self.fetcher = fetcher
        self.resolver = resolver or EntityResolver()
        self.dedupe = dedupe or DedupeLedger()

        self.llm = llm
        self.use_llm = bool(
            use_llm and self.settings.has_any_llm
        )

        self.output: dict[str, list[dict[str, Any]]] = {
            key: [] for key in self.OUTPUT_KEYS
        }

        self.quarantine: list[dict[str, Any]] = []

        self.stats: dict[str, Any] = {
            "seen": 0,
            "accepted": 0,
            "duplicates": 0,
            "stale": 0,
            "invalid": 0,
            "quarantined": 0,
            "llm_enriched": 0,
            "llm_failed": 0,
            "missing_provenance": 0,
            "normalization_failed": 0,
            "by_type": {
                key: {
                    "seen": 0,
                    "accepted": 0,
                    "duplicates": 0,
                    "stale": 0,
                    "invalid": 0,
                    "quarantined": 0,
                }
                for key in self.OUTPUT_KEYS
            },
        }

    # ------------------------------------------------------------------
    # Public API expected by src.main
    # ------------------------------------------------------------------

    async def run_papers(
        self,
        limit: int,
        fetcher: AsyncFetcher | None = None,
    ) -> None:
        """Run the research-paper vertical."""
        active_fetcher = self._resolve_fetcher(fetcher)

        scraper = ArxivScraper(active_fetcher)

        await self._consume(
            scraper.scrape(limit),
            normalizer=self._normalize_paper,
        )

    async def run_startups(
        self,
        limit: int,
        fetcher: AsyncFetcher | None = None,
    ) -> None:
        """Run the startup/product vertical."""
        active_fetcher = self._resolve_fetcher(fetcher)

        scraper = StartupScraper(active_fetcher)

        await self._consume(
            scraper.scrape(limit),
            normalizer=self._normalize_startup_or_product,
        )

    async def run_news(
        self,
        limit: int,
        fetcher: AsyncFetcher | None = None,
    ) -> None:
        """Run the news vertical."""
        active_fetcher = self._resolve_fetcher(fetcher)

        scraper = NewsScraper(active_fetcher)

        await self._consume(
            scraper.scrape(limit),
            normalizer=self._normalize_news,
        )

    async def run_jobs(
        self,
        limit: int,
        fetcher: AsyncFetcher | None = None,
    ) -> None:
        """Run the jobs vertical."""
        active_fetcher = self._resolve_fetcher(fetcher)

        scraper = JobsScraper(active_fetcher)

        await self._consume(
            scraper.scrape(limit),
            normalizer=self._normalize_job,
        )

    def _resolve_fetcher(
        self,
        fetcher: AsyncFetcher | None,
    ) -> AsyncFetcher:
        """Resolve the fetcher supplied by the CLI."""
        if fetcher is not None:
            self.fetcher = fetcher

        if self.fetcher is None:
            raise RuntimeError(
                "Pipeline requires an AsyncFetcher. "
                "Pass one to the run_* method."
            )

        return self.fetcher

    # ------------------------------------------------------------------
    # Raw ingestion
    # ------------------------------------------------------------------

    async def _consume(
        self,
        records: AsyncIterator[dict[str, Any]],
        *,
        normalizer: Any,
    ) -> None:
        async for raw in records:
            self.stats["seen"] += 1

            normalized = await self._safe_normalize(
                raw,
                normalizer,
            )

            if normalized is None:
                continue

            for record_type, candidate in normalized:
                if record_type not in self.OUTPUT_KEYS:
                    self._quarantine(
                        record_type,
                        candidate,
                        "unsupported_record_type",
                        f"Unsupported output type: {record_type}",
                    )
                    continue

                self.stats["by_type"][record_type]["seen"] += 1

                await self._process_candidate(
                    record_type,
                    candidate,
                )

    async def _safe_normalize(
        self,
        raw: dict[str, Any],
        normalizer: Any,
    ) -> list[tuple[str, dict[str, Any]]] | None:
        try:
            return await normalizer(raw)

        except Exception as exc:  # noqa: BLE001
            self.stats["normalization_failed"] += 1

            self._quarantine(
                self._infer_raw_type(raw),
                raw,
                "normalization_error",
                str(exc),
            )

            return None

    # ------------------------------------------------------------------
    # Candidate processing
    # ------------------------------------------------------------------

    async def _process_candidate(
        self,
        record_type: str,
        record: dict[str, Any],
    ) -> None:
        source = record.get("source")

        if not isinstance(source, dict):
            self.stats["missing_provenance"] += 1

            self._quarantine(
                record_type,
                record,
                "missing_provenance",
                "source object is required",
            )
            return

        source_name = self._clean_string(
            source.get("name")
        )

        source_url = self._clean_string(
            source.get("url")
        )

        if not source_name or not source_url:
            self.stats["missing_provenance"] += 1

            self._quarantine(
                record_type,
                record,
                "missing_provenance",
                "source.name and source.url are required",
            )
            return

        if record_type in {"news", "jobs"}:
            if not self._passes_freshness(
                record_type,
                record,
            ):
                self.stats["stale"] += 1
                self.stats["by_type"][record_type]["stale"] += 1
                return

        identity = self._identity(
            record_type,
            record,
        )

        try:
            is_new = self.dedupe.is_new(identity)

        except Exception as exc:  # noqa: BLE001
            self._quarantine(
                record_type,
                record,
                "dedupe_error",
                str(exc),
            )
            return

        if not is_new:
            self.stats["duplicates"] += 1
            self.stats["by_type"][record_type]["duplicates"] += 1
            return

        if self.use_llm and self.llm is not None:
            try:
                await self._apply_llm(
                    record_type,
                    record,
                )

                self.stats["llm_enriched"] += 1

            except AllProvidersFailed as exc:
                self.stats["llm_failed"] += 1

                self._quarantine(
                    record_type,
                    record,
                    "llm_failed",
                    str(exc),
                )
                return

            except Exception as exc:  # noqa: BLE001
                self.stats["llm_failed"] += 1

                self._quarantine(
                    record_type,
                    record,
                    "llm_error",
                    str(exc),
                )
                return

        self._resolve_entities(
            record_type,
            record,
        )

        try:
            validated = self._validate(
                record_type,
                record,
            )

        except ValidationError as exc:
            self.stats["invalid"] += 1
            self.stats["by_type"][record_type]["invalid"] += 1

            self._quarantine(
                record_type,
                record,
                "schema_validation_failed",
                str(exc),
            )
            return

        self.output[record_type].append(
            validated
        )

        self.stats["accepted"] += 1
        self.stats["by_type"][record_type]["accepted"] += 1

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    async def _normalize_paper(
        self,
        raw: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        title = self._clean_string(
            raw.get("title")
        )

        source_name = self._clean_string(
            raw.get("source_name")
        )

        source_url = self._clean_string(
            raw.get("source_url")
        )

        if not title or not source_name or not source_url:
            return []

        published = self._parse_datetime(
            raw.get("published_date")
        )

        github_url = self._find_github_url(
            raw.get("candidate_links")
        )

        return [
            (
                "papers",
                {
                    "schemaVersion": "1.0",
                    "source": self._source(
                        source_name,
                        source_url,
                    ),
                    "content": {
                        "title": title,
                        "authors": self._clean_string_list(
                            raw.get("authors")
                        ),
                        "paper_url": source_url,
                        "github_url": github_url,
                        "github_stars": None,
                        "published_date": published,
                    },
                },
            )
        ]

    async def _normalize_news(
        self,
        raw: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        title = self._clean_string(
            raw.get("title")
        )

        source_name = self._clean_string(
            raw.get("source_name")
        )

        source_url = self._clean_string(
            raw.get("source_url")
        )

        if not title or not source_name or not source_url:
            return []

        published = self._parse_datetime(
            raw.get("published_date")
        )

        return [
            (
                "news",
                {
                    "schemaVersion": "1.0",
                    "source": self._source(
                        source_name,
                        source_url,
                    ),
                    "content": {
                        "title": title,
                        "published_date": published,
                        "full_text": self._clean_string(
                            raw.get("full_text")
                            or raw.get("summary")
                        ),
                    },
                },
            )
        ]

    async def _normalize_job(
        self,
        raw: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        title = self._clean_string(
            raw.get("title")
        )

        company = self._clean_string(
            raw.get("company")
        )

        source_name = self._clean_string(
            raw.get("source_name")
        )

        source_url = self._clean_string(
            raw.get("source_url")
        )

        if not company or not source_name or not source_url:
            return []

        published = self._parse_datetime(
            raw.get("published_date")
        )

        return [
            (
                "jobs",
                {
                    "schemaVersion": "1.0",
                    "source": self._source(
                        source_name,
                        source_url,
                    ),
                    "content": {
                        "company": company,
                        "date": published,
                        "is_remote": bool(
                            raw.get("is_remote", False)
                        ),
                        "role_family": self._clean_string(
                            raw.get("role_family")
                        ),
                        "title": title,
                    },
                },
            )
        ]

    async def _normalize_startup_or_product(
        self,
        raw: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        record_kind = self._clean_string(
            raw.get("record_kind")
        )

        source_name = self._clean_string(
            raw.get("source_name")
        )

        source_url = self._clean_string(
            raw.get("source_url")
        )

        if not source_name or not source_url:
            return []

        source = self._source(
            source_name,
            source_url,
        )

        if record_kind == "STARTUP":
            name = self._clean_string(
                raw.get("name")
            )

            if not name:
                return []

            employee_count: int | None = None

            team_size = raw.get("team_size")

            if team_size is not None:
                try:
                    employee_count = int(team_size)
                except (TypeError, ValueError):
                    employee_count = None

            return [
                (
                    "startups",
                    {
                        "schemaVersion": "1.0",
                        "source": source,
                        "content": {
                            "entityName": name,
                            "data": {
                                "employeeCount": employee_count,
                            },
                        },
                    },
                )
            ]

        if record_kind == "PRODUCT":
            product_name = self._clean_string(
                raw.get("product_name")
                or raw.get("name")
            )

            startup_name = self._clean_string(
                raw.get("startup_name")
            )

            if not product_name or not startup_name:
                return []

            return [
                (
                    "products",
                    {
                        "schemaVersion": "1.0",
                        "source": source,
                        "content": {
                            "startupName": startup_name,
                            "pricingModel": self._parse_pricing_model(
                                raw.get("pricing_model")
                            ),
                        },
                    },
                )
            ]

        return []

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def _passes_freshness(
        self,
        record_type: str,
        record: dict[str, Any],
    ) -> bool:
        content = record.get("content")

        if not isinstance(content, dict):
            return False

        published = content.get(
            "published_date"
        )

        if published is None:
            published = content.get(
                "date"
            )

        if published is not None:
            return is_fresh(
                published,
                window_hours=self.settings.freshness_window_hours,
            )

        identity = self._identity(
            record_type,
            record,
        )

        seen_before = (
            self.dedupe.first_seen(identity)
            is not None
        )

        return heuristic_is_new(
            None,
            seen_before=seen_before,
            window_hours=self.settings.freshness_window_hours,
        )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    async def _apply_llm(
        self,
        record_type: str,
        record: dict[str, Any],
    ) -> None:
        if self.llm is None:
            return

        text = self._llm_text(
            record_type,
            record,
        )

        if not text:
            return

        extracted = await self.llm.extract(
            record_type,
            text,
        )

        if not isinstance(extracted, dict):
            raise ValueError(
                "LLM provider returned a non-object result"
            )

        content = record.setdefault(
            "content",
            {},
        )

        protected = {
            "schemaVersion",
            "recordType",
            "source",
            "collectedAt",
            "published_date",
            "date",
        }

        for key, value in extracted.items():
            if key in protected:
                continue

            if value is None:
                continue

            content[key] = value

    @staticmethod
    def _llm_text(
        record_type: str,
        record: dict[str, Any],
    ) -> str:
        content = record.get("content")

        if not isinstance(content, dict):
            return ""

        fields = {
            "startups": (
                "entityName",
            ),
            "products": (
                "startupName",
            ),
            "papers": (
                "title",
            ),
            "jobs": (
                "title",
                "company",
            ),
            "news": (
                "title",
                "full_text",
            ),
        }

        parts: list[str] = []

        for field in fields.get(
            record_type,
            (),
        ):
            value = content.get(field)

            if value is None:
                continue

            if isinstance(value, list):
                value = ", ".join(
                    str(item)
                    for item in value
                )

            text = str(value).strip()

            if text:
                parts.append(
                    f"{field}: {text}"
                )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _resolve_entities(
        self,
        record_type: str,
        record: dict[str, Any],
    ) -> None:
        content = record.get("content")

        if not isinstance(content, dict):
            return

        fields = {
            "startups": (
                "entityName",
            ),
            "products": (
                "startupName",
            ),
            "jobs": (
                "company",
            ),
        }.get(
            record_type,
            (),
        )

        for field in fields:
            value = content.get(field)

            if not isinstance(value, str):
                continue

            value = value.strip()

            if not value:
                continue

            result = self.resolver.resolve(
                value
            )

            if result.canonical:
                content[field] = result.canonical

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        record_type: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        schema = self.SCHEMAS[record_type]

        validated = schema.model_validate(
            record
        )

        return validated.model_dump(
            mode="json"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _source(
        name: str,
        url: str,
    ) -> dict[str, str]:
        return {
            "name": name.strip(),
            "url": url.strip(),
        }

    @staticmethod
    def _clean_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        text = str(value).strip()

        return text or None

    @staticmethod
    def _clean_string_list(
        value: Any,
    ) -> list[str]:
        if not isinstance(value, list):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    @staticmethod
    def _parse_datetime(
        value: Any,
    ) -> datetime | None:
        if value is None or value == "":
            return None

        if isinstance(value, datetime):
            dt = value
        else:
            dt = parse_date(value)

        if dt is None:
            return None

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    @staticmethod
    def _find_github_url(
        links: Any,
    ) -> str | None:
        if not isinstance(links, list):
            return None

        for link in links:
            if not isinstance(link, str):
                continue

            candidate = link.strip()

            if "github.com/" in candidate.lower():
                return candidate

        return None

    @staticmethod
    def _parse_pricing_model(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        candidate = str(value).strip().upper()

        allowed = {
            "FREE",
            "FREEMIUM",
            "PAID",
            "ENTERPRISE",
        }

        if candidate in allowed:
            return candidate

        return None

    @staticmethod
    def _identity(
        record_type: str,
        record: dict[str, Any],
    ) -> str:
        source = record["source"]

        url = str(
            source["url"]
        ).strip().lower()

        return f"{record_type}:{url}"

    @staticmethod
    def _infer_raw_type(
        raw: dict[str, Any],
    ) -> str:
        kind = str(
            raw.get("record_kind", "")
        ).upper()

        if kind == "STARTUP":
            return "startups"

        if kind == "PRODUCT":
            return "products"

        if "candidate_links" in raw:
            return "papers"

        if "company" in raw:
            return "jobs"

        return "news"

    def _quarantine(
        self,
        record_type: str,
        record: dict[str, Any],
        reason: str,
        detail: str,
    ) -> None:
        self.stats["quarantined"] += 1

        if record_type in self.stats["by_type"]:
            self.stats["by_type"][
                record_type
            ]["quarantined"] += 1

        self.quarantine.append(
            {
                "record_type": record_type,
                "reason": reason,
                "detail": detail,
                "record": record,
            }
        )

        log.warning(
            "record_quarantined",
            record_type=record_type,
            reason=reason,
            detail=detail,
        )