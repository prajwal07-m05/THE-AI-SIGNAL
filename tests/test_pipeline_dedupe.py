"""Pipeline-level tests for failure-safe deduplication claims."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.dedupe import DedupeLedger
from src.pipeline.runner import Pipeline


class FailingResolver:
    """Resolver that simulates a downstream processing failure."""

    def resolve(self, value: str):
        raise RuntimeError("resolver temporarily unavailable")


class FakeResolver:
    """Resolver that leaves values unchanged."""

    def resolve(self, value: str):
        class Result:
            canonical = None

        return Result()


def _job() -> dict:
    return {
        "schemaVersion": "1.0",
        "source": {
            "name": "Test Jobs",
            "url": "https://example.com/jobs/1",
        },
        "content": {
            "company": "Example AI",
            "date": None,
            "is_remote": True,
            "role_family": "AI Engineer",
            "title": "AI Engineer",
        },
    }


def _paper() -> dict:
    return {
        "schemaVersion": "1.0",
        "source": {
            "name": "Test Source",
            "url": "https://example.com/paper/1",
        },
        "content": {
            "title": "A reliable test paper",
            "authors": ["Author"],
            "paper_url": "https://example.com/paper/1",
            "github_url": None,
            "github_stars": None,
            "published_date": None,
        },
    }


@pytest.mark.asyncio
async def test_resolver_failure_releases_dedupe_claim(
    tmp_path: Path,
):
    ledger = DedupeLedger(
        sqlite_path=str(
            tmp_path / "ledger.sqlite"
        )
    )

    pipeline = Pipeline(
        dedupe=ledger,
        resolver=FailingResolver(),
        use_llm=False,
    )

    record = _job()
    identity = pipeline._identity(
        "jobs",
        record,
    )

    await pipeline._process_candidate(
        "jobs",
        record,
    )

    assert pipeline.output["jobs"] == []
    assert pipeline.stats["quarantined"] == 1
    assert (
        pipeline.quarantine[0]["reason"]
        == "entity_resolution_error"
    )
    assert ledger.first_seen(identity) is None

    ledger.close()


@pytest.mark.asyncio
async def test_successful_processing_keeps_dedupe_claim(
    tmp_path: Path,
):
    ledger = DedupeLedger(
        sqlite_path=str(
            tmp_path / "ledger.sqlite"
        )
    )

    pipeline = Pipeline(
        dedupe=ledger,
        resolver=FakeResolver(),
        use_llm=False,
    )

    record = _paper()
    identity = pipeline._identity(
        "papers",
        record,
    )

    await pipeline._process_candidate(
        "papers",
        record,
    )

    assert len(pipeline.output["papers"]) == 1
    assert pipeline.stats["accepted"] == 1
    assert ledger.first_seen(identity) is not None

    ledger.close()


@pytest.mark.asyncio
async def test_validation_failure_releases_dedupe_claim(
    tmp_path: Path,
):
    ledger = DedupeLedger(
        sqlite_path=str(
            tmp_path / "ledger.sqlite"
        )
    )

    pipeline = Pipeline(
        dedupe=ledger,
        resolver=FakeResolver(),
        use_llm=False,
    )

    record = _paper()
    record["content"]["title"] = None

    identity = pipeline._identity(
        "papers",
        record,
    )

    await pipeline._process_candidate(
        "papers",
        record,
    )

    assert pipeline.output["papers"] == []
    assert pipeline.stats["invalid"] == 1
    assert (
        pipeline.quarantine[0]["reason"]
        == "schema_validation_failed"
    )
    assert ledger.first_seen(identity) is None

    ledger.close()


@pytest.mark.asyncio
async def test_released_record_can_be_processed_again(
    tmp_path: Path,
):
    ledger = DedupeLedger(
        sqlite_path=str(
            tmp_path / "ledger.sqlite"
        )
    )

    failing_pipeline = Pipeline(
        dedupe=ledger,
        resolver=FailingResolver(),
        use_llm=False,
    )

    await failing_pipeline._process_candidate(
        "jobs",
        _job(),
    )

    assert failing_pipeline.output["jobs"] == []
    assert failing_pipeline.stats["quarantined"] == 1

    successful_pipeline = Pipeline(
        dedupe=ledger,
        resolver=FakeResolver(),
        use_llm=False,
    )

    await successful_pipeline._process_candidate(
        "jobs",
        _job(),
    )

    assert len(
        successful_pipeline.output["jobs"]
    ) == 1
    assert successful_pipeline.stats["accepted"] == 1

    ledger.close()
