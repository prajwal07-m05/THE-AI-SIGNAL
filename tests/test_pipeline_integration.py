"""End-to-end pipeline boundary tests.

These tests exercise the processing chain across normalization,
provenance, freshness, deduplication, entity resolution and schema
validation rather than testing each component in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.dedupe import DedupeLedger
from src.pipeline.runner import Pipeline


class FakeResolver:
    def __init__(self, canonical: str | None = None):
        self.canonical = canonical

    def resolve(self, value: str):
        class Result:
            pass

        result = Result()
        result.canonical = self.canonical
        return result


class FailingResolver:
    def resolve(self, value: str):
        raise RuntimeError("resolver unavailable")


def _pipeline(
    tmp_path: Path,
    *,
    resolver=None,
    use_llm: bool = False,
) -> Pipeline:
    return Pipeline(
        dedupe=DedupeLedger(
            sqlite_path=str(
                tmp_path / "ledger.sqlite"
            )
        ),
        resolver=resolver or FakeResolver(),
        use_llm=use_llm,
    )


def _paper() -> dict:
    return {
        "schemaVersion": "1.0",
        "source": {
            "name": "Test Source",
            "url": "https://example.com/paper/1",
        },
        "content": {
            "title": "Reliable AI Systems",
            "authors": ["Author"],
            "paper_url": "https://example.com/paper/1",
            "github_url": None,
            "github_stars": None,
            "published_date": None,
        },
    }


def _startup() -> dict:
    return {
        "schemaVersion": "1.0",
        "source": {
            "name": "Test YC",
            "url": "https://www.ycombinator.com/companies/example-ai",
        },
        "content": {
            "entityName": "Example AI",
            "data": {
                "employeeCount": 12,
            },
        },
    }


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


def _news() -> dict:
    return {
        "schemaVersion": "1.0",
        "source": {
            "name": "Test News",
            "url": "https://example.com/news/1",
        },
        "content": {
            "title": "AI industry update",
            "published_date": None,
            "full_text": "Important AI developments.",
        },
    }


@pytest.mark.asyncio
async def test_valid_paper_flows_to_accepted_output(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    await pipeline._process_candidate(
        "papers",
        _paper(),
    )

    assert len(pipeline.output["papers"]) == 1
    assert pipeline.stats["seen"] == 0
    assert pipeline.stats["accepted"] == 1
    assert pipeline.stats["quarantined"] == 0
    assert pipeline.quarantine == []


@pytest.mark.asyncio
async def test_missing_provenance_is_quarantined(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    record = _paper()
    del record["source"]["url"]

    await pipeline._process_candidate(
        "papers",
        record,
    )

    assert pipeline.output["papers"] == []
    assert pipeline.stats["missing_provenance"] == 1
    assert pipeline.stats["quarantined"] == 1
    assert (
        pipeline.quarantine[0]["reason"]
        == "missing_provenance"
    )


@pytest.mark.asyncio
async def test_missing_source_object_is_quarantined(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    record = _paper()
    record.pop("source")

    await pipeline._process_candidate(
        "papers",
        record,
    )

    assert pipeline.output["papers"] == []
    assert pipeline.stats["missing_provenance"] == 1
    assert (
        pipeline.quarantine[0]["reason"]
        == "missing_provenance"
    )


@pytest.mark.asyncio
async def test_duplicate_record_is_not_emitted_twice(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    record = _paper()

    await pipeline._process_candidate(
        "papers",
        record,
    )

    await pipeline._process_candidate(
        "papers",
        record,
    )

    assert len(pipeline.output["papers"]) == 1
    assert pipeline.stats["accepted"] == 1
    assert pipeline.stats["duplicates"] == 1
    assert pipeline.stats["quarantined"] == 0


@pytest.mark.asyncio
async def test_schema_failure_is_quarantined(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    record = _paper()
    record["content"]["title"] = None

    await pipeline._process_candidate(
        "papers",
        record,
    )

    assert pipeline.output["papers"] == []
    assert pipeline.stats["invalid"] == 1
    assert pipeline.stats["quarantined"] == 1
    assert (
        pipeline.quarantine[0]["reason"]
        == "schema_validation_failed"
    )


@pytest.mark.asyncio
async def test_schema_failure_does_not_poison_identity(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

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

    assert pipeline.dedupe.first_seen(identity) is None


@pytest.mark.asyncio
async def test_entity_resolution_changes_canonical_field(
    tmp_path: Path,
):
    pipeline = _pipeline(
        tmp_path,
        resolver=FakeResolver(
            canonical="Canonical AI Corp"
        ),
    )

    record = _job()

    await pipeline._process_candidate(
        "jobs",
        record,
    )

    assert len(pipeline.output["jobs"]) == 1
    assert (
        pipeline.output["jobs"][0]["content"]["company"]
        == "Canonical AI Corp"
    )


@pytest.mark.asyncio
async def test_entity_resolution_failure_quarantines_record(
    tmp_path: Path,
):
    pipeline = _pipeline(
        tmp_path,
        resolver=FailingResolver(),
    )

    await pipeline._process_candidate(
        "jobs",
        _job(),
    )

    assert pipeline.output["jobs"] == []
    assert pipeline.stats["quarantined"] == 1
    assert (
        pipeline.quarantine[0]["reason"]
        == "entity_resolution_error"
    )


@pytest.mark.asyncio
async def test_entity_resolution_failure_releases_claim(
    tmp_path: Path,
):
    pipeline = _pipeline(
        tmp_path,
        resolver=FailingResolver(),
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

    assert pipeline.dedupe.first_seen(identity) is None


@pytest.mark.asyncio
async def test_released_record_can_be_reprocessed_successfully(
    tmp_path: Path,
):
    failing_pipeline = _pipeline(
        tmp_path,
        resolver=FailingResolver(),
    )

    record = _job()

    await failing_pipeline._process_candidate(
        "jobs",
        record,
    )

    successful_pipeline = Pipeline(
        dedupe=failing_pipeline.dedupe,
        resolver=FakeResolver(),
        use_llm=False,
    )

    await successful_pipeline._process_candidate(
        "jobs",
        record,
    )

    assert len(successful_pipeline.output["jobs"]) == 1
    assert successful_pipeline.stats["accepted"] == 1


@pytest.mark.asyncio
async def test_startup_record_reaches_schema_validation(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    await pipeline._process_candidate(
        "startups",
        _startup(),
    )

    assert len(pipeline.output["startups"]) == 1

    output = pipeline.output["startups"][0]

    assert output["recordType"] == "STARTUP"
    assert output["content"]["entityName"] == "Example AI"
    assert output["content"]["data"]["employeeCount"] == 12


@pytest.mark.asyncio
async def test_news_without_published_date_uses_new_record_heuristic(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    await pipeline._process_candidate(
        "news",
        _news(),
    )

    assert len(pipeline.output["news"]) == 1
    assert pipeline.stats["accepted"] == 1


@pytest.mark.asyncio
async def test_job_without_date_uses_new_record_heuristic(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    await pipeline._process_candidate(
        "jobs",
        _job(),
    )

    assert len(pipeline.output["jobs"]) == 1
    assert pipeline.stats["accepted"] == 1


@pytest.mark.asyncio
async def test_multiple_record_types_are_processed_independently(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    await pipeline._process_candidate(
        "papers",
        _paper(),
    )

    await pipeline._process_candidate(
        "startups",
        _startup(),
    )

    await pipeline._process_candidate(
        "jobs",
        _job(),
    )

    await pipeline._process_candidate(
        "news",
        _news(),
    )

    assert len(pipeline.output["papers"]) == 1
    assert len(pipeline.output["startups"]) == 1
    assert len(pipeline.output["jobs"]) == 1
    assert len(pipeline.output["news"]) == 1
    assert pipeline.stats["accepted"] == 4
    assert pipeline.stats["quarantined"] == 0


@pytest.mark.asyncio
async def test_unsupported_record_type_is_quarantined(
    tmp_path: Path,
):
    pipeline = _pipeline(tmp_path)

    await pipeline._consume(
        _records(
            {
                "anything": "invalid",
            }
        ),
        normalizer=_unsupported_normalizer,
    )

    assert pipeline.output == {
        "startups": [],
        "products": [],
        "papers": [],
        "jobs": [],
        "news": [],
    }

    assert pipeline.stats["quarantined"] == 1
    assert (
        pipeline.quarantine[0]["reason"]
        == "unsupported_record_type"
    )


async def _records(record: dict):
    yield record


async def _unsupported_normalizer(
    raw: dict,
):
    return [
        (
            "unsupported",
            raw,
        )
    ]
