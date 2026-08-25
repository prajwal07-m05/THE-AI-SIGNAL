"""Canonical schema contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    JobRecord,
    NewsRecord,
    ProductRecord,
    ResearchPaperRecord,
    Source,
    StartupRecord,
)


def _source() -> dict[str, str]:
    return {
        "name": "Example Source",
        "url": "https://example.com/source",
    }


def _paper() -> dict:
    return {
        "schemaVersion": "1.0",
        "recordType": "RESEARCH_PAPER",
        "source": _source(),
        "collectedAt": "2026-08-25T12:00:00Z",
        "content": {
            "title": "Reliable AI Systems",
            "authors": ["Alice Example"],
            "paper_url": "https://arxiv.org/abs/1234.5678",
            "github_url": None,
            "github_stars": 10,
            "published_date": "2026-08-25T10:00:00Z",
        },
    }


def test_source_requires_valid_url():
    with pytest.raises(ValidationError):
        Source(
            name="Example",
            url="not-a-url",
        )


def test_source_rejects_blank_name():
    with pytest.raises(ValidationError):
        Source(
            name="   ",
            url="https://example.com",
        )


def test_schema_version_is_fixed():
    record = _paper()
    record["schemaVersion"] = "2.0"

    with pytest.raises(ValidationError):
        ResearchPaperRecord.model_validate(record)


def test_collected_at_must_be_timezone_aware():
    record = _paper()
    record["collectedAt"] = datetime(
        2026,
        8,
        25,
        12,
        0,
    )

    with pytest.raises(ValidationError):
        ResearchPaperRecord.model_validate(record)


def test_paper_rejects_blank_title():
    record = _paper()
    record["content"]["title"] = "   "

    with pytest.raises(ValidationError):
        ResearchPaperRecord.model_validate(record)


def test_paper_rejects_blank_author():
    record = _paper()
    record["content"]["authors"] = [
        "Alice Example",
        "   ",
    ]

    with pytest.raises(ValidationError):
        ResearchPaperRecord.model_validate(record)


def test_paper_rejects_negative_github_stars():
    record = _paper()
    record["content"]["github_stars"] = -1

    with pytest.raises(ValidationError):
        ResearchPaperRecord.model_validate(record)


def test_startup_rejects_blank_entity_name():
    record = {
        "schemaVersion": "1.0",
        "recordType": "STARTUP",
        "source": _source(),
        "content": {
            "entityName": "   ",
            "data": {
                "employeeCount": 10,
            },
        },
    }

    with pytest.raises(ValidationError):
        StartupRecord.model_validate(record)


def test_startup_rejects_negative_employee_count():
    record = {
        "schemaVersion": "1.0",
        "recordType": "STARTUP",
        "source": _source(),
        "content": {
            "entityName": "Example AI",
            "data": {
                "employeeCount": -1,
            },
        },
    }

    with pytest.raises(ValidationError):
        StartupRecord.model_validate(record)


def test_product_rejects_blank_startup_name():
    record = {
        "schemaVersion": "1.0",
        "recordType": "PRODUCT",
        "source": _source(),
        "content": {
            "startupName": "   ",
            "pricingModel": "FREE",
        },
    }

    with pytest.raises(ValidationError):
        ProductRecord.model_validate(record)


def test_job_rejects_blank_company():
    record = {
        "schemaVersion": "1.0",
        "recordType": "JOB",
        "source": _source(),
        "content": {
            "company": "   ",
            "date": None,
            "is_remote": True,
            "role_family": "AI",
            "title": "AI Engineer",
        },
    }

    with pytest.raises(ValidationError):
        JobRecord.model_validate(record)


def test_job_normalizes_optional_blank_strings():
    record = {
        "schemaVersion": "1.0",
        "recordType": "JOB",
        "source": _source(),
        "content": {
            "company": "Example AI",
            "date": None,
            "is_remote": True,
            "role_family": "   ",
            "title": "   ",
        },
    }

    validated = JobRecord.model_validate(record)

    assert validated.content.role_family is None
    assert validated.content.title is None


def test_news_rejects_blank_title():
    record = {
        "schemaVersion": "1.0",
        "recordType": "NEWS",
        "source": _source(),
        "content": {
            "title": "   ",
            "published_date": None,
            "full_text": "Article body",
        },
    }

    with pytest.raises(ValidationError):
        NewsRecord.model_validate(record)


def test_valid_record_round_trips():
    validated = ResearchPaperRecord.model_validate(_paper())

    assert validated.recordType == "RESEARCH_PAPER"
    assert validated.schemaVersion == "1.0"
    assert validated.source.name == "Example Source"
    assert validated.content.title == "Reliable AI Systems"
    assert validated.collectedAt.tzinfo is not None


def test_extra_top_level_fields_are_rejected():
    record = _paper()
    record["unexpected"] = "should not be accepted"

    with pytest.raises(ValidationError):
        ResearchPaperRecord.model_validate(record)


def test_extra_content_fields_are_rejected():
    record = _paper()
    record["content"]["hallucinated_field"] = "bad data"

    with pytest.raises(ValidationError):
        ResearchPaperRecord.model_validate(record)
