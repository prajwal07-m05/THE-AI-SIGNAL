"""Startup/product pipeline integration coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.dedupe import DedupeLedger
from src.pipeline.runner import Pipeline


def _startup_raw() -> dict:
    return {
        "record_kind": "STARTUP",
        "source_name": "Y Combinator",
        "source_url": "https://www.ycombinator.com/companies/example",
        "name": "Example AI",
        "team_size": 12,
        "one_liner": "AI infrastructure for developers",
        "long_description": "Example AI builds developer infrastructure.",
        "batch": "W24",
    }


def _product_raw() -> dict:
    return {
        "record_kind": "PRODUCT",
        "source_name": "Y Combinator",
        "source_url": "https://www.ycombinator.com/companies/example",
        "startup_name": "Example AI",
        "product_desc": "AI infrastructure for developers",
    }


@pytest.mark.asyncio
async def test_startup_normalization_produces_canonical_record(
    tmp_path: Path,
):
    pipeline = Pipeline(
        dedupe=DedupeLedger(
            sqlite_path=str(tmp_path / "ledger.sqlite")
        ),
        use_llm=False,
    )

    normalized = await pipeline._normalize_startup_or_product(
        _startup_raw()
    )

    assert len(normalized) == 1

    record_type, record = normalized[0]

    assert record_type == "startups"
    assert record["schemaVersion"] == "1.0"
    assert record["source"]["name"] == "Y Combinator"
    assert (
        str(record["source"]["url"])
        == "https://www.ycombinator.com/companies/example"
    )
    assert record["content"]["entityName"] == "Example AI"
    assert record["content"]["data"]["employeeCount"] == 12


@pytest.mark.asyncio
async def test_product_normalization_produces_canonical_record(
    tmp_path: Path,
):
    pipeline = Pipeline(
        dedupe=DedupeLedger(
            sqlite_path=str(tmp_path / "ledger.sqlite")
        ),
        use_llm=False,
    )

    normalized = await pipeline._normalize_startup_or_product(
        _product_raw()
    )

    assert len(normalized) == 1

    record_type, record = normalized[0]

    assert record_type == "products"
    assert record["schemaVersion"] == "1.0"
    assert record["source"]["name"] == "Y Combinator"
    assert (
        str(record["source"]["url"])
        == "https://www.ycombinator.com/companies/example"
    )
    assert record["content"]["startupName"] == "Example AI"


@pytest.mark.asyncio
async def test_startup_without_employee_count_is_valid(
    tmp_path: Path,
):
    raw = _startup_raw()
    raw["team_size"] = None

    pipeline = Pipeline(
        dedupe=DedupeLedger(
            sqlite_path=str(tmp_path / "ledger.sqlite")
        ),
        use_llm=False,
    )

    normalized = await pipeline._normalize_startup_or_product(raw)

    assert len(normalized) == 1

    _, record = normalized[0]

    assert record["content"]["data"]["employeeCount"] is None


@pytest.mark.asyncio
async def test_product_pricing_model_is_normalized(
    tmp_path: Path,
):
    raw = _product_raw()
    raw["pricing_model"] = "freemium"

    pipeline = Pipeline(
        dedupe=DedupeLedger(
            sqlite_path=str(tmp_path / "ledger.sqlite")
        ),
        use_llm=False,
    )

    normalized = await pipeline._normalize_startup_or_product(raw)

    assert len(normalized) == 1

    record_type, record = normalized[0]

    assert record_type == "products"
    assert record["content"]["pricingModel"] == "FREEMIUM"


@pytest.mark.asyncio
async def test_invalid_startup_provenance_is_rejected(
    tmp_path: Path,
):
    raw = _startup_raw()
    raw["source_url"] = ""

    pipeline = Pipeline(
        dedupe=DedupeLedger(
            sqlite_path=str(tmp_path / "ledger.sqlite")
        ),
        use_llm=False,
    )

    normalized = await pipeline._normalize_startup_or_product(raw)

    assert normalized == []


@pytest.mark.asyncio
async def test_invalid_product_provenance_is_rejected(
    tmp_path: Path,
):
    raw = _product_raw()
    raw["source_name"] = ""

    pipeline = Pipeline(
        dedupe=DedupeLedger(
            sqlite_path=str(tmp_path / "ledger.sqlite")
        ),
        use_llm=False,
    )

    normalized = await pipeline._normalize_startup_or_product(raw)

    assert normalized == []
