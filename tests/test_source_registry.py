"""Tests for the configuration-backed source registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.source_registry import (
    SourceRegistry,
    SourceRegistryError,
    get_source_registry,
)


def test_default_registry_loads():
    registry = get_source_registry()

    assert registry.sources("research_papers")
    assert registry.sources("startups_products")
    assert registry.sources("news")
    assert registry.sources("jobs")


def test_default_registry_contains_expected_research_source():
    registry = get_source_registry()

    arxiv = registry.get(
        "research_papers",
        "arXiv",
    )

    assert arxiv["type"] == "api"
    assert arxiv["endpoint"]


def test_source_lookup_is_case_insensitive():
    registry = get_source_registry()

    source = registry.get(
        "research_papers",
        "ARXIV",
    )

    assert source["name"] == "arXiv"


def test_endpoint_lookup():
    registry = get_source_registry()

    endpoint = registry.endpoint(
        "jobs",
        "Remotive",
    )

    assert endpoint == "https://remotive.com/api/remote-jobs"


def test_unknown_category_is_rejected():
    registry = get_source_registry()

    with pytest.raises(SourceRegistryError):
        registry.sources("unknown")


def test_unknown_source_is_rejected():
    registry = get_source_registry()

    with pytest.raises(SourceRegistryError):
        registry.get(
            "jobs",
            "DefinitelyNotARealSource",
        )


def test_missing_registry_file_is_rejected(tmp_path: Path):
    with pytest.raises(SourceRegistryError, match="not found"):
        SourceRegistry(
            tmp_path / "missing.yaml"
        )


def test_invalid_yaml_is_rejected(tmp_path: Path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "this: [is: invalid",
        encoding="utf-8",
    )

    with pytest.raises(SourceRegistryError, match="Invalid YAML"):
        SourceRegistry(path)


def test_invalid_source_shape_is_rejected(tmp_path: Path):
    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "jobs": [
                    {
                        "name": "Broken",
                        "type": "api",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceRegistryError,
        match="missing required fields",
    ):
        SourceRegistry(path)


def test_non_list_category_is_rejected(tmp_path: Path):
    path = tmp_path / "sources.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "jobs": {
                    "name": "Broken"
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceRegistryError,
        match="must be a list",
    ):
        SourceRegistry(path)