"""Configuration-backed source registry.

The source registry is the single source of truth for external ingestion
sources. Scrapers consume source definitions from config/sources.yaml rather
than duplicating configuration throughout the codebase.

Every source must provide:
    - name
    - type
    - at least one source-specific configuration field

Source-specific fields are intentionally flexible so different source types
can define endpoints, indexes, categories, notes, and other configuration
without forcing unrelated sources into one rigid schema.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class SourceRegistryError(RuntimeError):
    """Raised when the source registry cannot be loaded or is invalid."""


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "sources.yaml"

_REQUIRED_SOURCE_FIELDS = {"name", "type"}

_VALID_CATEGORIES = {
    "research_papers",
    "startups_products",
    "news",
    "jobs",
}


class SourceRegistry:
    """Read and validate ingestion sources from YAML configuration."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _DEFAULT_CONFIG_PATH
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise SourceRegistryError(
                f"Source registry not found: {self.path}"
            )

        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise SourceRegistryError(
                f"Invalid YAML in source registry: {self.path}"
            ) from exc

        if not isinstance(data, dict):
            raise SourceRegistryError(
                "Source registry root must be a mapping."
            )

        self._validate(data)
        return data

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        for category in _VALID_CATEGORIES:
            sources = data.get(category, [])

            if not isinstance(sources, list):
                raise SourceRegistryError(
                    f"Source category '{category}' must be a list."
                )

            for index, source in enumerate(sources):
                if not isinstance(source, dict):
                    raise SourceRegistryError(
                        f"{category}[{index}] must be a mapping."
                    )

                missing = _REQUIRED_SOURCE_FIELDS - set(source)

                if missing:
                    missing_fields = ", ".join(sorted(missing))
                    raise SourceRegistryError(
                        f"{category}[{index}] is missing required fields: "
                        f"{missing_fields}"
                    )

                for field in _REQUIRED_SOURCE_FIELDS:
                    value = source.get(field)

                    if not isinstance(value, str) or not value.strip():
                        raise SourceRegistryError(
                            f"{category}[{index}].{field} must be a "
                            "non-empty string."
                        )

                # name/type alone is not a usable source definition.
                # Require at least one source-specific configuration field.
                source_specific_fields = set(source) - _REQUIRED_SOURCE_FIELDS

                if not source_specific_fields:
                    raise SourceRegistryError(
                        f"{category}[{index}] is missing required fields: "
                        "source-specific configuration"
                    )

    def sources(self, category: str) -> list[dict[str, Any]]:
        """Return copies of configured sources for a category."""
        if category not in _VALID_CATEGORIES:
            raise SourceRegistryError(
                f"Unknown source category: {category!r}"
            )

        return [
            dict(source)
            for source in self._data.get(category, [])
        ]

    def get(self, category: str, name: str) -> dict[str, Any]:
        """Return one source by category and name."""
        for source in self.sources(category):
            if source["name"].casefold() == name.casefold():
                return source

        raise SourceRegistryError(
            f"Source '{name}' not found in category '{category}'."
        )

    def endpoint(self, category: str, name: str) -> str:
        """Return a configured endpoint.

        This helper is intentionally strict because not every source type has
        an HTTP endpoint. For example, an Algolia source may use an index
        configuration instead.
        """
        source = self.get(category, name)

        endpoint = source.get("endpoint")

        if not isinstance(endpoint, str) or not endpoint.strip():
            raise SourceRegistryError(
                f"Source '{name}' in category '{category}' has no endpoint."
            )

        return endpoint


@lru_cache(maxsize=1)
def get_source_registry() -> SourceRegistry:
    """Return the process-wide source registry."""
    return SourceRegistry()
