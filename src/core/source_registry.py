"""Configuration-backed source registry.

The YAML source registry is the canonical configuration for external sources.

The registry supports both:
    * structured SourceDefinition objects
    * dictionary-style access for backward compatibility

Source validation is source-type aware:
    * API/RSS sources require an endpoint.
    * Algolia sources require an index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.logging import get_logger

log = get_logger(__name__)


class SourceRegistryError(ValueError):
    """Raised when the source registry is missing or malformed."""


@dataclass(frozen=True)
class SourceDefinition:
    """Normalized definition for one configured source."""

    name: str
    source_type: str
    endpoint: str | None = None
    categories: tuple[str, ...] = ()
    index: str | None = None
    note: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """Return the configured source type."""
        return self.source_type

    def __getitem__(self, key: str) -> Any:
        """Provide backward-compatible dictionary-style access."""
        values = {
            "name": self.name,
            "type": self.source_type,
            "endpoint": self.endpoint,
            "categories": list(self.categories),
            "index": self.index,
            "note": self.note,
            **self.options,
        }

        try:
            return values[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-compatible get()."""
        values = {
            "name": self.name,
            "type": self.source_type,
            "endpoint": self.endpoint,
            "categories": list(self.categories),
            "index": self.index,
            "note": self.note,
            **self.options,
        }

        return values.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the source definition as a regular dictionary."""
        return {
            "name": self.name,
            "type": self.source_type,
            "endpoint": self.endpoint,
            "categories": list(self.categories),
            "index": self.index,
            "note": self.note,
            **self.options,
        }


class SourceRegistry:
    """Load, validate and query sources.yaml."""

    VALID_GROUPS = {
        "research_papers",
        "startups_products",
        "news",
        "jobs",
    }

    # All sources must have these fields.
    BASE_REQUIRED_FIELDS = {
        "name",
        "type",
    }

    def __init__(
        self,
        path: str | Path = "config/sources.yaml",
    ) -> None:
        self.path = Path(path)
        self._data = self._load()
        self._sources = self._build_sources()

    def _load(self) -> dict[str, Any]:
        """Load and validate the YAML root."""
        if not self.path.exists():
            raise SourceRegistryError(
                f"source registry not found: {self.path}"
            )

        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise SourceRegistryError(
                f"Invalid YAML in source registry {self.path}: {exc}"
            ) from exc

        if data is None:
            raise SourceRegistryError(
                f"source registry is empty: {self.path}"
            )

        if not isinstance(data, dict):
            raise SourceRegistryError(
                "source registry root must be a mapping"
            )

        for key in data:
            if key == "freshness_window_hours":
                continue

            if key not in self.VALID_GROUPS:
                raise SourceRegistryError(
                    f"unknown source category: {key}"
                )

        return data

    @staticmethod
    def _required_fields_for(
        source_type: str,
    ) -> set[str]:
        """Return required fields based on the source transport."""
        normalized = source_type.strip().lower()

        if normalized == "algolia":
            return {"name", "type", "index"}

        if normalized in {"api", "rss"}:
            return {"name", "type", "endpoint"}

        # Unknown/custom transports still require the common identity fields.
        # Their additional configuration can live in `options`.
        return {"name", "type"}

    def _build_sources(
        self,
    ) -> dict[str, list[SourceDefinition]]:
        """Validate and normalize all source groups."""
        result: dict[str, list[SourceDefinition]] = {}

        for group in self.VALID_GROUPS:
            raw_sources = self._data.get(group, [])

            if not isinstance(raw_sources, list):
                raise SourceRegistryError(
                    f"source category '{group}' must be a list"
                )

            definitions: list[SourceDefinition] = []

            for position, raw in enumerate(raw_sources):
                if not isinstance(raw, dict):
                    raise SourceRegistryError(
                        f"source category '{group}' entry {position} "
                        "must be a mapping"
                    )

                missing_base = (
                    self.BASE_REQUIRED_FIELDS - set(raw)
                )

                if missing_base:
                    fields = ", ".join(sorted(missing_base))
                    raise SourceRegistryError(
                        f"source category '{group}' entry {position} "
                        f"missing required fields: {fields}"
                    )

                name = raw.get("name")
                source_type = raw.get("type")

                if not isinstance(name, str) or not name.strip():
                    raise SourceRegistryError(
                        f"source category '{group}' entry {position} "
                        "field 'name' must be a non-empty string"
                    )

                if (
                    not isinstance(source_type, str)
                    or not source_type.strip()
                ):
                    raise SourceRegistryError(
                        f"source category '{group}' entry {position} "
                        "field 'type' must be a non-empty string"
                    )

                normalized_type = source_type.strip().lower()

                required_fields = self._required_fields_for(
                    normalized_type
                )

                missing = required_fields - set(raw)

                if missing:
                    fields = ", ".join(sorted(missing))
                    raise SourceRegistryError(
                        f"source category '{group}' entry {position} "
                        f"missing required fields: {fields}"
                    )

                endpoint = raw.get("endpoint")

                if endpoint is not None:
                    if (
                        not isinstance(endpoint, str)
                        or not endpoint.strip()
                    ):
                        raise SourceRegistryError(
                            f"source category '{group}' entry {position} "
                            "field 'endpoint' must be a non-empty string"
                        )

                    endpoint = endpoint.strip()

                index_value = raw.get("index")

                if index_value is not None:
                    if (
                        not isinstance(index_value, str)
                        or not index_value.strip()
                    ):
                        raise SourceRegistryError(
                            f"source category '{group}' entry {position} "
                            "field 'index' must be a non-empty string"
                        )

                    index_value = index_value.strip()

                categories_raw = raw.get("categories", [])

                if not isinstance(categories_raw, list):
                    raise SourceRegistryError(
                        f"source category '{group}' entry {position} "
                        "field 'categories' must be a list"
                    )

                categories: list[str] = []

                for category in categories_raw:
                    if not isinstance(category, str):
                        raise SourceRegistryError(
                            f"source category '{group}' entry {position} "
                            "contains a non-string category"
                        )

                    categories.append(category)

                note = raw.get("note")

                if note is not None and not isinstance(note, str):
                    raise SourceRegistryError(
                        f"source category '{group}' entry {position} "
                        "field 'note' must be a string"
                    )

                options = {
                    key: value
                    for key, value in raw.items()
                    if key
                    not in {
                        "name",
                        "type",
                        "endpoint",
                        "categories",
                        "index",
                        "note",
                    }
                }

                definitions.append(
                    SourceDefinition(
                        name=name.strip(),
                        source_type=normalized_type,
                        endpoint=endpoint,
                        categories=tuple(categories),
                        index=index_value,
                        note=note,
                        options=options,
                    )
                )

            result[group] = definitions

        return result

    def sources(
        self,
        category: str,
    ) -> tuple[SourceDefinition, ...]:
        """Return all configured sources in a category."""
        if category not in self.VALID_GROUPS:
            raise SourceRegistryError(
                f"unknown source category: {category}"
            )

        return tuple(self._sources.get(category, []))

    def get_all(
        self,
        category: str,
    ) -> tuple[SourceDefinition, ...]:
        """Alias for sources()."""
        return self.sources(category)

    def get(
        self,
        category: str,
        name: str,
    ) -> SourceDefinition:
        """Return one source by category and name."""
        for source in self.sources(category):
            if source.name.casefold() == name.casefold():
                return source

        raise SourceRegistryError(
            f"source '{name}' not found in category '{category}'"
        )

    def find(
        self,
        category: str,
        name: str,
    ) -> SourceDefinition | None:
        """Return a source or None when it does not exist."""
        try:
            return self.get(category, name)
        except SourceRegistryError:
            return None

    def endpoint(
        self,
        category: str,
        name: str,
    ) -> str:
        """Return the configured endpoint for a source."""
        source = self.get(category, name)

        if not source.endpoint:
            raise SourceRegistryError(
                f"source '{name}' in category '{category}' "
                "has no endpoint"
            )

        return source.endpoint

    def require_endpoint(
        self,
        category: str,
        name: str,
    ) -> SourceDefinition:
        """Return a source and guarantee that it has an endpoint."""
        source = self.get(category, name)

        if not source.endpoint:
            raise SourceRegistryError(
                f"source '{name}' in category '{category}' "
                "has no endpoint"
            )

        return source

    def freshness_window_hours(self) -> int:
        """Return the configured freshness window."""
        value = self._data.get(
            "freshness_window_hours",
            24,
        )

        if not isinstance(value, int) or isinstance(value, bool):
            raise SourceRegistryError(
                "freshness_window_hours must be a positive integer"
            )

        if value <= 0:
            raise SourceRegistryError(
                "freshness_window_hours must be a positive integer"
            )

        return value

    def __len__(self) -> int:
        """Return the total number of configured sources."""
        return sum(
            len(source_group)
            for source_group in self._sources.values()
        )


_default_registry: SourceRegistry | None = None
_default_registry_path: Path | None = None


def get_source_registry(
    path: str | Path = "config/sources.yaml",
) -> SourceRegistry:
    """Return the cached default registry or a fresh custom registry."""
    global _default_registry
    global _default_registry_path

    normalized = Path(path)

    if normalized == Path("config/sources.yaml"):
        if (
            _default_registry is None
            or _default_registry_path != normalized
        ):
            _default_registry = SourceRegistry(normalized)
            _default_registry_path = normalized

        return _default_registry

    return SourceRegistry(normalized)
