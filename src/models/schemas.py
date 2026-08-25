"""Canonical record schemas and validation contracts.

This module is the single source of truth for the records emitted by the
pipeline. Every record carries explicit provenance, a fixed schema version,
a record type, and a timezone-aware collection timestamp.

The models intentionally reject structurally unsafe values before records
reach the output layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def _require_non_blank(value: str, field_name: str) -> str:
    """Normalize a required string and reject blank values."""
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    return value


class Source(BaseModel):
    """Provenance information for a record."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable source name",
    )
    url: HttpUrl = Field(
        ...,
        description="Original source URL",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _require_non_blank(value, "source.name")


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"


class _Envelope(BaseModel):
    """Common record envelope."""

    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source: Source
    collectedAt: datetime = Field(default_factory=_utcnow)

    @field_validator("collectedAt")
    @classmethod
    def validate_collected_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("collectedAt must be timezone-aware")
        return value


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
class StartupData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employeeCount: int | None = Field(
        default=None,
        ge=0,
    )


class StartupContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entityName: str = Field(
        ...,
        min_length=1,
        description="Canonical startup name",
    )
    data: StartupData = Field(default_factory=StartupData)

    @field_validator("entityName")
    @classmethod
    def validate_entity_name(cls, value: str) -> str:
        return _require_non_blank(value, "entityName")


class StartupRecord(_Envelope):
    recordType: Literal["STARTUP"] = "STARTUP"
    content: StartupContent


# --------------------------------------------------------------------------- #
# Product
# --------------------------------------------------------------------------- #
class ProductContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    startupName: str = Field(
        ...,
        min_length=1,
        description="Canonical startup name",
    )
    pricingModel: PricingModel | None = None

    @field_validator("startupName")
    @classmethod
    def validate_startup_name(cls, value: str) -> str:
        return _require_non_blank(value, "startupName")


class ProductRecord(_Envelope):
    recordType: Literal["PRODUCT"] = "PRODUCT"
    content: ProductContent


# --------------------------------------------------------------------------- #
# Research paper
# --------------------------------------------------------------------------- #
class ResearchPaperContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=1,
    )
    authors: list[str] = Field(default_factory=list)
    paper_url: HttpUrl
    github_url: HttpUrl | None = None
    github_stars: int | None = Field(
        default=None,
        ge=0,
    )
    published_date: datetime | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _require_non_blank(value, "title")

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []

        for author in value:
            author = author.strip()

            if not author:
                raise ValueError("authors cannot contain blank values")

            cleaned.append(author)

        return cleaned


class ResearchPaperRecord(_Envelope):
    recordType: Literal["RESEARCH_PAPER"] = "RESEARCH_PAPER"
    content: ResearchPaperContent


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #
class JobContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str = Field(
        ...,
        min_length=1,
        description="Canonical company name",
    )
    date: datetime | None = None
    is_remote: bool = False
    role_family: str | None = None
    title: str | None = None

    @field_validator("company")
    @classmethod
    def validate_company(cls, value: str) -> str:
        return _require_non_blank(value, "company")

    @field_validator("title", "role_family")
    @classmethod
    def validate_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class JobRecord(_Envelope):
    recordType: Literal["JOB"] = "JOB"
    content: JobContent


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
class NewsContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=1,
    )
    published_date: datetime | None = None
    full_text: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _require_non_blank(value, "title")


class NewsRecord(_Envelope):
    recordType: Literal["NEWS"] = "NEWS"
    content: NewsContent


AnyRecord = (
    StartupRecord
    | ProductRecord
    | ResearchPaperRecord
    | JobRecord
    | NewsRecord
)
