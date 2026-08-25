"""Canonical JSON schemas — the single source of truth for record shape.

These mirror the "Expected Schemas" section of the assignment exactly, wrapped
in a common envelope (schemaVersion / recordType / source / content / collectedAt).
Every record produced by the pipeline is validated against these models before
it is written anywhere. A record that fails validation is quarantined, never
emitted — this is a hard guard against the "hallucinated data => disqualification"
rule: every record MUST carry a real `source.url`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Source(BaseModel):
    name: str = Field(..., description="Name of the source site")
    url: HttpUrl = Field(..., description="Original source URL — MUST be real & valid")


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"


class _Envelope(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    source: Source
    collectedAt: datetime = Field(default_factory=_utcnow)

    @field_validator("collectedAt")
    @classmethod
    def _ensure_iso(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
class StartupData(BaseModel):
    employeeCount: int | None = None


class StartupContent(BaseModel):
    entityName: str = Field(..., description="Canonical startup name")
    data: StartupData = Field(default_factory=StartupData)


class StartupRecord(_Envelope):
    recordType: Literal["STARTUP"] = "STARTUP"
    content: StartupContent


# --------------------------------------------------------------------------- #
# Product
# --------------------------------------------------------------------------- #
class ProductContent(BaseModel):
    startupName: str = Field(..., description="Canonical startup name")
    pricingModel: PricingModel | None = None


class ProductRecord(_Envelope):
    recordType: Literal["PRODUCT"] = "PRODUCT"
    content: ProductContent


# --------------------------------------------------------------------------- #
# Research paper
# --------------------------------------------------------------------------- #
class ResearchPaperContent(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    paper_url: HttpUrl
    github_url: HttpUrl | None = None
    github_stars: int | None = None
    published_date: datetime | None = None


class ResearchPaperRecord(_Envelope):
    recordType: Literal["RESEARCH_PAPER"] = "RESEARCH_PAPER"
    content: ResearchPaperContent


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #
class JobContent(BaseModel):
    company: str = Field(..., description="Canonical company name")
    date: datetime | None = None
    is_remote: bool = False
    role_family: str | None = None
    title: str | None = None


class JobRecord(_Envelope):
    recordType: Literal["JOB"] = "JOB"
    content: JobContent


# --------------------------------------------------------------------------- #
# News (Phase II signal)
# --------------------------------------------------------------------------- #
class NewsContent(BaseModel):
    title: str
    published_date: datetime | None = None
    full_text: str | None = None


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
