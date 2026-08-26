"""Extraction prompts for the LLM enrichment layer.

The prompt layer accepts both pipeline record types (``papers``) and
canonical LLM record types (``RESEARCH_PAPER``).

The LLM is used only for structured extraction/enrichment. Source-owned
fields such as URLs and article bodies must never be fabricated or replaced
by the model.
"""

from __future__ import annotations


SYSTEM = (
    "You are a meticulous data-extraction engine for an intelligence graph. "
    "Your task is structured extraction from supplied source text. "
    "Use ONLY information explicitly present in the source text. "
    "NEVER guess, infer, hallucinate, fabricate, or use outside knowledge. "
    "If the evidence for a field is absent or ambiguous, return null. "
    "Preserve exact names, titles, URLs, dates, and author names when present. "
    "Do not invent missing values. "
    "Do not return fields outside the requested schema. "
    "Do not add commentary, explanations, markdown, or code fences. "
    "Return exactly one JSON object matching the requested schema."
)


# Canonical schemas used by the LLM layer.
#
# These schemas intentionally describe only fields that the LLM is allowed
# to enrich. Source-owned fields such as source.url and news.full_text are
# preserved by the pipeline rather than reconstructed by the model.
SCHEMAS: dict[str, str] = {
    "STARTUP": (
        "{"
        '"entityName": string|null, '
        '"employeeCount": integer|null'
        "}"
    ),
    "PRODUCT": (
        "{"
        '"startupName": string|null, '
        '"pricingModel": '
        '"FREE"|"FREEMIUM"|"PAID"|"ENTERPRISE"|null'
        "}"
    ),
    "RESEARCH_PAPER": (
        "{"
        '"title": string|null, '
        '"authors": string[], '
        '"github_url": string|null, '
        '"published_date": "ISO-8601"|null'
        "}"
    ),
    "JOB": (
        "{"
        '"company": string|null, '
        '"date": "ISO-8601"|null, '
        '"is_remote": boolean|null, '
        '"role_family": string|null, '
        '"title": string|null'
        "}"
    ),
    "NEWS": (
        "{"
        '"title": string|null, '
        '"published_date": "ISO-8601"|null'
        "}"
    ),
}


# Pipeline record names and canonical LLM schema names.
RECORD_TYPE_MAP: dict[str, str] = {
    "startups": "STARTUP",
    "products": "PRODUCT",
    "papers": "RESEARCH_PAPER",
    "jobs": "JOB",
    "news": "NEWS",
    "STARTUP": "STARTUP",
    "PRODUCT": "PRODUCT",
    "RESEARCH_PAPER": "RESEARCH_PAPER",
    "JOB": "JOB",
    "NEWS": "NEWS",
}


def _product_instructions() -> str:
    return (
        "\nPRODUCT EXTRACTION RULES:\n"
        "- pricingModel may be FREE, FREEMIUM, PAID, ENTERPRISE, or null.\n"
        "- FREE means the supplied text explicitly describes the product as "
        "free with no paid tier mentioned.\n"
        "- FREEMIUM means the supplied text explicitly describes a free tier "
        "alongside paid or premium functionality.\n"
        "- PAID means the supplied text explicitly describes the product as "
        "paid, subscription-based, or requiring payment, without evidence "
        "that it is primarily an enterprise-only offering.\n"
        "- ENTERPRISE means the supplied text explicitly describes "
        "enterprise-only or enterprise-targeted pricing/access.\n"
        "- If the pricing information is absent or ambiguous, return null.\n"
        "- Do not classify pricing from assumptions about the product.\n"
    )


def _job_instructions() -> str:
    return (
        "\nJOB EXTRACTION RULES:\n"
        "- title must come directly from the supplied source text.\n"
        "- company must come directly from the supplied source text.\n"
        "- date must only be returned when an explicit date is present.\n"
        "- is_remote may be true only when the supplied text explicitly "
        "indicates remote work; otherwise return false only when the source "
        "explicitly indicates the role is not remote. If neither is stated, "
        "return null.\n"
        "- role_family must be supported by the supplied job title or text.\n"
        "- Do not invent a specialization that is not stated.\n"
        "- If role_family cannot be determined from explicit evidence, "
        "return null.\n"
    )


def _paper_instructions() -> str:
    return (
        "\nRESEARCH PAPER EXTRACTION RULES:\n"
        "- title must be the exact paper title present in the source.\n"
        "- authors must contain only authors explicitly listed in the source.\n"
        "- github_url must be returned only when an explicit GitHub URL is "
        "present in the supplied text.\n"
        "- published_date must only be returned when an explicit publication "
        "date is present.\n"
        "- Never construct a GitHub URL from a repository name or infer one "
        "from the paper title.\n"
    )


def _startup_instructions() -> str:
    return (
        "\nSTARTUP EXTRACTION RULES:\n"
        "- entityName must come directly from the supplied source text.\n"
        "- employeeCount must only be returned when an explicit employee or "
        "team count is present.\n"
        "- Do not estimate employee count from company size, funding, age, "
        "or any other indirect signal.\n"
    )


def _news_instructions() -> str:
    return (
        "\nNEWS EXTRACTION RULES:\n"
        "- title must come directly from the supplied source text.\n"
        "- published_date must only be returned when an explicit publication "
        "date is present.\n"
        "- Do not summarize or recreate article body text.\n"
        "- The pipeline preserves source article text separately.\n"
    )


def _schema_instructions(schema_type: str) -> str:
    if schema_type == "STARTUP":
        return _startup_instructions()

    if schema_type == "PRODUCT":
        return _product_instructions()

    if schema_type == "RESEARCH_PAPER":
        return _paper_instructions()

    if schema_type == "JOB":
        return _job_instructions()

    if schema_type == "NEWS":
        return _news_instructions()

    return ""


def build_user_prompt(
    record_type: str,
    text: str,
) -> str:
    """Build a strict extraction prompt for a pipeline record type.

    Both lowercase pipeline names such as ``papers`` and canonical names
    such as ``RESEARCH_PAPER`` are accepted.
    """
    try:
        schema_type = RECORD_TYPE_MAP[record_type]
    except KeyError as exc:
        supported = ", ".join(
            sorted(RECORD_TYPE_MAP)
        )

        raise ValueError(
            f"Unsupported LLM record type: {record_type!r}. "
            f"Expected one of: {supported}"
        ) from exc

    schema = SCHEMAS[schema_type]
    instructions = _schema_instructions(schema_type)

    return (
        f"Extract a {schema_type} record as JSON matching this schema:\n"
        f"{schema}\n"
        f"{instructions}\n"
        "IMPORTANT:\n"
        "- Use only the supplied source text.\n"
        "- Missing or ambiguous information MUST be null.\n"
        "- Do not add extra JSON fields.\n"
        "- Return exactly one JSON object.\n\n"
        "--- SOURCE TEXT START ---\n"
        f"{text}\n"
        "--- SOURCE TEXT END ---"
    )