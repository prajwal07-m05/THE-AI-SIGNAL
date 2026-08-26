"""Extraction prompts for the LLM enrichment layer.

The prompt layer accepts both pipeline record types (``papers``) and
canonical LLM record types (``RESEARCH_PAPER``). This keeps the pipeline,
orchestrator, and provider tests compatible without weakening validation.

The system prompt strictly forbids hallucination: if a field is not
explicitly present in the supplied text, the model must return null rather
than guessing or inferring.
"""

from __future__ import annotations


SYSTEM = (
    "You are a meticulous data-extraction engine for an intelligence graph. "
    "Extract ONLY facts explicitly present in the supplied text. "
    "If a field is not present, return null — NEVER guess, infer, or fabricate. "
    "Do not add commentary. Respond with a single JSON object matching the "
    "requested schema exactly."
)


# Canonical schema names used by the LLM layer.
SCHEMAS: dict[str, str] = {
    "STARTUP": (
        '{"entityName": string, "employeeCount": integer|null}'
    ),
    "PRODUCT": (
        '{"startupName": string, "pricingModel": '
        '"FREE"|"FREEMIUM"|"PAID"|"ENTERPRISE"|null}'
    ),
    "RESEARCH_PAPER": (
        '{"title": string, "authors": string[], '
        '"github_url": string|null, '
        '"published_date": "ISO-8601"|null}'
    ),
    "JOB": (
        '{"company": string, "date": "ISO-8601"|null, '
        '"is_remote": boolean, '
        '"role_family": string|null, '
        '"title": string|null}'
    ),
    "NEWS": (
        '{"title": string, "published_date": "ISO-8601"|null}'
    ),
}


# The ingestion pipeline uses lowercase plural names, while the LLM
# orchestrator/tests use canonical schema names. Support both explicitly.
RECORD_TYPE_MAP: dict[str, str] = {
    # Pipeline record types.
    "startups": "STARTUP",
    "products": "PRODUCT",
    "papers": "RESEARCH_PAPER",
    "jobs": "JOB",
    "news": "NEWS",

    # Canonical LLM record types.
    "STARTUP": "STARTUP",
    "PRODUCT": "PRODUCT",
    "RESEARCH_PAPER": "RESEARCH_PAPER",
    "JOB": "JOB",
    "NEWS": "NEWS",
}


def build_user_prompt(
    record_type: str,
    text: str,
) -> str:
    """Build a strict extraction prompt for a pipeline record type.

    Both pipeline names such as ``papers`` and canonical LLM names such as
    ``RESEARCH_PAPER`` are accepted. Unsupported record types fail clearly
    before a provider request is made.
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

    return (
        f"Extract a {schema_type} record as JSON matching this schema:\n"
        f"{schema}\n\n"
        f"--- SOURCE TEXT START ---\n"
        f"{text}\n"
        f"--- SOURCE TEXT END ---"
    )