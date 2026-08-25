"""Extraction prompts. Strict, anti-hallucination system instruction.

The system prompt hard-forbids invention: if a field is not present in the
supplied text, the model MUST return null. This is our first line of defense
against the "hallucinated data => disqualification" rule; the second is Pydantic
validation; the third is that `source.url` is injected by us, never by the LLM.
"""
from __future__ import annotations

SYSTEM = (
    "You are a meticulous data-extraction engine for an intelligence graph. "
    "Extract ONLY facts explicitly present in the supplied text. "
    "If a field is not present, return null — NEVER guess, infer, or fabricate. "
    "Do not add commentary. Respond with a single JSON object matching the "
    "requested schema exactly."
)

SCHEMAS: dict[str, str] = {
    "STARTUP": (
        '{"entityName": string, "employeeCount": integer|null}'
    ),
    "PRODUCT": (
        '{"startupName": string, "pricingModel": '
        '"FREE"|"FREEMIUM"|"PAID"|"ENTERPRISE"|null}'
    ),
    "RESEARCH_PAPER": (
        '{"title": string, "authors": string[], "github_url": string|null, '
        '"published_date": "ISO-8601"|null}'
    ),
    "JOB": (
        '{"company": string, "date": "ISO-8601"|null, "is_remote": boolean, '
        '"role_family": string|null, "title": string|null}'
    ),
    "NEWS": (
        '{"title": string, "published_date": "ISO-8601"|null}'
    ),
}


def build_user_prompt(record_type: str, text: str) -> str:
    schema = SCHEMAS[record_type]
    return (
        f"Extract a {record_type} record as JSON matching this schema:\n{schema}\n\n"
        f"--- SOURCE TEXT START ---\n{text}\n--- SOURCE TEXT END ---"
    )
