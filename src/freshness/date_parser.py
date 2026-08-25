"""Date normalization + freshness heuristics (Phase II).

Handles the messy reality of publication dates:
  * ISO-8601 / RFC-822 meta tags               -> dateutil
  * Relative strings ("2 hours ago", "yesterday", "3d") -> custom parser
  * Missing dates                              -> intelligent heuristic that
    treats "never seen this URL before" as a freshness signal (via the ledger).

Everything is normalized to timezone-aware UTC datetimes so the 24-hour window
check is unambiguous across sources and crawler nodes.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from dateutil import parser as dtparse

from src.core.logging import get_logger

log = get_logger(__name__)

_REL_RE = re.compile(
    r"(?P<num>\d+)\s*(?P<unit>second|sec|minute|min|hour|hr|day|week|month|year)s?\s*ago",
    re.IGNORECASE,
)
_SHORT_RE = re.compile(r"^\s*(?P<num>\d+)\s*(?P<unit>s|m|h|d|w)\s*$", re.IGNORECASE)
_UNIT_SECONDS = {
    "second": 1, "sec": 1, "s": 1,
    "minute": 60, "min": 60, "m": 60,
    "hour": 3600, "hr": 3600, "h": 3600,
    "day": 86400, "d": 86400,
    "week": 604800, "w": 604800,
    "month": 2_592_000, "year": 31_536_000,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(
    raw: str | int | float | None, *, now: datetime | None = None
) -> datetime | None:
    """Best-effort parse of any human/machine date value into UTC.

    Accepts ISO/RFC strings, relative strings, and epoch seconds (int/float).
    Returns None only when the value carries no temporal meaning at all.
    """
    if raw is None or raw == "":
        return None
    now = now or _now()

    # Epoch seconds (e.g. Arbeitnow `created_at`, some job boards)
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(raw).strip()

    low = text.lower()
    if low in {"just now", "now", "moments ago"}:
        return now
    if low in {"yesterday"}:
        return now - timedelta(days=1)
    if low in {"today"}:
        return now

    if (m := _REL_RE.search(text)) or (m := _SHORT_RE.match(text)):
        secs = int(m.group("num")) * _UNIT_SECONDS[m.group("unit").lower()]
        return now - timedelta(seconds=secs)

    try:
        dt = dtparse.parse(text, fuzzy=True)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, TypeError):
        log.debug("unparseable_date", raw=raw)
        return None


def is_fresh(dt: datetime | None, *, window_hours: int, now: datetime | None = None) -> bool:
    """True if `dt` falls within the freshness window (default 24h)."""
    if dt is None:
        return False
    now = now or _now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt) <= timedelta(hours=window_hours) and dt <= now + timedelta(minutes=5)


def heuristic_is_new(
    published: datetime | None,
    *,
    seen_before: bool,
    window_hours: int,
) -> bool:
    """Fallback freshness decision when a strict date is absent.

    Rule: if we have a real date, trust the window. If we DON'T, treat a URL we
    have never seen in the dedupe ledger as "new since last run" — the standard
    incremental-crawl heuristic. Anything already in the ledger is stale.
    """
    if published is not None:
        return is_fresh(published, window_hours=window_hours)
    return not seen_before
