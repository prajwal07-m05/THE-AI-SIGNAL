"""Date-normalization + freshness tests (Phase II)."""
from datetime import datetime, timedelta, timezone

from src.freshness.date_parser import heuristic_is_new, is_fresh, parse_date

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


def test_relative_dates():
    assert parse_date("2 hours ago", now=NOW) == NOW - timedelta(hours=2)
    assert parse_date("3d", now=NOW) == NOW - timedelta(days=3)
    assert parse_date("yesterday", now=NOW) == NOW - timedelta(days=1)
    assert parse_date("just now", now=NOW) == NOW


def test_iso_and_epoch():
    assert parse_date("2026-08-25T10:00:00Z").astimezone(timezone.utc).hour == 10
    assert parse_date(1_756_120_800).tzinfo is not None  # epoch seconds


def test_freshness_window():
    assert is_fresh(NOW - timedelta(hours=5), window_hours=24, now=NOW)
    assert not is_fresh(NOW - timedelta(hours=30), window_hours=24, now=NOW)


def test_heuristic_when_no_date():
    # Never seen before => treat as new.
    assert heuristic_is_new(None, seen_before=False, window_hours=24)
    # Seen before, no date => stale.
    assert not heuristic_is_new(None, seen_before=True, window_hours=24)
