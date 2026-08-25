"""Persistent deduplication ledger tests."""

from __future__ import annotations

from pathlib import Path

from src.core.dedupe import DedupeLedger, _fingerprint


def test_fingerprint_normalizes_identity():
    assert _fingerprint(
        " https://example.com/article "
    ) == _fingerprint(
        "https://EXAMPLE.com/article"
    )


def test_identity_is_claimed_only_once(tmp_path: Path):
    ledger = DedupeLedger(
        sqlite_path=str(tmp_path / "ledger.sqlite")
    )

    identity = "https://example.com/article"

    assert ledger.is_new(identity) is True
    assert ledger.is_new(identity) is False

    ledger.close()


def test_first_seen_is_recorded(tmp_path: Path):
    ledger = DedupeLedger(
        sqlite_path=str(tmp_path / "ledger.sqlite")
    )

    identity = "https://example.com/paper"

    assert ledger.first_seen(identity) is None

    assert ledger.is_new(identity) is True

    first_seen = ledger.first_seen(identity)

    assert first_seen is not None
    assert isinstance(first_seen, float)

    ledger.close()


def test_deduplication_persists_across_instances(tmp_path: Path):
    ledger_path = str(
        tmp_path / "persistent.sqlite"
    )

    first = DedupeLedger(
        sqlite_path=ledger_path
    )

    identity = "https://example.com/persistent"

    assert first.is_new(identity) is True

    first.close()

    second = DedupeLedger(
        sqlite_path=ledger_path
    )

    assert second.is_new(identity) is False

    second.close()


def test_different_identities_are_independent(tmp_path: Path):
    ledger = DedupeLedger(
        sqlite_path=str(tmp_path / "ledger.sqlite")
    )

    assert ledger.is_new(
        "https://example.com/a"
    ) is True

    assert ledger.is_new(
        "https://example.com/b"
    ) is True

    assert ledger.is_new(
        "https://example.com/a"
    ) is False

    ledger.close()


def test_release_allows_failed_record_to_be_reprocessed(
    tmp_path: Path,
):
    ledger = DedupeLedger(
        sqlite_path=str(tmp_path / "ledger.sqlite")
    )

    identity = "https://example.com/retryable"

    assert ledger.is_new(identity) is True

    assert ledger.release(identity) is True

    assert ledger.first_seen(identity) is None

    assert ledger.is_new(identity) is True

    ledger.close()


def test_release_unknown_identity_is_safe(tmp_path: Path):
    ledger = DedupeLedger(
        sqlite_path=str(tmp_path / "ledger.sqlite")
    )

    assert ledger.release(
        "https://example.com/never-claimed"
    ) is False

    ledger.close()
