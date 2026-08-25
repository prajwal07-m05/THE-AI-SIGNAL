"""Distributed deduplication and freshness ledger.

Redis is used as the shared source of truth when configured. SQLite is used
as a reliable local fallback for development and single-node deployments.

The ledger stores a normalized identity fingerprint and its first-seen
timestamp. Redis uses atomic SET NX semantics so multiple crawler workers can
safely compete for the same identity.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Protocol

from src.core.logging import get_logger
from src.settings import get_settings

log = get_logger(__name__)


def _fingerprint(identity: str) -> str:
    """Return a stable fingerprint for a normalized identity."""
    return hashlib.sha1(
        identity.strip().lower().encode("utf-8")
    ).hexdigest()


class _LedgerBackend(Protocol):
    """Interface implemented by local and distributed ledgers."""

    def claim(self, identity: str) -> bool:
        """Atomically claim an identity."""

    def first_seen(self, identity: str) -> float | None:
        """Return the first-seen timestamp for an identity."""


class _SQLiteLedger:
    """Local persistent deduplication ledger."""

    def __init__(
        self,
        path: str = "./out/ledger.sqlite",
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._db = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )

        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
                fp TEXT PRIMARY KEY,
                first_seen REAL NOT NULL
            )
            """
        )

        self._db.commit()

    def claim(
        self,
        identity: str,
    ) -> bool:
        """Claim an identity exactly once."""
        fp = _fingerprint(identity)

        try:
            self._db.execute(
                """
                INSERT INTO seen (fp, first_seen)
                VALUES (?, ?)
                """,
                (
                    fp,
                    time.time(),
                ),
            )

            self._db.commit()

            return True

        except sqlite3.IntegrityError:
            return False

    def first_seen(
        self,
        identity: str,
    ) -> float | None:
        """Return the original claim timestamp."""
        row = self._db.execute(
            """
            SELECT first_seen
            FROM seen
            WHERE fp = ?
            """,
            (
                _fingerprint(identity),
            ),
        ).fetchone()

        return row[0] if row else None

    def close(self) -> None:
        """Close the SQLite connection."""
        self._db.close()


class _RedisLedger:
    """Distributed Redis-backed deduplication ledger."""

    def __init__(
        self,
        url: str,
    ) -> None:
        import redis

        self._r = redis.Redis.from_url(
            url
        )

        self._r.ping()

    def claim(
        self,
        identity: str,
    ) -> bool:
        """Atomically claim an identity using SET NX."""
        fp = _fingerprint(identity)

        won = self._r.set(
            f"seen:{fp}",
            time.time(),
            nx=True,
        )

        return bool(won)

    def first_seen(
        self,
        identity: str,
    ) -> float | None:
        """Return the first-seen timestamp from Redis."""
        value = self._r.get(
            f"seen:{_fingerprint(identity)}"
        )

        return float(value) if value else None

    def close(self) -> None:
        """Close the Redis client."""
        self._r.close()


class DedupeLedger:
    """Facade selecting Redis or SQLite as the deduplication backend.

    Parameters
    ----------
    sqlite_path:
        Optional local SQLite database path.

        When omitted, the production/development default remains:
        ``./out/ledger.sqlite``.
    """

    def __init__(
        self,
        *,
        sqlite_path: str = "./out/ledger.sqlite",
    ) -> None:
        settings = get_settings()

        self._impl: _LedgerBackend

        redis_url = settings.redis_url

        if redis_url:
            try:
                self._impl = _RedisLedger(
                    redis_url
                )

                log.info(
                    "dedupe_backend",
                    backend="redis",
                )

                return

            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "redis_unavailable_fallback_sqlite",
                    error=str(exc),
                )

        self._impl = _SQLiteLedger(
            sqlite_path
        )

        log.info(
            "dedupe_backend",
            backend="sqlite",
            path=sqlite_path,
        )

    def is_new(
        self,
        identity: str,
    ) -> bool:
        """Return True only when this caller claims identity first."""
        return self._impl.claim(
            identity
        )

    def first_seen(
        self,
        identity: str,
    ) -> float | None:
        """Return when the identity was first claimed."""
        return self._impl.first_seen(
            identity
        )

    def close(self) -> None:
        """Close the underlying ledger when supported."""
        close = getattr(
            self._impl,
            "close",
            None,
        )

        if close is not None:
            close()