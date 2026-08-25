"""Persistent deduplication and freshness ledger.

The ledger supports two backends:

* Redis for shared, distributed deployments.
* SQLite for local development and single-node deployments.

A record is first *claimed* before expensive downstream processing. Claims can
be released when processing fails, preventing transient LLM/network/validation
failures from permanently poisoning the deduplication ledger.

Successful claims remain committed and prevent duplicate processing on later
runs.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

from src.core.logging import get_logger
from src.settings import get_settings

log = get_logger(__name__)


def _fingerprint(identity: str) -> str:
    """Return a stable fingerprint for a normalized identity."""
    return hashlib.sha1(identity.strip().lower().encode("utf-8")).hexdigest()


class _SQLiteLedger:
    """Thread-safe persistent SQLite implementation."""

    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(
            path,
            check_same_thread=False,
        )
        self._lock = threading.RLock()

        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
                fp TEXT PRIMARY KEY,
                first_seen REAL NOT NULL
            )
            """
        )
        self._db.commit()

    def claim(self, identity: str) -> bool:
        """Atomically claim an identity.

        Returns True only for the caller that successfully inserted it.
        """
        fp = _fingerprint(identity)

        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO seen (fp, first_seen) VALUES (?, ?)",
                    (fp, time.time()),
                )
                self._db.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def release(self, identity: str) -> bool:
        """Remove a previously claimed identity.

        Returns True when a ledger entry was removed.
        """
        fp = _fingerprint(identity)

        with self._lock:
            cursor = self._db.execute(
                "DELETE FROM seen WHERE fp = ?",
                (fp,),
            )
            self._db.commit()
            return cursor.rowcount > 0

    def first_seen(self, identity: str) -> float | None:
        fp = _fingerprint(identity)

        with self._lock:
            row = self._db.execute(
                "SELECT first_seen FROM seen WHERE fp = ?",
                (fp,),
            ).fetchone()

        return float(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._db.close()


class _RedisLedger:
    """Redis implementation using atomic SET NX claims."""

    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url)
        self._r.ping()

    def claim(self, identity: str) -> bool:
        """Atomically claim an identity across distributed workers."""
        fp = _fingerprint(identity)

        won = self._r.set(
            f"seen:{fp}",
            time.time(),
            nx=True,
        )

        return bool(won)

    def release(self, identity: str) -> bool:
        """Release a previously claimed identity."""
        fp = _fingerprint(identity)

        return bool(
            self._r.delete(f"seen:{fp}")
        )

    def first_seen(self, identity: str) -> float | None:
        fp = _fingerprint(identity)

        value = self._r.get(f"seen:{fp}")

        if value is None:
            return None

        return float(value)

    def close(self) -> None:
        close = getattr(self._r, "close", None)

        if close is not None:
            close()


class DedupeLedger:
    """Backend-independent deduplication facade."""

    def __init__(
        self,
        *,
        sqlite_path: str = "./out/ledger.sqlite",
    ) -> None:
        settings = get_settings()

        self._impl: _RedisLedger | _SQLiteLedger

        if settings.redis_url:
            try:
                self._impl = _RedisLedger(settings.redis_url)

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

        self._impl = _SQLiteLedger(sqlite_path)

        log.info(
            "dedupe_backend",
            backend="sqlite",
            path=sqlite_path,
        )

    def is_new(self, identity: str) -> bool:
        """Claim an identity.

        Returns True when this caller successfully claims it for processing.
        """
        return self._impl.claim(identity)

    def release(self, identity: str) -> bool:
        """Release a claim after downstream processing fails."""
        return self._impl.release(identity)

    def first_seen(self, identity: str) -> float | None:
        """Return the first-seen timestamp, if present."""
        return self._impl.first_seen(identity)

    def close(self) -> None:
        """Close the underlying backend."""
        self._impl.close()
