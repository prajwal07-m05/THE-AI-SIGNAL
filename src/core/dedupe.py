"""Distributed dedupe + freshness ledger (Phase VI, point 3).

Guarantees we never process the same article/job/entity twice — even across
distributed crawler nodes. Redis is the shared source of truth when available
(atomic `SET key val NX` => a node "claims" a URL); otherwise we fall back to a
local SQLite ledger so the pipeline still runs on a single laptop.

Keys are content-addressed: sha1 of a normalized identity string (usually the
canonical URL). We store `first_seen` so freshness heuristics can ask
"have we seen this before, and if so when?".
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from src.core.logging import get_logger
from src.settings import get_settings

log = get_logger(__name__)


def _fingerprint(identity: str) -> str:
    return hashlib.sha1(identity.strip().lower().encode()).hexdigest()


class _SQLiteLedger:
    def __init__(self, path: str = "./out/ledger.sqlite") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS seen "
            "(fp TEXT PRIMARY KEY, first_seen REAL NOT NULL)"
        )
        self._db.commit()

    def claim(self, identity: str) -> bool:
        fp = _fingerprint(identity)
        try:
            self._db.execute(
                "INSERT INTO seen (fp, first_seen) VALUES (?, ?)", (fp, time.time())
            )
            self._db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def first_seen(self, identity: str) -> float | None:
        row = self._db.execute(
            "SELECT first_seen FROM seen WHERE fp = ?", (_fingerprint(identity),)
        ).fetchone()
        return row[0] if row else None


class _RedisLedger:
    def __init__(self, url: str) -> None:
        import redis  # local import so redis is optional

        self._r = redis.Redis.from_url(url)
        self._r.ping()

    def claim(self, identity: str) -> bool:
        fp = _fingerprint(identity)
        # Atomic across all nodes: only one caller wins the SET NX.
        won = self._r.set(f"seen:{fp}", time.time(), nx=True)
        return bool(won)

    def first_seen(self, identity: str) -> float | None:
        val = self._r.get(f"seen:{_fingerprint(identity)}")
        return float(val) if val else None


class DedupeLedger:
    """Facade that picks Redis when configured, else SQLite."""

    def __init__(self) -> None:
        url = get_settings().redis_url
        self._impl: _RedisLedger | _SQLiteLedger
        if url:
            try:
                self._impl = _RedisLedger(url)
                log.info("dedupe_backend", backend="redis")
                return
            except Exception as e:  # noqa: BLE001
                log.warning("redis_unavailable_fallback_sqlite", error=str(e))
        self._impl = _SQLiteLedger()
        log.info("dedupe_backend", backend="sqlite")

    def is_new(self, identity: str) -> bool:
        """True if THIS node just claimed `identity` for the first time."""
        return self._impl.claim(identity)

    def first_seen(self, identity: str) -> float | None:
        return self._impl.first_seen(identity)
