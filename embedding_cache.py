"""Persistent, revision-safe cache for document embedding vectors."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = 1


def embedding_cache_key(embedding_fingerprint: str, prompted_text: str) -> str:
    digest = hashlib.sha256()
    digest.update(embedding_fingerprint.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompted_text.encode("utf-8"))
    return digest.hexdigest()


class EmbeddingCache:
    """Store float32 vectors keyed by the exact model prompt and revision."""

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_fingerprint "
            "ON embedding_cache(fingerprint)"
        )
        self.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _decode(dimension: int, payload: bytes) -> list[float] | None:
        if dimension <= 0 or len(payload) != dimension * 4:
            return None
        try:
            return list(struct.unpack(f"<{dimension}f", payload))
        except struct.error:
            return None

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        rows = self.connection.execute(
            f"SELECT cache_key, dimension, vector FROM embedding_cache "
            f"WHERE fingerprint = ? AND cache_key IN ({placeholders})",
            [self.fingerprint, *keys],
        ).fetchall()
        values: dict[str, list[float]] = {}
        corrupt: list[tuple[str]] = []
        for cache_key, dimension, payload in rows:
            vector = self._decode(int(dimension), bytes(payload))
            if vector is None:
                corrupt.append((str(cache_key),))
            else:
                values[str(cache_key)] = vector
        if corrupt:
            self.connection.executemany(
                "DELETE FROM embedding_cache WHERE cache_key = ?",
                corrupt,
            )
            self.connection.commit()
        return values

    def put_many(self, values: Iterable[tuple[str, Sequence[float]]]) -> None:
        timestamp = datetime.now(tz=UTC).isoformat()
        rows = []
        for cache_key, vector in values:
            floats = [float(value) for value in vector]
            if not floats:
                continue
            rows.append(
                (
                    cache_key,
                    self.fingerprint,
                    len(floats),
                    sqlite3.Binary(struct.pack(f"<{len(floats)}f", *floats)),
                    timestamp,
                )
            )
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT INTO embedding_cache (
                cache_key, fingerprint, dimension, vector, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                fingerprint = excluded.fingerprint,
                dimension = excluded.dimension,
                vector = excluded.vector,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self.connection.commit()

    def prune_other_fingerprints(self) -> int:
        cursor = self.connection.execute(
            "DELETE FROM embedding_cache WHERE fingerprint <> ?",
            (self.fingerprint,),
        )
        self.connection.commit()
        return max(0, int(cursor.rowcount))
