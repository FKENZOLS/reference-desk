"""Persistent local research workspace for saved passages and search history."""

from __future__ import annotations

import io
import hashlib
import json
import math
import os
import sqlite3
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_WORKSPACE_DB = Path(
    os.environ.get(
        "RAG_WORKSPACE_DB",
        Path(__file__).resolve().with_name("reference_workspace.sqlite3"),
    )
)

FEEDBACK_JUDGMENTS = {
    "relevant",
    "wrong_passage",
    "wrong_document",
    "no_relevant_result",
    "expected_passage",
    "ambiguous",
}

WORKSPACE_SCHEMA_VERSION = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkspaceStore:
    """Small SQLite store that remains independent from the vector database."""

    def __init__(self, path: str | Path = DEFAULT_WORKSPACE_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_before_migration()
        self._initialize()

    def _backup_before_migration(self) -> None:
        """Make one recoverable SQLite snapshot before changing an old schema."""

        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        source = sqlite3.connect(self.path, timeout=30)
        try:
            version = int(source.execute("PRAGMA user_version").fetchone()[0])
            if version >= WORKSPACE_SCHEMA_VERSION:
                return
            backup = self.path.with_name(
                f"{self.path.name}.pre-v{WORKSPACE_SCHEMA_VERSION}-migration.bak"
            )
            if backup.exists():
                return
            destination = sqlite3.connect(backup)
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            current_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if current_version > WORKSPACE_SCHEMA_VERSION:
                raise RuntimeError(
                    "This workspace database was created by a newer Reference Desk "
                    f"schema (v{current_version}); this build supports v"
                    f"{WORKSPACE_SCHEMA_VERSION}."
                )
            connection.executescript(
                """
                BEGIN IMMEDIATE;

                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE,
                    source_id TEXT NOT NULL,
                    document_title TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    content_type TEXT NOT NULL DEFAULT '',
                    document_date TEXT NOT NULL DEFAULT '',
                    excerpt TEXT NOT NULL,
                    citation_label TEXT NOT NULL DEFAULT '',
                    citation_url TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL DEFAULT '',
                    collection_id INTEGER REFERENCES collections(id)
                        ON DELETE SET NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    source_filter TEXT NOT NULL DEFAULT '',
                    section_filter TEXT NOT NULL DEFAULT '',
                    content_filter TEXT NOT NULL DEFAULT '',
                    date_filter TEXT NOT NULL DEFAULT '',
                    within_results INTEGER NOT NULL DEFAULT 0,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    judgment TEXT NOT NULL CHECK(judgment IN (
                        'relevant', 'wrong_passage', 'wrong_document',
                        'no_relevant_result', 'expected_passage', 'ambiguous'
                    )),
                    chunk_id TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    document_title TEXT NOT NULL DEFAULT '',
                    page_start INTEGER,
                    page_end INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    excerpt TEXT NOT NULL DEFAULT '',
                    source_filter TEXT NOT NULL DEFAULT '',
                    section_filter TEXT NOT NULL DEFAULT '',
                    content_filter TEXT NOT NULL DEFAULT '',
                    date_filter TEXT NOT NULL DEFAULT '',
                    result_rank INTEGER,
                    rerank_logit REAL,
                    final_score REAL,
                    reranker_model TEXT NOT NULL DEFAULT '',
                    reranker_fingerprint TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    expected_source_id TEXT NOT NULL DEFAULT '',
                    expected_page INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(
                        query, source_filter, section_filter, content_filter,
                        date_filter, target_key, reranker_fingerprint
                    )
                );

                CREATE TABLE IF NOT EXISTS quality_calibration (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    threshold REAL,
                    positive_count INTEGER NOT NULL DEFAULT 0,
                    negative_count INTEGER NOT NULL DEFAULT 0,
                    positive_recall REAL,
                    specificity REAL,
                    balanced_accuracy REAL,
                    ready INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    reranker_model TEXT NOT NULL DEFAULT '',
                    reranker_fingerprint TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quality_calibration_v2 (
                    reranker_fingerprint TEXT PRIMARY KEY,
                    reranker_model TEXT NOT NULL DEFAULT '',
                    threshold REAL,
                    positive_count INTEGER NOT NULL DEFAULT 0,
                    negative_count INTEGER NOT NULL DEFAULT 0,
                    positive_recall REAL,
                    specificity REAL,
                    balanced_accuracy REAL,
                    ready INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quality_benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    version INTEGER NOT NULL DEFAULT 1,
                    content_hash TEXT NOT NULL,
                    case_count INTEGER NOT NULL,
                    content_jsonl TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quality_benchmark_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_id INTEGER NOT NULL REFERENCES quality_benchmarks(id)
                        ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    case_count INTEGER NOT NULL,
                    content_jsonl TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(benchmark_id, version)
                );

                CREATE TABLE IF NOT EXISTS quality_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    benchmark_id INTEGER REFERENCES quality_benchmarks(id)
                        ON DELETE SET NULL,
                    benchmark_name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'queued', 'running', 'complete', 'failed'
                    )),
                    config_json TEXT NOT NULL,
                    results_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    production INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL DEFAULT 'system',
                    title TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK(status IN (
                        'info', 'success', 'warning', 'error'
                    )),
                    task_key TEXT NOT NULL DEFAULT '',
                    read_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_bookmarks_collection
                    ON bookmarks(collection_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_history_created
                    ON search_history(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_updated
                    ON retrieval_feedback(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_judgment
                    ON retrieval_feedback(judgment, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_experiments_updated
                    ON quality_experiments(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_benchmark_versions
                    ON quality_benchmark_versions(benchmark_id, version DESC);
                CREATE INDEX IF NOT EXISTS idx_notifications_created
                    ON notifications(created_at DESC);
                """
            )
            self._ensure_column(
                connection,
                "retrieval_feedback",
                "reranker_model",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "retrieval_feedback",
                "reranker_fingerprint",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "retrieval_feedback",
                "reason",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "retrieval_feedback",
                "expected_source_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "retrieval_feedback",
                "expected_page",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "quality_benchmarks",
                "metadata_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(
                connection,
                "quality_calibration",
                "reranker_model",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "quality_calibration",
                "reranker_fingerprint",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._migrate_feedback_identity(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_reranker
                ON retrieval_feedback(reranker_fingerprint, updated_at DESC)
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO quality_calibration_v2(
                    reranker_fingerprint, reranker_model, threshold,
                    positive_count, negative_count, positive_recall,
                    specificity, balanced_accuracy, ready, enabled, updated_at
                )
                SELECT reranker_fingerprint, reranker_model, threshold,
                       positive_count, negative_count, positive_recall,
                       specificity, balanced_accuracy, ready, enabled, updated_at
                FROM quality_calibration WHERE id = 1
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO quality_benchmark_versions(
                    benchmark_id, version, content_hash, case_count,
                    content_jsonl, metadata_json, created_at
                )
                SELECT id, version, content_hash, case_count, content_jsonl,
                       metadata_json, updated_at
                FROM quality_benchmarks
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (WORKSPACE_SCHEMA_VERSION, utc_now()),
            )
            connection.execute(f"PRAGMA user_version = {WORKSPACE_SCHEMA_VERSION}")

    @staticmethod
    def _migrate_feedback_identity(connection: sqlite3.Connection) -> None:
        """Keep judgments for different rerankers as independent records."""

        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("retrieval_feedback",),
        ).fetchone()
        table_sql = "".join(str(row["sql"] or "").lower().split()) if row else ""
        expected_identity = (
            "unique(query,source_filter,section_filter,content_filter,"
            "date_filter,target_key,reranker_fingerprint)"
        )
        if expected_identity in table_sql and "'expected_passage'" in table_sql:
            return

        connection.execute(
            "ALTER TABLE retrieval_feedback RENAME TO retrieval_feedback_legacy"
        )
        connection.execute(
            """
            CREATE TABLE retrieval_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                target_key TEXT NOT NULL,
                judgment TEXT NOT NULL CHECK(judgment IN (
                    'relevant', 'wrong_passage', 'wrong_document',
                    'no_relevant_result', 'expected_passage', 'ambiguous'
                )),
                chunk_id TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL DEFAULT '',
                document_title TEXT NOT NULL DEFAULT '',
                page_start INTEGER,
                page_end INTEGER,
                section TEXT NOT NULL DEFAULT '',
                excerpt TEXT NOT NULL DEFAULT '',
                source_filter TEXT NOT NULL DEFAULT '',
                section_filter TEXT NOT NULL DEFAULT '',
                content_filter TEXT NOT NULL DEFAULT '',
                date_filter TEXT NOT NULL DEFAULT '',
                result_rank INTEGER,
                rerank_logit REAL,
                final_score REAL,
                reranker_model TEXT NOT NULL DEFAULT '',
                reranker_fingerprint TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                expected_source_id TEXT NOT NULL DEFAULT '',
                expected_page INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(
                    query, source_filter, section_filter, content_filter,
                    date_filter, target_key, reranker_fingerprint
                )
            )
            """
        )
        connection.execute(
            """
            INSERT INTO retrieval_feedback(
                id, query, target_key, judgment, chunk_id, source_id,
                document_title, page_start, page_end, section, excerpt,
                source_filter, section_filter, content_filter, date_filter,
                result_rank, rerank_logit, final_score, reranker_model,
                reranker_fingerprint, reason, expected_source_id,
                expected_page, created_at, updated_at
            )
            SELECT
                id, query, target_key, judgment, chunk_id, source_id,
                document_title, page_start, page_end, section, excerpt,
                source_filter, section_filter, content_filter, date_filter,
                result_rank, rerank_logit, final_score, reranker_model,
                reranker_fingerprint, reason, expected_source_id,
                expected_page, created_at, updated_at
            FROM retrieval_feedback_legacy
            """
        )
        connection.execute("DROP TABLE retrieval_feedback_legacy")
        connection.execute(
            "CREATE INDEX idx_feedback_updated ON retrieval_feedback(updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX idx_feedback_judgment "
            "ON retrieval_feedback(judgment, updated_at DESC)"
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    @staticmethod
    def _rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    def list_collections(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT collections.*, COUNT(bookmarks.id) AS bookmark_count
                FROM collections
                LEFT JOIN bookmarks ON bookmarks.collection_id = collections.id
                GROUP BY collections.id
                ORDER BY collections.name COLLATE NOCASE
                """
            ).fetchall()
        return self._rows(rows)

    def create_collection(
        self,
        name: str,
        description: str = "",
    ) -> dict[str, Any]:
        name = name.strip()[:120]
        if not name:
            raise ValueError("Collection name is required")
        now = utc_now()
        try:
            with self.connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO collections(name, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, description.strip()[:1000], now, now),
                )
                collection_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            raise ValueError("A collection with this name already exists") from error
        return self.get_collection(collection_id)

    def get_collection(self, collection_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM collections WHERE id = ?",
                (int(collection_id),),
            ).fetchone()
        if row is None:
            raise KeyError(collection_id)
        return dict(row)

    def get_bookmark(
        self,
        bookmark_id: int | None = None,
        chunk_id: str | None = None,
    ) -> dict[str, Any] | None:
        if bookmark_id is None and not chunk_id:
            return None
        clause, value = (
            ("bookmarks.id", int(bookmark_id))
            if bookmark_id is not None
            else ("bookmarks.chunk_id", str(chunk_id))
        )
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT bookmarks.*, collections.name AS collection_name
                FROM bookmarks
                LEFT JOIN collections ON collections.id = bookmarks.collection_id
                WHERE {clause} = ?
                """,
                (value,),
            ).fetchone()
        return dict(row) if row is not None else None

    def upsert_bookmark(self, payload: dict[str, Any]) -> dict[str, Any]:
        chunk_id = str(payload.get("chunk_id") or "").strip()[:200]
        source_id = str(payload.get("source_id") or "").strip()[:1000]
        excerpt = str(payload.get("excerpt") or "").strip()
        if not chunk_id or not source_id or not excerpt:
            raise ValueError("chunk_id, source_id, and excerpt are required")

        collection_id = payload.get("collection_id") or None
        if collection_id is not None:
            collection_id = int(collection_id)
            self.get_collection(collection_id)
        now = utc_now()
        values = (
            chunk_id,
            source_id,
            str(payload.get("document_title") or source_id).strip()[:500],
            _optional_positive_int(payload.get("page_start")),
            _optional_positive_int(payload.get("page_end")),
            str(payload.get("section") or "").strip()[:2000],
            str(payload.get("content_type") or "").strip()[:250],
            str(payload.get("document_date") or "").strip()[:100],
            excerpt,
            str(payload.get("citation_label") or "").strip()[:2000],
            str(payload.get("citation_url") or "").strip()[:4000],
            str(payload.get("query") or "").strip()[:2000],
            collection_id,
            str(payload.get("note") or "").strip(),
            now,
            now,
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO bookmarks(
                    chunk_id, source_id, document_title, page_start, page_end,
                    section, content_type, document_date, excerpt,
                    citation_label, citation_url, query, collection_id, note,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    document_title = excluded.document_title,
                    page_start = excluded.page_start,
                    page_end = excluded.page_end,
                    section = excluded.section,
                    content_type = excluded.content_type,
                    document_date = excluded.document_date,
                    excerpt = excluded.excerpt,
                    citation_label = excluded.citation_label,
                    citation_url = excluded.citation_url,
                    query = excluded.query,
                    collection_id = excluded.collection_id,
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                values,
            )
        bookmark = self.get_bookmark(chunk_id=chunk_id)
        assert bookmark is not None
        return bookmark

    def update_bookmark(
        self,
        bookmark_id: int,
        *,
        note: str | None = None,
        collection_id: int | None | object = ...,
    ) -> dict[str, Any]:
        existing = self.get_bookmark(bookmark_id=bookmark_id)
        if existing is None:
            raise KeyError(bookmark_id)
        assignments: list[str] = []
        values: list[Any] = []
        if note is not None:
            assignments.append("note = ?")
            values.append(note.strip())
        if collection_id is not ...:
            normalized_collection = int(collection_id) if collection_id else None
            if normalized_collection is not None:
                self.get_collection(normalized_collection)
            assignments.append("collection_id = ?")
            values.append(normalized_collection)
        if assignments:
            assignments.append("updated_at = ?")
            values.append(utc_now())
            values.append(int(bookmark_id))
            with self.connect() as connection:
                connection.execute(
                    f"UPDATE bookmarks SET {', '.join(assignments)} WHERE id = ?",
                    values,
                )
        updated = self.get_bookmark(bookmark_id=bookmark_id)
        assert updated is not None
        return updated

    def delete_bookmark(self, bookmark_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM bookmarks WHERE id = ?",
                (int(bookmark_id),),
            )
        return cursor.rowcount > 0

    def list_bookmarks(
        self,
        collection_id: int | None = None,
    ) -> list[dict[str, Any]]:
        where = "WHERE bookmarks.collection_id = ?" if collection_id else ""
        parameters: tuple[Any, ...] = (int(collection_id),) if collection_id else ()
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT bookmarks.*, collections.name AS collection_name
                FROM bookmarks
                LEFT JOIN collections ON collections.id = bookmarks.collection_id
                {where}
                ORDER BY bookmarks.updated_at DESC, bookmarks.id DESC
                """,
                parameters,
            ).fetchall()
        return self._rows(rows)

    def bookmarks_by_ids(self, bookmark_ids: Sequence[int]) -> list[dict[str, Any]]:
        ordered_ids = list(dict.fromkeys(int(item) for item in bookmark_ids))
        if not ordered_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT bookmarks.*, collections.name AS collection_name
                FROM bookmarks
                LEFT JOIN collections ON collections.id = bookmarks.collection_id
                WHERE bookmarks.id IN ({placeholders})
                """,
                ordered_ids,
            ).fetchall()
        by_id = {int(row["id"]): dict(row) for row in rows}
        return [by_id[item] for item in ordered_ids if item in by_id]

    def record_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or "").strip()[:2000]
        if not query:
            raise ValueError("Search query is required")
        values = (
            query,
            str(payload.get("source_filter") or "").strip()[:1000],
            str(payload.get("section_filter") or "").strip()[:1000],
            str(payload.get("content_filter") or "").strip()[:100],
            str(payload.get("date_filter") or "").strip()[:100],
            1 if payload.get("within_results") else 0,
            max(0, int(payload.get("result_count") or 0)),
            utc_now(),
        )
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO search_history(
                    query, source_filter, section_filter, content_filter,
                    date_filter, within_results, result_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            history_id = int(cursor.lastrowid)
            # Keep this utility database small without hiding recent work.
            connection.execute(
                """
                DELETE FROM search_history WHERE id NOT IN (
                    SELECT id FROM search_history ORDER BY id DESC LIMIT 500
                )
                """
            )
            row = connection.execute(
                "SELECT * FROM search_history WHERE id = ?",
                (history_id,),
            ).fetchone()
        assert row is not None
        return dict(row)

    def list_history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM search_history ORDER BY id DESC LIMIT ?",
                (min(max(int(limit), 1), 500),),
            ).fetchall()
        return self._rows(rows)

    def upsert_feedback(
        self,
        payload: dict[str, Any],
        *,
        min_positive: int = 20,
        min_negative: int = 20,
        min_recall: float = 0.90,
    ) -> dict[str, Any]:
        """Store one explicit retrieval judgment and refresh calibration."""

        query = str(payload.get("query") or "").strip()[:2000]
        judgment = str(payload.get("judgment") or "").strip()
        chunk_id = str(payload.get("chunk_id") or "").strip()[:200]
        if not query:
            raise ValueError("Search query is required")
        if judgment not in FEEDBACK_JUDGMENTS:
            raise ValueError("Unknown feedback judgment")
        if judgment in {"relevant", "wrong_passage", "wrong_document"} and not chunk_id:
            raise ValueError("A result chunk is required for this judgment")

        expected_source_id = str(
            payload.get("expected_source_id") or payload.get("source_id") or ""
        ).strip()[:1000]
        expected_page = _optional_positive_int(
            payload.get("expected_page") or payload.get("page_start")
        )
        if judgment == "expected_passage" and not (expected_source_id and expected_page):
            raise ValueError("An expected document and page are required")
        if chunk_id:
            target_key = chunk_id
        elif judgment == "expected_passage":
            target_key = f"__expected__:{expected_source_id}:{expected_page or ''}"
        elif judgment == "ambiguous":
            target_key = "__ambiguous__"
        else:
            target_key = "__no_relevant_result__"
        source_filter = str(payload.get("source_filter") or "").strip()[:1000]
        section_filter = str(payload.get("section_filter") or "").strip()[:1000]
        content_filter = str(payload.get("content_filter") or "").strip()[:100]
        date_filter = str(payload.get("date_filter") or "").strip()[:100]
        reranker_model = str(payload.get("reranker_model") or "").strip()[:500]
        reranker_fingerprint = str(
            payload.get("reranker_fingerprint") or ""
        ).strip()[:128]
        now = utc_now()
        values = (
            query,
            target_key,
            judgment,
            chunk_id,
            str(payload.get("source_id") or "").strip()[:1000],
            str(payload.get("document_title") or "").strip()[:500],
            _optional_positive_int(payload.get("page_start")),
            _optional_positive_int(payload.get("page_end")),
            str(payload.get("section") or "").strip()[:2000],
            str(payload.get("excerpt") or "").strip()[:20000],
            source_filter,
            section_filter,
            content_filter,
            date_filter,
            _optional_positive_int(payload.get("result_rank")),
            _optional_finite_float(payload.get("rerank_logit")),
            _optional_finite_float(payload.get("final_score")),
            reranker_model,
            reranker_fingerprint,
            str(payload.get("reason") or "").strip()[:4000],
            expected_source_id,
            expected_page,
            now,
            now,
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_feedback(
                    query, target_key, judgment, chunk_id, source_id,
                    document_title, page_start, page_end, section, excerpt,
                    source_filter, section_filter, content_filter, date_filter,
                    result_rank, rerank_logit, final_score, reranker_model,
                    reranker_fingerprint, reason, expected_source_id,
                    expected_page, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    query, source_filter, section_filter, content_filter,
                    date_filter, target_key, reranker_fingerprint
                ) DO UPDATE SET
                    judgment = excluded.judgment,
                    chunk_id = excluded.chunk_id,
                    source_id = excluded.source_id,
                    document_title = excluded.document_title,
                    page_start = excluded.page_start,
                    page_end = excluded.page_end,
                    section = excluded.section,
                    excerpt = excluded.excerpt,
                    result_rank = excluded.result_rank,
                    rerank_logit = excluded.rerank_logit,
                    final_score = excluded.final_score,
                    reranker_model = excluded.reranker_model,
                    reranker_fingerprint = excluded.reranker_fingerprint,
                    reason = excluded.reason,
                    expected_source_id = excluded.expected_source_id,
                    expected_page = excluded.expected_page,
                    updated_at = excluded.updated_at
                """,
                values,
            )
            row = connection.execute(
                """
                SELECT * FROM retrieval_feedback
                WHERE query = ? AND source_filter = ? AND section_filter = ?
                  AND content_filter = ? AND date_filter = ? AND target_key = ?
                  AND reranker_fingerprint = ?
                """,
                (
                    query,
                    source_filter,
                    section_filter,
                    content_filter,
                    date_filter,
                    target_key,
                    reranker_fingerprint,
                ),
            ).fetchone()
        self.calibrate_feedback(
            min_positive=min_positive,
            min_negative=min_negative,
            min_recall=min_recall,
            reranker_model=reranker_model,
            reranker_fingerprint=reranker_fingerprint,
        )
        assert row is not None
        return dict(row)

    def list_feedback(
        self,
        limit: int = 200,
        judgment: str | None = None,
        reranker_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        conditions: list[str] = []
        if judgment:
            if judgment not in FEEDBACK_JUDGMENTS:
                raise ValueError("Unknown feedback judgment")
            conditions.append("judgment = ?")
            parameters.append(judgment)
        if reranker_fingerprint is not None:
            conditions.append("reranker_fingerprint = ?")
            parameters.append(str(reranker_fingerprint))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(min(max(int(limit), 1), 2000))
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM retrieval_feedback
                {where}
                ORDER BY updated_at DESC, id DESC LIMIT ?
                """,
                parameters,
            ).fetchall()
        return self._rows(rows)

    def delete_feedback(
        self,
        feedback_id: int,
        *,
        min_positive: int = 20,
        min_negative: int = 20,
        min_recall: float = 0.90,
    ) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT reranker_model, reranker_fingerprint
                FROM retrieval_feedback WHERE id = ?
                """,
                (int(feedback_id),),
            ).fetchone()
            cursor = connection.execute(
                "DELETE FROM retrieval_feedback WHERE id = ?",
                (int(feedback_id),),
            )
        if cursor.rowcount:
            self.calibrate_feedback(
                min_positive=min_positive,
                min_negative=min_negative,
                min_recall=min_recall,
                reranker_model=str(row["reranker_model"] or "") if row else "",
                reranker_fingerprint=(
                    str(row["reranker_fingerprint"] or "") if row else ""
                ),
            )
        return cursor.rowcount > 0

    def benchmark_cases_from_feedback(
        self,
        reranker_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create evaluation cases only from unambiguous human judgments."""

        groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for item in reversed(
            self.list_feedback(
                limit=2000,
                reranker_fingerprint=reranker_fingerprint,
            )
        ):
            key = (
                str(item["query"]).casefold().strip(),
                str(item["source_filter"]),
                str(item["section_filter"]),
                str(item["content_filter"]),
                str(item["date_filter"]),
            )
            groups.setdefault(key, []).append(item)

        cases: list[dict[str, Any]] = []
        for rows in groups.values():
            if any(row["judgment"] == "ambiguous" for row in rows):
                continue
            relevant = [row for row in rows if row["judgment"] == "relevant"]
            expected = [
                row for row in rows if row["judgment"] == "expected_passage"
            ]
            hard_negatives = [
                row
                for row in rows
                if row["judgment"] in {"wrong_passage", "wrong_document"}
            ]
            rejected = any(
                row["judgment"] == "no_relevant_result" for row in rows
            )
            if not relevant and not expected and not rejected:
                # Incorrect-result labels are hard negatives, but they do not prove
                # that the collection has no answer to the query.
                continue
            first = rows[0]
            case: dict[str, Any] = {
                "id": f"feedback-{len(cases) + 1:04d}",
                "query": first["query"],
                "answerable": bool(relevant or expected),
                "split": "calibration",
                "category": "feedback",
                "language": "und",
            }
            if first["source_filter"]:
                case["source_filter"] = first["source_filter"]
            if relevant or expected:
                targets: list[dict[str, Any]] = []
                seen_targets: set[tuple[str, str, int | None]] = set()
                for row in relevant:
                    chunk_id = str(row["chunk_id"] or "")
                    source_id = str(row["source_id"] or "")
                    page = _optional_positive_int(row["page_start"])
                    key = (chunk_id, source_id, page)
                    if key == ("", "", None) or key in seen_targets:
                        continue
                    seen_targets.add(key)
                    target: dict[str, Any] = {}
                    if chunk_id:
                        target["chunk_id"] = chunk_id
                    if source_id and page is not None:
                        target.update({"source_id": source_id, "page": page})
                    targets.append(target)
                for row in expected:
                    source_id = str(row["expected_source_id"] or "")
                    page = _optional_positive_int(row["expected_page"])
                    key = ("", source_id, page)
                    if not source_id or page is None or key in seen_targets:
                        continue
                    seen_targets.add(key)
                    targets.append({"source_id": source_id, "page": page})
                if targets:
                    case["relevant_targets"] = targets
                case["relevant_chunk_ids"] = list(
                    dict.fromkeys(row["chunk_id"] for row in relevant if row["chunk_id"])
                )
                locations = []
                seen_locations: set[tuple[str, int]] = set()
                for row in relevant:
                    source_id = str(row["source_id"] or "")
                    page = _optional_positive_int(row["page_start"])
                    if not source_id or page is None:
                        continue
                    location_key = (source_id, page)
                    if location_key in seen_locations:
                        continue
                    seen_locations.add(location_key)
                    locations.append({"source_id": source_id, "page": page})
                for row in expected:
                    source_id = str(row["expected_source_id"] or "")
                    page = _optional_positive_int(row["expected_page"])
                    if not source_id or page is None:
                        continue
                    location_key = (source_id, page)
                    if location_key in seen_locations:
                        continue
                    seen_locations.add(location_key)
                    locations.append({"source_id": source_id, "page": page})
                if locations:
                    case["relevant_locations"] = locations
            hard_negative_ids = list(
                dict.fromkeys(
                    str(row["chunk_id"])
                    for row in hard_negatives
                    if row["chunk_id"]
                )
            )
            if hard_negative_ids:
                case["hard_negative_chunk_ids"] = hard_negative_ids
            cases.append(case)
        return cases

    def benchmark_jsonl(self, reranker_fingerprint: str | None = None) -> str:
        cases = self.benchmark_cases_from_feedback(reranker_fingerprint)
        if not cases:
            return ""
        return "\n".join(
            json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases
        ) + "\n"

    @staticmethod
    def _validate_benchmark_jsonl(content: str) -> tuple[str, int, dict[str, Any]]:
        normalized: list[str] = []
        splits: dict[str, int] = {}
        categories: dict[str, int] = {}
        languages: dict[str, int] = {}
        hard_negative_count = 0
        for line_number, raw in enumerate(str(content).splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid benchmark JSON on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(case, dict) or not str(case.get("query") or "").strip():
                raise ValueError(f"Benchmark line {line_number} needs a query.")
            case = dict(case)
            case["query"] = str(case["query"]).strip()
            case["id"] = str(case.get("id") or f"case-{line_number:04d}")[:200]
            split = str(case.get("split") or "test").strip().casefold()
            if split not in {"calibration", "test"}:
                raise ValueError(
                    f"Benchmark line {line_number} split must be calibration or test."
                )
            category = str(case.get("category") or "general").strip()[:120] or "general"
            language = str(case.get("language") or "und").strip()[:40] or "und"
            case["split"] = split
            case["category"] = category
            case["language"] = language
            for key in ("relevant_chunk_ids", "hard_negative_chunk_ids"):
                values = case.get(key) or []
                if not isinstance(values, list):
                    raise ValueError(f"Benchmark line {line_number} {key} must be a list.")
                case[key] = list(
                    dict.fromkeys(str(value).strip() for value in values if str(value).strip())
                )
            locations = case.get("relevant_locations") or []
            if not isinstance(locations, list):
                raise ValueError(
                    f"Benchmark line {line_number} relevant_locations must be a list."
                )
            clean_locations: list[dict[str, Any]] = []
            for location in locations:
                if not isinstance(location, dict):
                    raise ValueError(
                        f"Benchmark line {line_number} has an invalid relevant location."
                    )
                source_id = str(location.get("source_id") or "").strip()
                page = _optional_positive_int(location.get("page"))
                if not source_id or page is None:
                    raise ValueError(
                        f"Benchmark line {line_number} locations need source_id and page."
                    )
                clean_locations.append({"source_id": source_id, "page": page})
            case["relevant_locations"] = clean_locations
            targets = case.get("relevant_targets") or []
            if not isinstance(targets, list):
                raise ValueError(
                    f"Benchmark line {line_number} relevant_targets must be a list."
                )
            clean_targets: list[dict[str, Any]] = []
            seen_targets: set[tuple[str, str, int | None]] = set()
            for target in targets:
                if not isinstance(target, dict):
                    raise ValueError(
                        f"Benchmark line {line_number} has an invalid relevance target."
                    )
                chunk_id = str(target.get("chunk_id") or "").strip()
                source_id = str(target.get("source_id") or "").strip()
                page = _optional_positive_int(target.get("page"))
                if not chunk_id and (not source_id or page is None):
                    raise ValueError(
                        f"Benchmark line {line_number} relevance targets need "
                        "a chunk_id or source_id and page."
                    )
                key = (chunk_id, source_id, page)
                if key in seen_targets:
                    continue
                seen_targets.add(key)
                clean_target: dict[str, Any] = {}
                if chunk_id:
                    clean_target["chunk_id"] = chunk_id
                if source_id and page is not None:
                    clean_target.update({"source_id": source_id, "page": page})
                clean_targets.append(clean_target)
            case["relevant_targets"] = clean_targets
            if bool(case.get("answerable", True)) and not (
                case.get("relevant_targets")
                or case.get("relevant_chunk_ids")
                or case.get("relevant_locations")
            ):
                raise ValueError(
                    f"Answerable benchmark line {line_number} needs a relevant chunk or location."
                )
            splits[split] = splits.get(split, 0) + 1
            categories[category] = categories.get(category, 0) + 1
            languages[language] = languages.get(language, 0) + 1
            hard_negative_count += len(case["hard_negative_chunk_ids"])
            normalized.append(json.dumps(case, ensure_ascii=False, sort_keys=True))
        if not normalized:
            raise ValueError("The benchmark contains no cases.")
        metadata = {
            "splits": splits,
            "categories": categories,
            "languages": languages,
            "hard_negative_count": hard_negative_count,
        }
        return "\n".join(normalized) + "\n", len(normalized), metadata

    def save_benchmark(self, name: str, content: str) -> dict[str, Any]:
        name = str(name or "").strip()[:160]
        if not name:
            raise ValueError("Benchmark name is required.")
        if len(content.encode("utf-8")) > 10 * 1024 * 1024:
            raise ValueError("Benchmark files are limited to 10 MB.")
        normalized, case_count, metadata = self._validate_benchmark_jsonl(content)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        now = utc_now()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id, version, content_hash FROM quality_benchmarks WHERE name = ?",
                (name,),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO quality_benchmarks(
                        name, version, content_hash, case_count, content_jsonl,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, digest, case_count, normalized, metadata_json, now, now),
                )
                benchmark_id = int(cursor.lastrowid)
                version = 1
            else:
                benchmark_id = int(existing["id"])
                version = int(existing["version"])
                if str(existing["content_hash"]) != digest:
                    version += 1
                connection.execute(
                    """
                    UPDATE quality_benchmarks
                    SET version = ?, content_hash = ?, case_count = ?,
                        content_jsonl = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        version,
                        digest,
                        case_count,
                        normalized,
                        metadata_json,
                        now,
                        benchmark_id,
                    ),
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO quality_benchmark_versions(
                    benchmark_id, version, content_hash, case_count,
                    content_jsonl, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    benchmark_id,
                    version,
                    digest,
                    case_count,
                    normalized,
                    metadata_json,
                    now,
                ),
            )
        return self.get_benchmark(benchmark_id)

    @staticmethod
    def _benchmark_row(row: sqlite3.Row) -> dict[str, Any]:
        output = dict(row)
        try:
            output["metadata"] = json.loads(str(output.pop("metadata_json", "{}") or "{}"))
        except json.JSONDecodeError:
            output["metadata"] = {}
        return output

    def list_benchmarks(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, version, content_hash, case_count,
                       metadata_json, created_at, updated_at
                FROM quality_benchmarks ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._benchmark_row(row) for row in rows]

    def get_benchmark(self, benchmark_id: int, *, include_content: bool = False) -> dict[str, Any]:
        columns = "*" if include_content else (
            "id, name, version, content_hash, case_count, metadata_json, "
            "created_at, updated_at"
        )
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT {columns} FROM quality_benchmarks WHERE id = ?",
                (int(benchmark_id),),
            ).fetchone()
        if row is None:
            raise KeyError(benchmark_id)
        return self._benchmark_row(row)

    def get_benchmark_version(
        self,
        benchmark_id: int,
        version: int,
        *,
        include_content: bool = False,
    ) -> dict[str, Any]:
        columns = "*" if include_content else (
            "id, benchmark_id, version, content_hash, case_count, metadata_json, created_at"
        )
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT {columns} FROM quality_benchmark_versions "
                "WHERE benchmark_id = ? AND version = ?",
                (int(benchmark_id), int(version)),
            ).fetchone()
        if row is None:
            raise KeyError((benchmark_id, version))
        return self._benchmark_row(row)

    def create_experiment(
        self,
        name: str,
        benchmark_id: int | None,
        benchmark_name: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        clean_name = str(name or "").strip()[:160] or f"Experiment {now}"
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO quality_experiments(
                    name, benchmark_id, benchmark_name, status, config_json,
                    results_json, error, production, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, '{}', '', 0, ?, ?)
                """,
                (
                    clean_name,
                    int(benchmark_id) if benchmark_id is not None else None,
                    str(benchmark_name)[:160],
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            experiment_id = int(cursor.lastrowid)
        return self.get_experiment(experiment_id)

    @staticmethod
    def _experiment_row(row: sqlite3.Row) -> dict[str, Any]:
        output = dict(row)
        for source, target in (("config_json", "config"), ("results_json", "results")):
            try:
                output[target] = json.loads(str(output.pop(source) or "{}"))
            except json.JSONDecodeError:
                output[target] = {}
        output["production"] = bool(output.get("production"))
        return output

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM quality_experiments WHERE id = ?",
                (int(experiment_id),),
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return self._experiment_row(row)

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM quality_experiments ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [self._experiment_row(row) for row in rows]

    def update_experiment(
        self,
        experiment_id: int,
        *,
        status: str,
        results: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        if status not in {"queued", "running", "complete", "failed"}:
            raise ValueError("Invalid experiment status.")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE quality_experiments
                SET status = ?, results_json = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    json.dumps(results or {}, ensure_ascii=False, sort_keys=True),
                    str(error or "")[:4000],
                    utc_now(),
                    int(experiment_id),
                ),
            )
        if not cursor.rowcount:
            raise KeyError(experiment_id)
        return self.get_experiment(experiment_id)

    def set_production_experiment(self, experiment_id: int) -> dict[str, Any]:
        experiment = self.get_experiment(experiment_id)
        if experiment["status"] != "complete":
            raise ValueError("Only a completed experiment can become the production default.")
        regression = (experiment.get("results") or {}).get("regression") or {}
        if regression and not bool(regression.get("passed", False)):
            raise ValueError(
                "This experiment failed its quality regression threshold and cannot become production."
            )
        reranker = str((experiment.get("config") or {}).get("reranker") or "")
        if reranker not in {"gte", "bge"}:
            raise ValueError(
                "Production experiments must select exactly one reranker. "
                "Run GTE or BGE separately before promotion."
            )
        with self.connect() as connection:
            connection.execute("UPDATE quality_experiments SET production = 0")
            connection.execute(
                "UPDATE quality_experiments SET production = 1, updated_at = ? WHERE id = ?",
                (utc_now(), int(experiment_id)),
            )
        return self.get_experiment(experiment_id)

    def production_experiment(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM quality_experiments
                WHERE production = 1 AND status = 'complete'
                ORDER BY updated_at DESC LIMIT 1
                """
            ).fetchone()
        return self._experiment_row(row) if row is not None else None

    def add_notification(
        self,
        *,
        kind: str,
        title: str,
        message: str = "",
        status: str = "info",
        task_key: str = "",
    ) -> dict[str, Any]:
        if status not in {"info", "success", "warning", "error"}:
            raise ValueError("Invalid notification status.")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO notifications(
                    kind, title, message, status, task_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(kind or "system")[:80],
                    str(title or "Update")[:240],
                    str(message or "")[:2000],
                    status,
                    str(task_key or "")[:200],
                    utc_now(),
                ),
            )
            connection.execute(
                """
                DELETE FROM notifications WHERE id NOT IN (
                    SELECT id FROM notifications ORDER BY id DESC LIMIT 500
                )
                """
            )
            row = connection.execute(
                "SELECT * FROM notifications WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        assert row is not None
        return dict(row)

    def list_notifications(
        self,
        *,
        limit: int = 50,
        unread_only: bool = False,
    ) -> list[dict[str, Any]]:
        where = "WHERE read_at IS NULL" if unread_only else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM notifications {where} ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return self._rows(rows)

    def mark_notification_read(self, notification_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE notifications SET read_at = COALESCE(read_at, ?) WHERE id = ?",
                (utc_now(), int(notification_id)),
            )
            row = connection.execute(
                "SELECT * FROM notifications WHERE id = ?",
                (int(notification_id),),
            ).fetchone()
        if not cursor.rowcount or row is None:
            raise KeyError(notification_id)
        return dict(row)

    def schema_status(self) -> dict[str, Any]:
        with self.connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            migrations = self._rows(
                connection.execute(
                    "SELECT version, applied_at FROM schema_migrations ORDER BY version"
                ).fetchall()
            )
        return {
            "current_version": version,
            "target_version": WORKSPACE_SCHEMA_VERSION,
            "migrations": migrations,
        }

    def calibrate_feedback(
        self,
        *,
        min_positive: int = 20,
        min_negative: int = 20,
        min_recall: float = 0.90,
        reranker_model: str = "",
        reranker_fingerprint: str = "",
    ) -> dict[str, Any]:
        """Fit a transparent raw-logit cutoff from explicit result labels."""

        positives: list[float] = []
        negatives: list[float] = []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT judgment, rerank_logit FROM retrieval_feedback
                WHERE rerank_logit IS NOT NULL AND reranker_fingerprint = ?
                """,
                (reranker_fingerprint,),
            ).fetchall()
            previous = connection.execute(
                """
                SELECT enabled FROM quality_calibration_v2
                WHERE reranker_fingerprint = ?
                """,
                (reranker_fingerprint,),
            ).fetchone()
        for row in rows:
            score = _optional_finite_float(row["rerank_logit"])
            if score is None:
                continue
            if row["judgment"] == "relevant":
                positives.append(score)
            else:
                negatives.append(score)

        threshold: float | None = None
        positive_recall: float | None = None
        specificity: float | None = None
        balanced_accuracy: float | None = None
        ready = len(positives) >= max(1, int(min_positive)) and len(negatives) >= max(
            1, int(min_negative)
        )
        if ready:
            candidates = sorted(set([*positives, *negatives]))
            # A threshold just above the largest score permits a valid all-negative
            # candidate when the requested recall allows it.
            candidates.append(max(candidates) + 1e-9)
            scored: list[tuple[float, float, float, float]] = []
            for candidate in candidates:
                recall = sum(score >= candidate for score in positives) / len(positives)
                true_negative_rate = sum(
                    score < candidate for score in negatives
                ) / len(negatives)
                if recall + 1e-12 < float(min_recall):
                    continue
                balanced = (recall + true_negative_rate) / 2.0
                scored.append((balanced, true_negative_rate, candidate, recall))
            if scored:
                balanced_accuracy, specificity, threshold, positive_recall = max(scored)
            else:
                ready = False

        enabled = int(previous["enabled"]) if previous is not None else 1
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO quality_calibration_v2(
                    reranker_fingerprint, reranker_model, threshold,
                    positive_count, negative_count,
                    positive_recall, specificity, balanced_accuracy,
                    ready, enabled, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reranker_fingerprint) DO UPDATE SET
                    reranker_model = excluded.reranker_model,
                    threshold = excluded.threshold,
                    positive_count = excluded.positive_count,
                    negative_count = excluded.negative_count,
                    positive_recall = excluded.positive_recall,
                    specificity = excluded.specificity,
                    balanced_accuracy = excluded.balanced_accuracy,
                    ready = excluded.ready,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    reranker_fingerprint,
                    reranker_model,
                    threshold,
                    len(positives),
                    len(negatives),
                    positive_recall,
                    specificity,
                    balanced_accuracy,
                    1 if ready else 0,
                    enabled,
                    now,
                ),
            )
        return self.calibration_status(
            min_positive=min_positive,
            min_negative=min_negative,
            reranker_model=reranker_model,
            reranker_fingerprint=reranker_fingerprint,
        )

    def set_calibration_enabled(
        self,
        enabled: bool,
        *,
        reranker_model: str = "",
        reranker_fingerprint: str = "",
    ) -> dict[str, Any]:
        status = self.calibration_status(
            reranker_model=reranker_model,
            reranker_fingerprint=reranker_fingerprint,
        )
        if not status["updated_at"]:
            status = self.calibrate_feedback(
                reranker_model=reranker_model,
                reranker_fingerprint=reranker_fingerprint,
            )
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE quality_calibration_v2 SET enabled = ?, updated_at = ?
                WHERE reranker_fingerprint = ?
                """,
                (1 if enabled else 0, utc_now(), reranker_fingerprint),
            )
        return self.calibration_status(
            min_positive=status["minimum_positive"],
            min_negative=status["minimum_negative"],
            reranker_model=reranker_model,
            reranker_fingerprint=reranker_fingerprint,
        )

    def calibration_status(
        self,
        *,
        min_positive: int = 20,
        min_negative: int = 20,
        reranker_model: str = "",
        reranker_fingerprint: str = "",
    ) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM quality_calibration_v2
                WHERE reranker_fingerprint = ?
                """,
                (reranker_fingerprint,),
            ).fetchone()
            if row is None:
                counts = connection.execute(
                    """
                    SELECT
                        SUM(CASE WHEN judgment = 'relevant' AND rerank_logit IS NOT NULL THEN 1 ELSE 0 END) AS positives,
                        SUM(CASE WHEN judgment != 'relevant' AND rerank_logit IS NOT NULL THEN 1 ELSE 0 END) AS negatives
                    FROM retrieval_feedback
                    WHERE reranker_fingerprint = ?
                    """,
                    (reranker_fingerprint,),
                ).fetchone()
                positive_count = int(counts["positives"] or 0)
                negative_count = int(counts["negatives"] or 0)
                return {
                    "threshold": None,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "positive_recall": None,
                    "specificity": None,
                    "balanced_accuracy": None,
                    "ready": False,
                    "enabled": True,
                    "active": False,
                    "minimum_positive": int(min_positive),
                    "minimum_negative": int(min_negative),
                    "reranker_model": reranker_model,
                    "reranker_fingerprint": reranker_fingerprint,
                    "updated_at": "",
                }
        output = dict(row)
        output["ready"] = bool(
            output["ready"]
            and int(output["positive_count"]) >= int(min_positive)
            and int(output["negative_count"]) >= int(min_negative)
        )
        output["enabled"] = bool(output["enabled"])
        output["active"] = bool(output["ready"] and output["enabled"])
        output["minimum_positive"] = int(min_positive)
        output["minimum_negative"] = int(min_negative)
        return output

    def quality_summary(
        self,
        *,
        min_positive: int = 20,
        min_negative: int = 20,
        reranker_model: str = "",
        reranker_fingerprint: str = "",
    ) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT judgment, COUNT(*) AS count
                FROM retrieval_feedback
                WHERE reranker_fingerprint = ?
                GROUP BY judgment
                """,
                (reranker_fingerprint,),
            ).fetchall()
        counts = {judgment: 0 for judgment in FEEDBACK_JUDGMENTS}
        counts.update({str(row["judgment"]): int(row["count"]) for row in rows})
        cases = self.benchmark_cases_from_feedback(reranker_fingerprint)
        return {
            "total": sum(counts.values()),
            "counts": counts,
            "benchmark_cases": len(cases),
            "answerable_cases": sum(bool(case["answerable"]) for case in cases),
            "unanswerable_cases": sum(not case["answerable"] for case in cases),
            "calibration": self.calibration_status(
                min_positive=min_positive,
                min_negative=min_negative,
                reranker_model=reranker_model,
                reranker_fingerprint=reranker_fingerprint,
            ),
        }

    def markdown_export(self, bookmark_ids: Sequence[int]) -> str:
        bookmarks = self.bookmarks_by_ids(bookmark_ids)
        lines = ["# Research excerpts", ""]
        for index, item in enumerate(bookmarks, start=1):
            title = item["document_title"] or item["source_id"]
            lines.extend([f"## {index}. {title}", ""])
            details = _bookmark_details(item)
            if details:
                lines.extend([details, ""])
            excerpt = str(item["excerpt"]).replace("\r\n", "\n")
            lines.extend([*(f"> {line}" if line else ">" for line in excerpt.split("\n")), ""])
            note = str(item.get("note") or "").strip()
            if note:
                lines.extend([f"**Note:** {note}", ""])
            citation = str(item.get("citation_label") or "").strip()
            url = str(item.get("citation_url") or "").strip()
            if citation and url:
                lines.extend([f"**Citation:** [{citation}]({url})", ""])
            elif citation:
                lines.extend([f"**Citation:** {citation}", ""])
            elif url:
                lines.extend([f"**Source:** {url}", ""])
        return "\n".join(lines).rstrip() + "\n"

    def docx_export(self, bookmark_ids: Sequence[int]) -> bytes:
        bookmarks = self.bookmarks_by_ids(bookmark_ids)
        paragraphs: list[tuple[str, str]] = [("Research excerpts", "Title")]
        for index, item in enumerate(bookmarks, start=1):
            title = item["document_title"] or item["source_id"]
            paragraphs.append((f"{index}. {title}", "Heading1"))
            details = _bookmark_details(item)
            if details:
                paragraphs.append((details, "Normal"))
            paragraphs.append((str(item["excerpt"]), "Quote"))
            note = str(item.get("note") or "").strip()
            if note:
                paragraphs.append((f"Note: {note}", "Normal"))
            citation = str(item.get("citation_label") or "").strip()
            url = str(item.get("citation_url") or "").strip()
            if citation or url:
                paragraphs.append((f"Citation: {citation} {url}".strip(), "Normal"))
        return _minimal_docx(paragraphs)


def _optional_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _optional_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bookmark_details(item: dict[str, Any]) -> str:
    pages = ""
    if item.get("page_start"):
        pages = f"Page {item['page_start']}"
        if item.get("page_end") and item["page_end"] != item["page_start"]:
            pages += f"-{item['page_end']}"
    details = [
        str(item.get("section") or "").strip(),
        pages,
        str(item.get("collection_name") or "").strip(),
    ]
    return " · ".join(detail for detail in details if detail)


def _minimal_docx(paragraphs: Sequence[tuple[str, str]]) -> bytes:
    """Create a portable .docx without requiring python-docx."""

    paragraph_xml: list[str] = []
    for text, style in paragraphs:
        runs = []
        for line_index, line in enumerate(str(text).replace("\r\n", "\n").split("\n")):
            if line_index:
                runs.append("<w:r><w:br/></w:r>")
            runs.append(
                '<w:r><w:t xml:space="preserve">'
                + escape(line)
                + "</w:t></w:r>"
            )
        style_xml = (
            f'<w:pPr><w:pStyle w:val="{escape(style)}"/></w:pPr>'
            if style
            else ""
        )
        paragraph_xml.append(f"<w:p>{style_xml}{''.join(runs)}</w:p>")

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraph_xml)}<w:sectPr/></w:body></w:document>"
    )
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr><w:rPr><w:i/></w:rPr></w:style>
</w:styles>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    package_relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document_relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_relationships)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/_rels/document.xml.rels", document_relationships)
    return output.getvalue()


def parse_export_ids(raw: str | Sequence[int]) -> list[int]:
    if not isinstance(raw, str):
        return list(dict.fromkeys(int(item) for item in raw))
    output: list[int] = []
    for part in raw.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if value > 0 and value not in output:
            output.append(value)
    return output[:200]


def json_script(value: Any) -> str:
    """Serialize JSON safely for an HTML script element."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
