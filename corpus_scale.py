"""Persistent corpus orchestration, health reporting, and portable snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from secrets import token_hex
from threading import RLock
from time import monotonic
from typing import Any, Iterable

from document_manager import DocumentRepository


BACKUP_VERSION = 1
PAUSED_EXIT_CODE = 75
_TERMINAL_QUEUE_STATES = {"complete", "failed", "quarantined"}
_CHROMA_SEGMENT_DIRECTORY = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)
_DEBUG_ARTIFACT_NAME = re.compile(
    r"^.+-[0-9a-f]{10}(?:\.md|\.chunks\.jsonl)$",
    flags=re.IGNORECASE,
)


def debug_artifact_paths(debug_dir: Path, source_id: str) -> tuple[Path, Path]:
    """Return the deterministic Markdown and chunk-debug paths for a source."""

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_id).stem)
    suffix = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:10]
    base_path = debug_dir / f"{safe_stem}-{suffix}"
    return base_path.with_suffix(".md"), base_path.with_suffix(".chunks.jsonl")


def remove_debug_artifacts(debug_dir: Path | None, source_id: str) -> int:
    """Delete only the two deterministic debug exports owned by a source."""

    if debug_dir is None:
        return 0
    removed = 0
    for path in debug_artifact_paths(debug_dir, source_id):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


class CorpusScaleManager:
    """Coordinate ingestion and corpus maintenance across app restarts."""

    def __init__(
        self,
        repository: DocumentRepository,
        *,
        state_path: Path,
        db_dir: Path,
        workspace_db: Path,
        backup_dir: Path,
        debug_dir: Path | None = None,
        collection_name: str = "technical_docs_qwen_v1",
        lexical_db: Path | None = None,
    ) -> None:
        self.repository = repository
        self.state_path = state_path.resolve()
        self.db_dir = db_dir.resolve()
        self.workspace_db = workspace_db.resolve()
        self.backup_dir = backup_dir.resolve()
        self.debug_dir = debug_dir.resolve() if debug_dir else None
        self.collection_name = str(collection_name)
        self.lexical_db = (
            lexical_db.resolve() if lexical_db else self.db_dir / "lexical.sqlite3"
        )
        self._lock = RLock()
        self._health_cache: tuple[float, dict[str, Any]] | None = None
        self._hash_cache: dict[str, tuple[int, int, str]] = {}
        self.recover_interrupted()

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "version": 2,
            "paused": False,
            "pause_requested_at": None,
            "run_id": None,
            "items": [],
            "migration_history": [],
            "updated_at": utc_now(),
        }

    def _read(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return self._default_state()
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_state()
        if not isinstance(state, dict):
            return self._default_state()
        previous_version = int(state.get("version") or 1)
        state["version"] = 2
        state.setdefault("paused", False)
        state.setdefault("pause_requested_at", None)
        state.setdefault("run_id", None)
        state["items"] = [
            item for item in state.get("items", []) if isinstance(item, dict)
        ][-300:]
        state.setdefault("migration_history", [])
        if previous_version < 2:
            backup = self.state_path.with_name(
                f"{self.state_path.name}.pre-v2-migration.bak"
            )
            if not backup.exists():
                shutil.copy2(self.state_path, backup)
            state["migration_history"] = [
                *state["migration_history"],
                {"from": previous_version, "to": 2, "applied_at": utc_now()},
            ][-20:]
            self._write(state)
        return state

    def _write(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def recover_interrupted(self) -> None:
        with self._lock:
            state = self._read()
            changed = False
            for item in state["items"]:
                if item.get("status") == "processing":
                    item["status"] = "queued"
                    item["error"] = "The previous app session ended during ingestion."
                    if hasattr(self.repository, "transition"):
                        try:
                            self.repository.transition(
                                str(item.get("source_id") or ""),
                                "pending",
                                reason="Recovered interrupted ingestion",
                                error=str(item["error"]),
                            )
                        except (OSError, ValueError):
                            pass
                    changed = True
            if changed:
                self._write(state)

    def prepare_queue(self, summary: dict[str, Any], *, force: bool) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            history = [
                item
                for item in state["items"]
                if item.get("status") in _TERMINAL_QUEUE_STATES
            ][-100:]
            sources = (
                [str(item["source_id"]) for item in summary.get("documents", [])]
                if force
                else [str(item) for item in summary.get("pending_sources", [])]
            )
            deleted = [str(item) for item in summary.get("deleted_sources", [])]
            run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S") + "-" + token_hex(3)
            now = utc_now()
            items: list[dict[str, Any]] = list(history)
            for source_id in dict.fromkeys(sources):
                items.append(
                    {
                        "id": token_hex(6),
                        "run_id": run_id,
                        "source_id": source_id,
                        "action": "ingest",
                        "status": "queued",
                        "attempts": 0,
                        "enqueued_at": now,
                        "started_at": None,
                        "finished_at": None,
                        "error": "",
                    }
                )
            for source_id in dict.fromkeys(deleted):
                items.append(
                    {
                        "id": token_hex(6),
                        "run_id": run_id,
                        "source_id": source_id,
                        "action": "delete",
                        "status": "queued",
                        "attempts": 0,
                        "enqueued_at": now,
                        "started_at": None,
                        "finished_at": None,
                        "error": "",
                    }
                )
            state.update(
                {
                    "paused": False,
                    "pause_requested_at": None,
                    "run_id": run_id,
                    "items": items,
                }
            )
            self._write(state)
            return self.snapshot()

    def request_pause(self) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            state["paused"] = True
            state["pause_requested_at"] = utc_now()
            self._write(state)
            return self.snapshot()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            state["paused"] = False
            state["pause_requested_at"] = None
            for item in state["items"]:
                if item.get("status") == "processing":
                    item["status"] = "queued"
            self._write(state)
            return self.snapshot()

    def is_paused(self) -> bool:
        with self._lock:
            return bool(self._read().get("paused"))

    def queued_sources(self) -> list[str]:
        with self._lock:
            state = self._read()
            run_id = state.get("run_id")
            return [
                str(item.get("source_id") or "")
                for item in state["items"]
                if item.get("run_id") == run_id
                and item.get("action") == "ingest"
                and item.get("status") == "queued"
            ]

    def queued_deletions(self) -> list[str]:
        with self._lock:
            state = self._read()
            run_id = state.get("run_id")
            return [
                str(item.get("source_id") or "")
                for item in state["items"]
                if item.get("run_id") == run_id
                and item.get("action") == "delete"
                and item.get("status") == "queued"
            ]

    def mark_event(
        self,
        source_id: str,
        status: str,
        *,
        error: str = "",
    ) -> None:
        with self._lock:
            state = self._read()
            run_id = state.get("run_id")
            candidates = [
                item
                for item in state["items"]
                if item.get("run_id") == run_id
                and item.get("source_id") == source_id
                and item.get("action") == "ingest"
            ]
            if not candidates:
                return
            item = candidates[-1]
            now = utc_now()
            if status == "processing":
                item["status"] = status
                item["started_at"] = item.get("started_at") or now
                item["attempts"] = int(item.get("attempts") or 0) + 1
            else:
                item["status"] = status
                item["finished_at"] = now
            if error:
                item["error"] = str(error)[:2000]
            self._write(state)

    def mark_deletions_complete(self) -> list[str]:
        with self._lock:
            state = self._read()
            run_id = state.get("run_id")
            completed: list[str] = []
            now = utc_now()
            for item in state["items"]:
                if (
                    item.get("run_id") == run_id
                    and item.get("action") == "delete"
                    and item.get("status") == "queued"
                ):
                    item["status"] = "complete"
                    item["started_at"] = item.get("started_at") or now
                    item["finished_at"] = now
                    item["attempts"] = int(item.get("attempts") or 0) + 1
                    completed.append(str(item.get("source_id") or ""))
            self._write(state)
            return completed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            run_id = state.get("run_id")
            current = [item for item in state["items"] if item.get("run_id") == run_id]
            counts = Counter(str(item.get("status") or "unknown") for item in current)
            return {
                "paused": bool(state.get("paused")),
                "pause_requested_at": state.get("pause_requested_at"),
                "run_id": run_id,
                "items": current,
                "counts": dict(counts),
                "remaining": sum(
                    item.get("status") in {"queued", "processing"} for item in current
                ),
                "updated_at": state.get("updated_at"),
            }

    def reconcile_indexed_documents(self, summary: dict[str, Any]) -> list[str]:
        """Finish stale queue items whose exact documents are already indexed."""

        pending = {str(item) for item in summary.get("pending_sources", [])}
        indexed = {
            str(item.get("source_id") or "")
            for item in summary.get("documents", [])
            if item.get("status") == "indexed"
        }
        with self._lock:
            state = self._read()
            run_id = state.get("run_id")
            now = utc_now()
            reconciled: list[str] = []
            for item in state["items"]:
                source_id = str(item.get("source_id") or "")
                if (
                    item.get("run_id") == run_id
                    and item.get("action") == "ingest"
                    and item.get("status") in {"queued", "processing"}
                    and source_id in indexed
                    and source_id not in pending
                ):
                    item["status"] = "complete"
                    item["started_at"] = item.get("started_at") or now
                    item["finished_at"] = now
                    item["error"] = "Already indexed; stale queue state reconciled."
                    reconciled.append(source_id)
            if reconciled:
                self._write(state)
                self.invalidate_health()
            return reconciled

    def reconcile_completed_removals(self, summary: dict[str, Any]) -> list[str]:
        """Clear removal markers only when no index layer retains the source."""

        pending_removals = {
            str(source_id)
            for source_id in summary.get("deleted_sources", [])
            if source_id
        }
        if not pending_removals:
            return []
        try:
            snapshot = self._index_snapshot()
        except (OSError, sqlite3.Error):
            return []
        indexed_sources = set().union(
            snapshot["manifest_sources"],
            snapshot["dense_sources"],
            snapshot["lexical_sources"],
        )
        reconciled = sorted(pending_removals - indexed_sources)
        if reconciled:
            self.repository.clear_pending_sources((), tuple(reconciled))
            self.invalidate_health()
        return reconciled

    def invalidate_health(self) -> None:
        self._health_cache = None

    @staticmethod
    def _sqlite_free_bytes(path: Path) -> int:
        if not path.is_file():
            return 0
        try:
            connection = sqlite3.connect(
                f"file:{path.as_posix()}?mode=ro",
                uri=True,
            )
            try:
                page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
                free_pages = int(
                    connection.execute("PRAGMA freelist_count").fetchone()[0]
                )
            finally:
                connection.close()
        except sqlite3.Error:
            return 0
        return page_size * free_pages

    def _referenced_chroma_segments(self) -> set[str]:
        database = self.db_dir / "chroma.sqlite3"
        if not database.is_file():
            return set()
        connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro",
            uri=True,
        )
        try:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT id FROM segments WHERE scope = 'VECTOR'"
                )
            }
        except sqlite3.Error:
            return set()
        finally:
            connection.close()

    def _chroma_segment_directories(self) -> list[Path]:
        if not self.db_dir.is_dir():
            return []
        return [
            child
            for child in self.db_dir.iterdir()
            if child.is_dir()
            and _CHROMA_SEGMENT_DIRECTORY.fullmatch(child.name)
            and (child / "header.bin").is_file()
        ]

    def _orphan_chroma_segments(self) -> list[Path]:
        referenced = self._referenced_chroma_segments()
        return [
            path
            for path in self._chroma_segment_directories()
            if path.name not in referenced
        ]

    def _stale_debug_artifacts(self, active_sources: set[str]) -> list[Path]:
        if self.debug_dir is None or not self.debug_dir.is_dir():
            return []
        expected = {
            path.resolve()
            for source_id in active_sources
            for path in debug_artifact_paths(self.debug_dir, source_id)
        }
        return [
            path
            for path in self.debug_dir.iterdir()
            if path.is_file()
            and _DEBUG_ARTIFACT_NAME.fullmatch(path.name)
            and path.resolve() not in expected
        ]

    def reclaimable_storage(self) -> int:
        """Estimate bytes that maintenance can safely reclaim immediately."""

        summary = self.repository.summary()
        active_sources = {
            str(item.get("source_id") or "") for item in summary["documents"]
        }
        database_free = sum(
            self._sqlite_free_bytes(path)
            for path in (
                self.db_dir / "chroma.sqlite3",
                self.lexical_db,
                self.db_dir / "embedding_cache.sqlite3",
            )
        )
        orphan_segments = sum(
            directory_size(path) for path in self._orphan_chroma_segments()
        )
        stale_debug = sum(
            directory_size(path)
            for path in self._stale_debug_artifacts(active_sources)
        )
        return database_free + orphan_segments + stale_debug

    def _index_snapshot(self) -> dict[str, Any]:
        """Read dense, lexical, and manifest identity without loading models."""

        manifest = self.repository._read_json(
            self.repository.manifest_path,
            {"sources": {}},
        )
        manifest_entries = {
            str(source_id): entry
            for source_id, entry in (manifest.get("sources") or {}).items()
            if isinstance(entry, dict) and entry.get("complete")
        }
        manifest_sources = set(manifest_entries)
        manifest_chunks = sum(
            int(entry.get("chunk_count") or 0)
            for entry in manifest_entries.values()
        )

        dense_database = self.db_dir / "chroma.sqlite3"
        dense_chunks = 0
        dense_sources: set[str] = set()
        if dense_database.is_file():
            connection = sqlite3.connect(
                f"file:{dense_database.as_posix()}?mode=ro",
                uri=True,
            )
            try:
                dense_chunks = int(
                    connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                )
                dense_sources = {
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT DISTINCT string_value
                        FROM embedding_metadata
                        WHERE key IN ('source_id', 'source')
                          AND string_value IS NOT NULL
                        """
                    )
                }
            finally:
                connection.close()

        lexical_chunks = 0
        lexical_sources: set[str] = set()
        if self.lexical_db.is_file():
            connection = sqlite3.connect(
                f"file:{self.lexical_db.as_posix()}?mode=ro",
                uri=True,
            )
            try:
                lexical_chunks = int(
                    connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                )
                lexical_sources = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT source_id FROM chunks_fts"
                    )
                }
            finally:
                connection.close()

        return {
            "manifest_chunks": manifest_chunks,
            "manifest_sources": manifest_sources,
            "dense_chunks": dense_chunks,
            "dense_sources": dense_sources,
            "lexical_chunks": lexical_chunks,
            "lexical_sources": lexical_sources,
        }

    def _verify_index_snapshot(self) -> dict[str, Any]:
        snapshot = self._index_snapshot()
        counts = {
            snapshot["manifest_chunks"],
            snapshot["dense_chunks"],
            snapshot["lexical_chunks"],
        }
        if len(counts) != 1:
            raise RuntimeError(
                "Storage optimization requires matching manifest, dense, and "
                "lexical chunk counts; found "
                f"{snapshot['manifest_chunks']}, {snapshot['dense_chunks']}, "
                f"and {snapshot['lexical_chunks']}. Apply index changes first."
            )
        if not (
            snapshot["manifest_sources"]
            == snapshot["dense_sources"]
            == snapshot["lexical_sources"]
        ):
            raise RuntimeError(
                "Storage optimization requires matching manifest, dense, and "
                "lexical source sets. Apply index changes first."
            )
        return snapshot

    @staticmethod
    def _managed_chroma_paths(root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        return [
            child
            for child in root.iterdir()
            if (
                child.is_file()
                and child.name.startswith("chroma.sqlite3")
            )
            or (
                child.is_dir()
                and _CHROMA_SEGMENT_DIRECTORY.fullmatch(child.name)
                and (child / "header.bin").is_file()
            )
        ]

    def _swap_rebuilt_chroma(self, rebuilt_root: Path, rollback_root: Path) -> None:
        old_paths = self._managed_chroma_paths(self.db_dir)
        new_paths = self._managed_chroma_paths(rebuilt_root)
        if not any(path.name == "chroma.sqlite3" for path in new_paths):
            raise RuntimeError("The rebuilt Chroma index has no database file.")

        moved_old: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        try:
            for old_path in old_paths:
                rollback_path = rollback_root / old_path.name
                os.replace(old_path, rollback_path)
                moved_old.append((old_path, rollback_path))
            for new_path in new_paths:
                destination = self.db_dir / new_path.name
                shutil.move(str(new_path), str(destination))
                installed.append(destination)
            self._verify_index_snapshot()
        except Exception:
            for installed_path in installed:
                resolved = installed_path.resolve()
                if not resolved.is_relative_to(self.db_dir):
                    continue
                if installed_path.is_dir():
                    shutil.rmtree(installed_path)
                elif installed_path.exists():
                    installed_path.unlink()
            for original_path, rollback_path in reversed(moved_old):
                if rollback_path.exists():
                    os.replace(rollback_path, original_path)
            raise

    def _rebuild_chroma_index(self, expected_chunks: int) -> bool:
        if expected_chunks < 0:
            raise ValueError("Expected chunk count cannot be negative.")
        if not (self.db_dir / "chroma.sqlite3").is_file():
            return False

        import chromadb

        with tempfile.TemporaryDirectory(
            prefix="reference-desk-optimize-",
            dir=str(self.db_dir.parent),
        ) as temporary_name:
            staging = Path(temporary_name)
            rebuilt_root = staging / "rebuilt-index"
            rollback_root = staging / "rollback"
            rebuilt_root.mkdir()
            rollback_root.mkdir()

            source_client = chromadb.PersistentClient(path=str(self.db_dir))
            target_client = chromadb.PersistentClient(path=str(rebuilt_root))
            try:
                source = next(
                    (
                        collection
                        for collection in source_client.list_collections()
                        if collection.name == self.collection_name
                    ),
                    None,
                )
                if source is None and expected_chunks:
                    raise RuntimeError(
                        f"Chroma collection {self.collection_name!r} is missing."
                    )
                hnsw = (
                    dict((source.configuration or {}).get("hnsw") or {})
                    if source is not None
                    else {}
                )
                hnsw.pop("embedding_function", None)
                metadata = (
                    dict(source.metadata or {}) if source is not None else {}
                )
                target = target_client.create_collection(
                    name=self.collection_name,
                    metadata=metadata or None,
                    configuration={"hnsw": hnsw} if hnsw else None,
                )
                batch_size = 250
                for offset in range(0, expected_chunks, batch_size):
                    if source is None:
                        raise RuntimeError("The source Chroma collection is unavailable.")
                    batch = source.get(
                        limit=batch_size,
                        offset=offset,
                        include=["embeddings", "documents", "metadatas"],
                    )
                    ids = list(batch.get("ids") or [])
                    if not ids:
                        continue
                    embeddings = batch.get("embeddings")
                    if hasattr(embeddings, "tolist"):
                        embeddings = embeddings.tolist()
                    target.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=list(batch.get("documents") or []),
                        metadatas=list(batch.get("metadatas") or []),
                    )
                if target.count() != expected_chunks:
                    raise RuntimeError(
                        "The rebuilt Chroma index did not preserve every active chunk."
                    )
            finally:
                target_client.close()
                source_client.close()

            self._swap_rebuilt_chroma(rebuilt_root, rollback_root)
        return True

    @staticmethod
    def _vacuum_database(path: Path) -> int:
        if not path.is_file():
            return 0
        before = path.stat().st_size
        connection = sqlite3.connect(path, timeout=120, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout = 120000")
            journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).casefold()
            if journal_mode == "wal":
                checkpoint = connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
                if checkpoint and int(checkpoint[0]) != 0:
                    raise RuntimeError(f"Could not checkpoint {path.name}.")
            connection.execute("VACUUM")
            connection.execute("PRAGMA optimize")
        finally:
            connection.close()
        return max(0, before - path.stat().st_size)

    def optimize_storage(self) -> dict[str, Any]:
        """Back up, rebuild, compact, and verify the active local index."""

        with self._lock:
            before_snapshot = self._verify_index_snapshot()
            summary = self.repository.summary()
            active_sources = {
                str(item.get("source_id") or "") for item in summary["documents"]
            }
            stale_debug = self._stale_debug_artifacts(active_sources)
            orphan_segments = self._orphan_chroma_segments()
            before_bytes = directory_size(self.db_dir) + (
                directory_size(self.debug_dir) if self.debug_dir else 0
            )
            backup = self.create_backup("Before storage optimization")

            rebuilt = self._rebuild_chroma_index(
                int(before_snapshot["manifest_chunks"])
            )
            removed_debug = 0
            for path in stale_debug:
                path.unlink(missing_ok=True)
                removed_debug += 1

            compacted: dict[str, int] = {}
            for database in (
                self.db_dir / "chroma.sqlite3",
                self.lexical_db,
                self.db_dir / "embedding_cache.sqlite3",
            ):
                if database.is_file():
                    compacted[database.name] = self._vacuum_database(database)

            after_snapshot = self._verify_index_snapshot()
            after_bytes = directory_size(self.db_dir) + (
                directory_size(self.debug_dir) if self.debug_dir else 0
            )
            self.invalidate_health()
            return {
                "optimized": True,
                "backup": backup,
                "reclaimed_bytes": max(0, before_bytes - after_bytes),
                "before_bytes": before_bytes,
                "after_bytes": after_bytes,
                "dense_index_rebuilt": rebuilt,
                "orphan_segments_removed": len(orphan_segments),
                "debug_files_removed": removed_debug,
                "databases_compacted": compacted,
                "chunks_verified": int(after_snapshot["manifest_chunks"]),
                "sources_verified": len(after_snapshot["manifest_sources"]),
            }

    def _document_hash(self, document: dict[str, Any]) -> str:
        source_id = str(document["source_id"])
        path = self.repository.resolve_source(source_id)
        stat = path.stat()
        known = str(document.get("file_hash") or "")
        if known and document.get("status") == "indexed":
            return known
        cached = self._hash_cache.get(source_id)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        digest = self.repository._file_hash(path)
        self._hash_cache[source_id] = (stat.st_mtime_ns, stat.st_size, digest)
        return digest

    def health(self, *, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            if (
                not refresh
                and self._health_cache is not None
                and monotonic() - self._health_cache[0] < 10
            ):
                return self._health_cache[1]

            summary = self.repository.summary()
            documents = summary["documents"]
            manifest = self.repository._read_json(
                self.repository.manifest_path,
                {"sources": {}},
            )
            manifest_sources = manifest.get("sources") or {}
            active_sources = {str(item["source_id"]) for item in documents}
            stale_sources = sorted(set(manifest_sources) - active_sources)
            invalid_sources = sorted(
                str(item["source_id"])
                for item in documents
                if item.get("pages") is None
            )
            hash_groups: dict[str, list[str]] = {}
            for document in documents:
                try:
                    digest = self._document_hash(document)
                except OSError:
                    continue
                hash_groups.setdefault(digest, []).append(str(document["source_id"]))
            duplicate_groups = [
                {"file_hash": digest, "sources": sorted(sources)}
                for digest, sources in hash_groups.items()
                if len(sources) > 1
            ]

            storage = {
                "documents": directory_size(self.repository.pdf_dir),
                "index": directory_size(self.db_dir),
                "workspace": sum(
                    directory_size(Path(str(self.workspace_db) + suffix))
                    for suffix in ("", "-wal", "-shm")
                ),
                "trash": directory_size(self.repository.trash_dir),
                "quarantine": directory_size(self.repository.quarantine_dir),
                "revisions": directory_size(self.repository.revision_dir),
                "backups": directory_size(self.backup_dir),
                "debug": directory_size(self.debug_dir) if self.debug_dir else 0,
                "reclaimable": self.reclaimable_storage(),
            }
            storage["active"] = (
                storage["documents"] + storage["index"] + storage["workspace"]
            )
            storage["total"] = sum(
                storage[key]
                for key in (
                    "documents",
                    "index",
                    "workspace",
                    "trash",
                    "quarantine",
                    "revisions",
                    "backups",
                    "debug",
                )
            )

            issues: list[dict[str, Any]] = []
            if invalid_sources:
                issues.append(
                    {
                        "severity": "critical",
                        "label": "Unreadable PDFs",
                        "count": len(invalid_sources),
                        "sources": invalid_sources[:20],
                    }
                )
            for label, values in (
                ("Quarantined documents", summary["quarantine"]),
                ("Pending index changes", summary["pending_sources"] + summary["deleted_sources"]),
                ("Duplicate document groups", duplicate_groups),
                ("Stale index entries", stale_sources),
            ):
                if values:
                    issues.append(
                        {
                            "severity": "attention",
                            "label": label,
                            "count": len(values),
                        }
                    )

            status = (
                "critical"
                if any(item["severity"] == "critical" for item in issues)
                else "attention"
                if issues
                else "healthy"
            )
            result = {
                "status": status,
                "generated_at": utc_now(),
                "documents": len(documents),
                "pages": sum(int(item.get("pages") or 0) for item in documents),
                "chunks": sum(int(item.get("chunks") or 0) for item in documents),
                "indexed": summary["counts"]["indexed"],
                "pending": summary["counts"]["pending"],
                "quarantined": len(summary["quarantine"]),
                "revisions": len(summary["revisions"]),
                "invalid_sources": invalid_sources,
                "stale_sources": stale_sources,
                "duplicate_groups": duplicate_groups,
                "issues": issues,
                "storage": storage,
            }
            self._health_cache = (monotonic(), result)
            return result

    def _backup_path(self, backup_id: str) -> Path:
        if not backup_id or any(character not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_" for character in backup_id):
            raise ValueError("Invalid backup identifier.")
        path = (self.backup_dir / f"{backup_id}.zip").resolve()
        if not path.is_relative_to(self.backup_dir):
            raise ValueError("Invalid backup identifier.")
        return path

    @staticmethod
    def _zip_tree(archive: zipfile.ZipFile, source: Path, prefix: str) -> None:
        archive.writestr(prefix.rstrip("/") + "/", b"")
        if not source.exists():
            return
        for child in sorted(source.rglob("*")):
            if child.is_file():
                relative = child.relative_to(source).as_posix()
                archive.write(child, f"{prefix.rstrip('/')}/{relative}")

    def create_backup(self, label: str = "") -> dict[str, Any]:
        with self._lock:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_id = (
                datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + token_hex(3)
            )
            destination = self._backup_path(backup_id)
            temporary = destination.with_suffix(".zip.part")
            workspace_snapshot: Path | None = None
            try:
                if self.workspace_db.is_file():
                    descriptor, snapshot_name = tempfile.mkstemp(
                        prefix="workspace-",
                        suffix=".sqlite3",
                        dir=str(self.backup_dir),
                    )
                    os.close(descriptor)
                    workspace_snapshot = Path(snapshot_name)
                    source_connection = sqlite3.connect(self.workspace_db)
                    target_connection = sqlite3.connect(workspace_snapshot)
                    try:
                        source_connection.backup(target_connection)
                    finally:
                        target_connection.close()
                        source_connection.close()

                metadata = {
                    "version": BACKUP_VERSION,
                    "backup_id": backup_id,
                    "label": str(label or "").strip()[:120],
                    "created_at": utc_now(),
                    "collection": "local-reference-corpus",
                }
                with zipfile.ZipFile(
                    temporary,
                    "w",
                    compression=zipfile.ZIP_STORED,
                    allowZip64=True,
                ) as archive:
                    archive.writestr(
                        "corpus-backup.json",
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                    )
                    self._zip_tree(archive, self.repository.pdf_dir, "documents")
                    self._zip_tree(archive, self.db_dir, "index")
                    self._zip_tree(archive, self.repository.trash_dir, "trash")
                    self._zip_tree(
                        archive,
                        self.repository.quarantine_dir,
                        "quarantine",
                    )
                    self._zip_tree(
                        archive,
                        self.repository.revision_dir,
                        "revisions",
                    )
                    archive.writestr("workspace/", b"")
                    if workspace_snapshot is not None:
                        archive.write(
                            workspace_snapshot,
                            "workspace/reference_workspace.sqlite3",
                        )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
                if workspace_snapshot is not None:
                    workspace_snapshot.unlink(missing_ok=True)
            self.invalidate_health()
            return {
                **metadata,
                "size": destination.stat().st_size,
                "filename": destination.name,
            }

    def list_backups(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.backup_dir.exists():
                return []
            backups: list[dict[str, Any]] = []
            for path in sorted(self.backup_dir.glob("*.zip"), reverse=True):
                try:
                    with zipfile.ZipFile(path) as archive:
                        metadata = json.loads(
                            archive.read("corpus-backup.json").decode("utf-8")
                        )
                except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
                    continue
                backups.append(
                    {
                        **metadata,
                        "backup_id": path.stem,
                        "filename": path.name,
                        "size": path.stat().st_size,
                    }
                )
            return backups

    def delete_backup(self, backup_id: str) -> bool:
        with self._lock:
            path = self._backup_path(backup_id)
            if not path.is_file():
                return False
            path.unlink()
            self.invalidate_health()
            return True

    @staticmethod
    def _validate_archive(archive: zipfile.ZipFile) -> None:
        for name in archive.namelist():
            candidate = PurePosixPath(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("The backup contains an unsafe path.")

    @staticmethod
    def _replace_path(
        source: Path,
        target: Path,
        rollback_root: Path,
    ) -> tuple[Path, Path | None]:
        target = target.resolve()
        rollback: Path | None = rollback_root / token_hex(6) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        if rollback is not None:
            os.replace(target, rollback)
        try:
            if source.is_dir():
                shutil.move(str(source), str(target))
            elif source.is_file():
                shutil.copy2(source, target)
            else:
                target.mkdir(parents=True, exist_ok=True)
        except Exception:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if rollback is not None and rollback.exists():
                os.replace(rollback, target)
            raise
        return target, rollback

    @staticmethod
    def _restore_sqlite(
        source: Path,
        target: Path,
        rollback_root: Path,
    ) -> tuple[Path, Path | None]:
        """Restore SQLite through its online backup API, including on Windows."""

        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        rollback: Path | None = None
        if target.is_file():
            rollback = rollback_root / (token_hex(6) + ".sqlite3")
            current_connection = sqlite3.connect(target)
            rollback_connection = sqlite3.connect(rollback)
            try:
                current_connection.backup(rollback_connection)
            finally:
                rollback_connection.close()
                current_connection.close()
        source_connection = sqlite3.connect(source)
        target_connection = sqlite3.connect(target)
        try:
            source_connection.backup(target_connection)
        finally:
            target_connection.close()
            source_connection.close()
        return target, rollback

    def restore_backup(self, backup_id: str) -> dict[str, Any]:
        with self._lock:
            source_backup = self._backup_path(backup_id)
            if not source_backup.is_file():
                raise FileNotFoundError(backup_id)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="corpus-restore-",
                dir=str(self.backup_dir),
            ) as temporary_name:
                staging = Path(temporary_name)
                with zipfile.ZipFile(source_backup) as archive:
                    self._validate_archive(archive)
                    metadata = json.loads(
                        archive.read("corpus-backup.json").decode("utf-8")
                    )
                    if int(metadata.get("version") or 0) != BACKUP_VERSION:
                        raise ValueError("This backup version is not supported.")
                    archive.extractall(staging)

                rollback_root = staging / ".rollback"
                rollback_root.mkdir()
                replaced: list[tuple[Path, Path | None]] = []
                try:
                    for source, target in (
                        (staging / "documents", self.repository.pdf_dir),
                        (staging / "index", self.db_dir),
                        (staging / "trash", self.repository.trash_dir),
                        (staging / "quarantine", self.repository.quarantine_dir),
                        (staging / "revisions", self.repository.revision_dir),
                    ):
                        replaced.append(self._replace_path(source, target, rollback_root))
                    workspace = staging / "workspace" / "reference_workspace.sqlite3"
                    if workspace.is_file():
                        replaced.append(
                            self._restore_sqlite(
                                workspace,
                                self.workspace_db,
                                rollback_root,
                            )
                        )
                except Exception:
                    for target, rollback in reversed(replaced):
                        if target.exists():
                            if target.is_dir():
                                shutil.rmtree(target)
                            else:
                                target.unlink()
                        if rollback is not None and rollback.exists():
                            os.replace(rollback, target)
                    raise

            self._health_cache = None
            self._hash_cache.clear()
            self.repository._page_count_cache.clear()
            self.repository._file_hash_cache.clear()
            self.recover_interrupted()
            return {
                "restored": True,
                "backup_id": backup_id,
                "created_at": metadata.get("created_at"),
            }
