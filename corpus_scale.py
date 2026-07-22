"""Persistent corpus orchestration, health reporting, and portable snapshots."""

from __future__ import annotations

import json
import os
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
    ) -> None:
        self.repository = repository
        self.state_path = state_path.resolve()
        self.db_dir = db_dir.resolve()
        self.workspace_db = workspace_db.resolve()
        self.backup_dir = backup_dir.resolve()
        self.debug_dir = debug_dir.resolve() if debug_dir else None
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

    def invalidate_health(self) -> None:
        self._health_cache = None

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
