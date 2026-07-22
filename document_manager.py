"""Safe local document-library operations and its self-contained web UI."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from secrets import token_hex
from threading import RLock
from typing import Any

import pypdfium2 as pdfium


_INVALID_SEGMENT = re.compile(r'[<>:"|?*\x00-\x1f]')
_TRASH_ID = re.compile(r"^[0-9A-Za-z_-]{8,80}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_PENDING_LIFECYCLE_STATES = {"uploaded", "pending", "processing", "failed"}
_DOCUMENT_LIFECYCLE_STATES = {
    "uploaded",
    "pending",
    "processing",
    "indexed",
    "failed",
    "quarantined",
    "deleted",
}


class DocumentPathError(ValueError):
    """Raised when a requested library path is unsafe or unsupported."""


class DuplicateDocumentError(FileExistsError):
    """Raised when an upload already exists elsewhere in the corpus."""

    def __init__(self, duplicate_of: str) -> None:
        self.duplicate_of = duplicate_of
        super().__init__(f"This PDF is already in the library as {duplicate_of}.")


class DocumentRepository:
    """Manage PDFs, pending index changes, and a recoverable local trash."""

    def __init__(
        self,
        pdf_dir: Path,
        manifest_path: Path,
        state_path: Path,
        trash_dir: Path,
        quarantine_dir: Path | None = None,
        revision_dir: Path | None = None,
    ) -> None:
        self.pdf_dir = pdf_dir.resolve()
        self.manifest_path = manifest_path.resolve()
        self.state_path = state_path.resolve()
        self.trash_dir = trash_dir.resolve()
        self.quarantine_dir = (
            quarantine_dir or self.pdf_dir.parent / "document_quarantine"
        ).resolve()
        self.revision_dir = (
            revision_dir or self.pdf_dir.parent / "document_revisions"
        ).resolve()
        self._lock = RLock()
        self._page_count_cache: dict[str, tuple[int, int, int | None]] = {}
        self._file_hash_cache: dict[str, tuple[int, int, str]] = {}

    @staticmethod
    def normalize_source_id(value: str) -> str:
        raw = str(value or "").strip().replace("\\", "/")
        if not raw or len(raw) > 500:
            raise DocumentPathError("Enter a PDF name or relative path.")
        candidate = PurePosixPath(raw)
        if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            raise DocumentPathError("The document path must stay inside the library.")
        if candidate.suffix.lower() != ".pdf":
            raise DocumentPathError("Only PDF files can be added to this library.")
        for part in candidate.parts:
            if _INVALID_SEGMENT.search(part) or part.endswith((" ", ".")):
                raise DocumentPathError(f"Unsupported character in path segment: {part!r}")
            if Path(part).stem.upper() in _WINDOWS_RESERVED:
                raise DocumentPathError(f"Reserved file name: {part!r}")
        return candidate.as_posix()

    def resolve_source(self, source_id: str) -> Path:
        normalized = self.normalize_source_id(source_id)
        resolved = (self.pdf_dir / Path(*PurePosixPath(normalized).parts)).resolve()
        if not resolved.is_relative_to(self.pdf_dir):
            raise DocumentPathError("The document path must stay inside the library.")
        return resolved

    @staticmethod
    def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(default)
        return value if isinstance(value, dict) else dict(default)

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _state(self) -> dict[str, Any]:
        raw = self._read_json(
            self.state_path,
            {"pending_sources": [], "deleted_sources": []},
        )
        legacy_pending = {
            str(item) for item in raw.get("pending_sources", []) if item
        }
        legacy_deleted = {
            str(item) for item in raw.get("deleted_sources", []) if item
        }
        state: dict[str, Any] = {
            "version": 2,
            "documents": {
                str(source): dict(entry)
                for source, entry in (raw.get("documents") or {}).items()
                if source and isinstance(entry, dict)
            },
        }
        migrated = int(raw.get("version") or 1) < 2 or "documents" not in raw
        if migrated:
            manifest = self._read_json(self.manifest_path, {"sources": {}})
            manifest_sources = manifest.get("sources") or {}
            sources = set(legacy_pending) | set(legacy_deleted)
            if self.pdf_dir.exists():
                sources.update(
                    path.resolve().relative_to(self.pdf_dir).as_posix()
                    for path in self.pdf_dir.rglob("*.pdf")
                    if path.is_file()
                )
            now = datetime.now(tz=UTC).isoformat()
            for source_id in sorted(sources):
                manifest_entry = manifest_sources.get(source_id) or {}
                if source_id in legacy_deleted:
                    status = "deleted"
                elif source_id in legacy_pending:
                    status = "pending"
                elif manifest_entry.get("complete") is True:
                    status = "indexed"
                else:
                    status = "pending"
                state["documents"].setdefault(
                    source_id,
                    {
                        "status": status,
                        "indexed_file_hash": str(manifest_entry.get("file_hash") or ""),
                        "current_file_hash": "",
                        "error": "",
                        "index_removal_pending": source_id in legacy_deleted,
                        "updated_at": now,
                        "history": [
                            {
                                "from": None,
                                "to": status,
                                "at": now,
                                "reason": "Migrated legacy document state",
                            }
                        ],
                    },
                )
        for source_id, entry in state["documents"].items():
            status = str(entry.get("status") or "pending")
            entry["status"] = status if status in _DOCUMENT_LIFECYCLE_STATES else "pending"
            entry["indexed_file_hash"] = str(entry.get("indexed_file_hash") or "")
            entry["current_file_hash"] = str(entry.get("current_file_hash") or "")
            entry["error"] = str(entry.get("error") or "")
            entry["index_removal_pending"] = bool(entry.get("index_removal_pending"))
            entry["updated_at"] = str(entry.get("updated_at") or "")
            entry["history"] = [
                dict(event)
                for event in entry.get("history", [])
                if isinstance(event, dict)
            ][-100:]
        self._sync_legacy_views(state)
        if migrated:
            backup = self.state_path.with_name(
                f"{self.state_path.name}.pre-v2-migration.bak"
            )
            if self.state_path.exists() and not backup.exists():
                shutil.copy2(self.state_path, backup)
            self._write_json(self.state_path, state)
        return state

    @staticmethod
    def _sync_legacy_views(state: dict[str, Any]) -> None:
        documents = state.get("documents") or {}
        state["pending_sources"] = sorted(
            source_id
            for source_id, entry in documents.items()
            if str(entry.get("status")) in _PENDING_LIFECYCLE_STATES
        )
        state["deleted_sources"] = sorted(
            source_id
            for source_id, entry in documents.items()
            if bool(entry.get("index_removal_pending"))
        )

    @staticmethod
    def _transition_in_state(
        state: dict[str, Any],
        source_id: str,
        status: str,
        *,
        reason: str = "",
        file_hash: str = "",
        error: str = "",
        index_removal_pending: bool | None = None,
    ) -> dict[str, Any]:
        if status not in _DOCUMENT_LIFECYCLE_STATES:
            raise ValueError(f"Unsupported document lifecycle state: {status}")
        now = datetime.now(tz=UTC).isoformat()
        documents = state.setdefault("documents", {})
        entry = documents.setdefault(
            source_id,
            {
                "status": "uploaded",
                "indexed_file_hash": "",
                "current_file_hash": "",
                "error": "",
                "index_removal_pending": False,
                "updated_at": now,
                "history": [],
            },
        )
        previous = str(entry.get("status") or "uploaded")
        if previous != status or reason or error:
            entry.setdefault("history", []).append(
                {
                    "from": previous if entry.get("history") else None,
                    "to": status,
                    "at": now,
                    "reason": str(reason or "")[:500],
                    "file_hash": str(file_hash or ""),
                    "error": str(error or "")[:2000],
                }
            )
            entry["history"] = entry["history"][-100:]
        entry["status"] = status
        entry["updated_at"] = now
        entry["error"] = str(error or "")[:2000]
        if file_hash:
            entry["current_file_hash"] = str(file_hash)
            if status == "indexed":
                entry["indexed_file_hash"] = str(file_hash)
        if index_removal_pending is not None:
            entry["index_removal_pending"] = bool(index_removal_pending)
        DocumentRepository._sync_legacy_views(state)
        return entry

    def transition(
        self,
        source_id: str,
        status: str,
        *,
        reason: str = "",
        file_hash: str = "",
        error: str = "",
        index_removal_pending: bool | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_source_id(source_id)
        with self._lock:
            state = self._state()
            entry = self._transition_in_state(
                state,
                normalized,
                status,
                reason=reason,
                file_hash=file_hash,
                error=error,
                index_removal_pending=index_removal_pending,
            )
            self._write_json(self.state_path, state)
            return dict(entry)

    def _mark_changes(
        self,
        *,
        pending: tuple[str, ...] = (),
        deleted: tuple[str, ...] = (),
        remove_deleted: tuple[str, ...] = (),
    ) -> None:
        state = self._state()
        for source_id in pending:
            existing = state["documents"].get(source_id)
            migrated_placeholder = bool(
                existing
                and len(existing.get("history") or []) == 1
                and str((existing.get("history") or [{}])[0].get("reason") or "")
                == "Migrated legacy document state"
                and existing.get("status") == "pending"
            )
            if source_id not in state["documents"] or migrated_placeholder:
                if migrated_placeholder:
                    state["documents"].pop(source_id, None)
                self._transition_in_state(
                    state,
                    source_id,
                    "uploaded",
                    reason="Document added to the library",
                )
            self._transition_in_state(
                state,
                source_id,
                "pending",
                reason="Waiting for indexing",
                index_removal_pending=False,
            )
        for source_id in deleted:
            self._transition_in_state(
                state,
                source_id,
                "deleted",
                reason="Document removed from the active library",
                index_removal_pending=True,
            )
        for source_id in remove_deleted:
            entry = state["documents"].get(source_id)
            if entry is not None:
                entry["index_removal_pending"] = False
        self._sync_legacy_views(state)
        self._write_json(self.state_path, state)

    def _page_count(self, path: Path) -> int | None:
        stat = path.stat()
        cache_key = str(path)
        cached = self._page_count_cache.get(cache_key)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        try:
            document = pdfium.PdfDocument(str(path))
            try:
                count: int | None = len(document)
            finally:
                document.close()
        except Exception:
            count = None
        self._page_count_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, count)
        return count

    def _file_hash(self, path: Path) -> str:
        stat = path.stat()
        cache_key = str(path.resolve())
        cached = self._file_hash_cache.get(cache_key)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        self._file_hash_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, value)
        return value

    def _existing_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path in sorted(self.pdf_dir.rglob("*.pdf")):
            if not path.is_file():
                continue
            source_id = path.resolve().relative_to(self.pdf_dir).as_posix()
            hashes[source_id] = self._file_hash(path)
        return hashes

    def _archive_revision(
        self,
        source_id: str,
        source_path: Path,
        *,
        replaced_by_hash: str,
    ) -> dict[str, Any]:
        revision_id = (
            datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S") + "-" + token_hex(4)
        )
        entry_dir = self.revision_dir / revision_id
        entry_dir.mkdir(parents=True, exist_ok=False)
        archived = entry_dir / "document.pdf"
        shutil.copy2(source_path, archived)
        record = {
            "revision_id": revision_id,
            "source_id": source_id,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "file_hash": self._file_hash(archived),
            "replaced_by_hash": replaced_by_hash,
            "size": archived.stat().st_size,
        }
        self._write_json(entry_dir / "record.json", record)
        return record

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            self.pdf_dir.mkdir(parents=True, exist_ok=True)
            manifest = self._read_json(self.manifest_path, {"sources": {}})
            manifest_sources = manifest.get("sources") or {}
            state = self._state()
            state_changed = False
            revision_counts: dict[str, int] = {}
            for revision in self.list_revisions():
                source = str(revision.get("source_id") or "")
                revision_counts[source] = revision_counts.get(source, 0) + 1
            documents: list[dict[str, Any]] = []
            for path in sorted(self.pdf_dir.rglob("*.pdf")):
                if not path.is_file():
                    continue
                source_id = path.resolve().relative_to(self.pdf_dir).as_posix()
                manifest_entry = manifest_sources.get(source_id) or {}
                lifecycle = state["documents"].get(source_id)
                if lifecycle is None:
                    self._transition_in_state(
                        state,
                        source_id,
                        "uploaded",
                        reason="Document discovered in the library",
                    )
                    lifecycle = self._transition_in_state(
                        state,
                        source_id,
                        "pending",
                        reason="Waiting for indexing",
                    )
                    state_changed = True
                stat = path.stat()
                current_hash = ""
                if (
                    lifecycle.get("status") == "pending"
                    and manifest_entry.get("complete") is True
                    and manifest_entry.get("file_hash")
                ):
                    current_hash = self._file_hash(path)
                    if str(manifest_entry["file_hash"]) == current_hash:
                        lifecycle = self._transition_in_state(
                            state,
                            source_id,
                            "indexed",
                            reason="Reconciled completed manifest",
                            file_hash=current_hash,
                        )
                        state_changed = True
                indexed_hash = str(
                    lifecycle.get("indexed_file_hash")
                    or manifest_entry.get("file_hash")
                    or ""
                )
                if lifecycle.get("status") == "indexed" and indexed_hash:
                    current_hash = current_hash or self._file_hash(path)
                    if current_hash != indexed_hash:
                        lifecycle = self._transition_in_state(
                            state,
                            source_id,
                            "pending",
                            reason="File content changed after indexing",
                            file_hash=current_hash,
                        )
                        state_changed = True
                status = str(lifecycle.get("status") or "pending")
                parent = PurePosixPath(source_id).parent.as_posix()
                documents.append(
                    {
                        "source_id": source_id,
                        "name": path.name,
                        "folder": "" if parent == "." else parent,
                        "size": stat.st_size,
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime,
                            tz=UTC,
                        ).isoformat(),
                        "pages": self._page_count(path),
                        "chunks": int(manifest_entry.get("chunk_count") or 0),
                        "status": status,
                        "file_hash": indexed_hash,
                        "indexed_file_hash": indexed_hash,
                        "state_updated_at": str(lifecycle.get("updated_at") or ""),
                        "state_error": str(lifecycle.get("error") or ""),
                        "state_history": list(lifecycle.get("history") or []),
                        "revision_count": revision_counts.get(source_id, 0),
                    }
                )
            if state_changed:
                self._sync_legacy_views(state)
                self._write_json(self.state_path, state)
            return documents

    def list_trash(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.trash_dir.exists():
                return []
            entries: list[dict[str, Any]] = []
            for record_path in sorted(
                self.trash_dir.glob("*/record.json"),
                reverse=True,
            ):
                record = self._read_json(record_path, {})
                source_id = str(record.get("source_id") or "")
                pdf_path = record_path.parent / "document.pdf"
                if not source_id or not pdf_path.is_file():
                    continue
                entries.append(
                    {
                        "trash_id": record_path.parent.name,
                        "source_id": source_id,
                        "deleted_at": str(record.get("deleted_at") or ""),
                        "size": pdf_path.stat().st_size,
                    }
                )
            return entries

    def _aux_entry(self, root: Path, entry_id: str) -> Path:
        if not _TRASH_ID.fullmatch(str(entry_id or "")):
            raise DocumentPathError("Invalid corpus entry.")
        entry = (root / entry_id).resolve()
        if not entry.is_relative_to(root):
            raise DocumentPathError("Invalid corpus entry.")
        return entry

    def list_quarantine(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.quarantine_dir.exists():
                return []
            entries: list[dict[str, Any]] = []
            for record_path in sorted(
                self.quarantine_dir.glob("*/record.json"),
                reverse=True,
            ):
                record = self._read_json(record_path, {})
                pdf_path = record_path.parent / "document.pdf"
                if not record.get("source_id") or not pdf_path.is_file():
                    continue
                entries.append(
                    {
                        **record,
                        "quarantine_id": record_path.parent.name,
                        "size": pdf_path.stat().st_size,
                    }
                )
            return entries

    def quarantine(self, source_id: str, error: str) -> dict[str, Any]:
        normalized = self.normalize_source_id(source_id)
        source = self.resolve_source(normalized)
        with self._lock:
            if not source.is_file():
                raise FileNotFoundError(normalized)
            self.transition(
                normalized,
                "failed",
                reason="Ingestion failed",
                error=str(error or "Unknown ingestion error"),
            )
            quarantine_id = (
                datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
                + "-"
                + token_hex(4)
            )
            entry_dir = self.quarantine_dir / quarantine_id
            entry_dir.mkdir(parents=True, exist_ok=False)
            destination = entry_dir / "document.pdf"
            os.replace(source, destination)
            record = {
                "quarantine_id": quarantine_id,
                "source_id": normalized,
                "quarantined_at": datetime.now(tz=UTC).isoformat(),
                "error": re.sub(r"\s+", " ", str(error or "Unknown ingestion error"))[:2000],
                "file_hash": self._file_hash(destination),
            }
            self._write_json(entry_dir / "record.json", record)
            self._remove_empty_parents(source.parent)
            self.transition(
                normalized,
                "quarantined",
                reason="Ingestion failed; document moved to quarantine",
                file_hash=str(record["file_hash"]),
                error=str(record["error"]),
                index_removal_pending=True,
            )
            return {**record, "size": destination.stat().st_size}

    def restore_quarantine(self, quarantine_id: str) -> dict[str, str]:
        with self._lock:
            entry = self._aux_entry(self.quarantine_dir, quarantine_id)
            record = self._read_json(entry / "record.json", {})
            normalized = self.normalize_source_id(str(record.get("source_id") or ""))
            source = entry / "document.pdf"
            target = self.resolve_source(normalized)
            if not source.is_file():
                raise FileNotFoundError(quarantine_id)
            if target.exists():
                raise FileExistsError("Restore destination already exists.")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            shutil.rmtree(entry, ignore_errors=True)
            self._mark_changes(pending=(normalized,), remove_deleted=(normalized,))
            return {"source_id": normalized}

    def delete_quarantine(self, quarantine_id: str) -> bool:
        with self._lock:
            entry = self._aux_entry(self.quarantine_dir, quarantine_id)
            if not entry.is_dir():
                return False
            shutil.rmtree(entry)
            return True

    def list_revisions(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.revision_dir.exists():
                return []
            entries: list[dict[str, Any]] = []
            for record_path in sorted(
                self.revision_dir.glob("*/record.json"),
                reverse=True,
            ):
                record = self._read_json(record_path, {})
                archived = record_path.parent / "document.pdf"
                if not record.get("source_id") or not archived.is_file():
                    continue
                entries.append(
                    {
                        **record,
                        "revision_id": record_path.parent.name,
                        "size": archived.stat().st_size,
                    }
                )
            return entries

    def restore_revision(self, revision_id: str) -> dict[str, str]:
        with self._lock:
            entry = self._aux_entry(self.revision_dir, revision_id)
            record = self._read_json(entry / "record.json", {})
            normalized = self.normalize_source_id(str(record.get("source_id") or ""))
            archived = entry / "document.pdf"
            target = self.resolve_source(normalized)
            if not archived.is_file():
                raise FileNotFoundError(revision_id)
            if target.is_file():
                self._archive_revision(
                    normalized,
                    target,
                    replaced_by_hash=str(record.get("file_hash") or ""),
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".revision-restore")
            shutil.copy2(archived, temporary)
            os.replace(temporary, target)
            self._mark_changes(pending=(normalized,), remove_deleted=(normalized,))
            return {"source_id": normalized, "revision_id": revision_id}

    def summary(self) -> dict[str, Any]:
        documents = self.list_documents()
        state = self._state()
        trash = self.list_trash()
        quarantine = self.list_quarantine()
        revisions = self.list_revisions()
        return {
            "documents": documents,
            "trash": trash,
            "quarantine": quarantine,
            "revisions": revisions,
            "counts": {
                "documents": len(documents),
                "indexed": sum(item["status"] == "indexed" for item in documents),
                "pending": len(state["pending_sources"])
                + len(state["deleted_sources"]),
                "trash": len(trash),
                "quarantine": len(quarantine),
                "revisions": len(revisions),
            },
            "pending_sources": state["pending_sources"],
            "deleted_sources": state["deleted_sources"],
        }

    def commit_upload(
        self,
        temporary_path: Path,
        source_id: str,
        *,
        replace: bool = False,
    ) -> dict[str, Any]:
        normalized = self.normalize_source_id(source_id)
        target = self.resolve_source(normalized)
        with self._lock:
            if not temporary_path.is_file() or temporary_path.stat().st_size < 5:
                raise ValueError("The uploaded PDF is empty.")
            with temporary_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise ValueError("The uploaded file is not a valid PDF.")
            incoming_hash = self._file_hash(temporary_path)
            existing_hashes = self._existing_hashes()
            for existing_source, existing_hash in existing_hashes.items():
                if existing_hash == incoming_hash:
                    raise DuplicateDocumentError(existing_source)
            if target.exists() and not replace:
                raise FileExistsError(
                    "A document already exists at this path. Enable Replace to overwrite it."
                )
            revision: dict[str, Any] | None = None
            if target.exists():
                revision = self._archive_revision(
                    normalized,
                    target,
                    replaced_by_hash=incoming_hash,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_path, target)
            self._mark_changes(pending=(normalized,), remove_deleted=(normalized,))
            return {
                "source_id": normalized,
                "size": target.stat().st_size,
                "file_hash": incoming_hash,
                "revision_id": revision.get("revision_id") if revision else None,
            }

    def move(self, source_id: str, target_source_id: str) -> dict[str, str]:
        source_normalized = self.normalize_source_id(source_id)
        target_normalized = self.normalize_source_id(target_source_id)
        source = self.resolve_source(source_normalized)
        target = self.resolve_source(target_normalized)
        with self._lock:
            if not source.is_file():
                raise FileNotFoundError(source_normalized)
            if target.exists():
                raise FileExistsError("A document already exists at the destination.")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            self._remove_empty_parents(source.parent)
            self._mark_changes(
                pending=(target_normalized,),
                deleted=(source_normalized,),
                remove_deleted=(target_normalized,),
            )
            return {"source_id": target_normalized, "previous_source_id": source_normalized}

    def trash(self, source_id: str) -> dict[str, str]:
        normalized = self.normalize_source_id(source_id)
        source = self.resolve_source(normalized)
        with self._lock:
            if not source.is_file():
                raise FileNotFoundError(normalized)
            trash_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S") + "-" + token_hex(4)
            entry_dir = self.trash_dir / trash_id
            entry_dir.mkdir(parents=True, exist_ok=False)
            destination = entry_dir / "document.pdf"
            os.replace(source, destination)
            self._write_json(
                entry_dir / "record.json",
                {
                    "source_id": normalized,
                    "deleted_at": datetime.now(tz=UTC).isoformat(),
                },
            )
            self._remove_empty_parents(source.parent)
            self._mark_changes(deleted=(normalized,))
            return {"trash_id": trash_id, "source_id": normalized}

    def _trash_entry(self, trash_id: str) -> Path:
        if not _TRASH_ID.fullmatch(str(trash_id or "")):
            raise DocumentPathError("Invalid trash entry.")
        entry = (self.trash_dir / trash_id).resolve()
        if not entry.is_relative_to(self.trash_dir):
            raise DocumentPathError("Invalid trash entry.")
        return entry

    def restore(self, trash_id: str) -> dict[str, str]:
        with self._lock:
            entry = self._trash_entry(trash_id)
            record = self._read_json(entry / "record.json", {})
            normalized = self.normalize_source_id(str(record.get("source_id") or ""))
            source = entry / "document.pdf"
            target = self.resolve_source(normalized)
            if not source.is_file():
                raise FileNotFoundError(trash_id)
            if target.exists():
                raise FileExistsError("Restore destination already exists.")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            shutil.rmtree(entry, ignore_errors=True)
            self._mark_changes(pending=(normalized,), remove_deleted=(normalized,))
            return {"source_id": normalized}

    def delete_forever(self, trash_id: str) -> bool:
        with self._lock:
            entry = self._trash_entry(trash_id)
            if not entry.is_dir():
                return False
            shutil.rmtree(entry)
            return True

    def clear_pending(self) -> None:
        with self._lock:
            state = self._state()
            manifest = self._read_json(self.manifest_path, {"sources": {}})
            manifest_sources = manifest.get("sources") or {}
            for source_id, entry in list(state["documents"].items()):
                if str(entry.get("status")) in _PENDING_LIFECYCLE_STATES:
                    manifest_entry = manifest_sources.get(source_id) or {}
                    self._transition_in_state(
                        state,
                        source_id,
                        "indexed",
                        reason="Index update completed",
                        file_hash=str(manifest_entry.get("file_hash") or ""),
                    )
                entry["index_removal_pending"] = False
            self._sync_legacy_views(state)
            self._write_json(self.state_path, state)

    def clear_pending_sources(
        self,
        sources: tuple[str, ...] = (),
        deleted: tuple[str, ...] = (),
    ) -> None:
        with self._lock:
            state = self._state()
            manifest = self._read_json(self.manifest_path, {"sources": {}})
            manifest_sources = manifest.get("sources") or {}
            for source_id in sources:
                manifest_entry = manifest_sources.get(source_id) or {}
                self._transition_in_state(
                    state,
                    source_id,
                    "indexed",
                    reason="Index update completed",
                    file_hash=str(manifest_entry.get("file_hash") or ""),
                )
            for source_id in deleted:
                entry = state["documents"].get(source_id)
                if entry is not None:
                    entry["index_removal_pending"] = False
            self._sync_legacy_views(state)
            self._write_json(self.state_path, state)

    def _remove_empty_parents(self, start: Path) -> None:
        current = start
        while current != self.pdf_dir and current.is_relative_to(self.pdf_dir):
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent


def document_manager_html() -> str:
    """Return the standalone local document-library interface."""

    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document library</title>
  <style>
    :root { color-scheme:dark; --bg:#0f1117; --panel:#171a22; --soft:#20242e; --line:#343a48; --ink:#f3f5f8; --muted:#aab2c0; --blue:#3b82f6; --blue2:#2563eb; --green:#8de4b7; --amber:#f4c35b; --danger:#ff9d9d; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    button,input,a { font:inherit; }
    button,input { border:1px solid var(--line); border-radius:9px; background:#252a35; color:var(--ink); }
    button { min-height:40px; padding:8px 13px; cursor:pointer; }
    button:hover:not(:disabled),.button:hover { background:#333a48; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    input { min-height:42px; padding:8px 11px; }
    .topbar { position:sticky; top:0; z-index:20; display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 24px; background:rgba(15,17,23,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }
    .topbar strong { font-size:1.08rem; }
    .nav,.actions,.row-actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .button { display:inline-flex; align-items:center; justify-content:center; min-height:40px; padding:8px 13px; border:1px solid var(--line); border-radius:9px; background:#252a35; color:var(--ink); text-decoration:none; }
    .primary { background:var(--blue2); border-color:#60a5fa; color:white; }
    button.primary:hover,.button.primary:hover { background:var(--blue); }
    .danger { color:var(--danger); }
    .shell { width:min(1500px,calc(100% - 34px)); margin:0 auto; padding:28px 0 54px; }
    .hero { display:flex; justify-content:space-between; align-items:flex-start; gap:24px; margin-bottom:22px; }
    h1 { margin:0; font-size:clamp(1.65rem,2.5vw,2.2rem); }
    h2 { margin:0; font-size:1.02rem; }
    .muted { color:var(--muted); }
    .stats { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:10px; margin:0 0 18px; }
    .stat { padding:14px 16px; border:1px solid var(--line); border-radius:12px; background:var(--panel); }
    .stat strong { display:block; margin-top:3px; font-size:1.45rem; }
    .stat span { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.07em; }
    .panel { margin-bottom:16px; padding:18px; border:1px solid var(--line); border-radius:14px; background:var(--panel); }
    .panel-head { display:flex; justify-content:space-between; align-items:center; gap:14px; margin-bottom:14px; }
    .upload-grid { display:grid; grid-template-columns:minmax(250px,1fr) minmax(180px,320px) auto; gap:10px; align-items:end; }
    .field { display:grid; gap:6px; }
    .field label { color:var(--muted); font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
    .drop { padding:17px; border:1px dashed #536176; border-radius:11px; background:#1c2029; }
    .drop.drag { border-color:#60a5fa; background:#1d2b45; }
    .drop input { width:100%; padding:7px; }
    .replace { display:flex; align-items:center; gap:8px; min-height:42px; color:var(--muted); }
    .replace input { min-height:0; width:17px; height:17px; }
    .job { display:none; margin-top:14px; padding:13px; border:1px solid #41506a; border-radius:10px; background:#1a2231; }
    .job.visible { display:block; }
    .job-head { display:flex; justify-content:space-between; gap:12px; }
    .progress { height:7px; margin:10px 0; overflow:hidden; border-radius:999px; background:#293241; }
    .progress span { display:block; width:35%; height:100%; background:#60a5fa; border-radius:inherit; animation:work 1.2s ease-in-out infinite alternate; }
    .job.done .progress span { width:100%; animation:none; background:#55c58a; }
    .job.failed .progress span { width:100%; animation:none; background:#dd6b72; }
    .job-actions { display:flex; justify-content:flex-end; margin-top:12px; }
    .job-actions button { min-width:150px; }
    @keyframes work { from{transform:translateX(-50%)} to{transform:translateX(220%)} }
    .log { max-height:150px; margin:8px 0 0; padding:10px; overflow:auto; border-radius:8px; background:#11141a; color:#c5ccda; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; white-space:pre-wrap; }
    .toolbar { display:grid; grid-template-columns:minmax(220px,1fr) auto; gap:10px; margin-bottom:12px; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:11px; }
    table { width:100%; border-collapse:collapse; min-width:920px; }
    th,td { padding:12px 13px; border-bottom:1px solid #2d3340; text-align:left; vertical-align:middle; }
    th { position:sticky; top:0; background:#20242d; color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
    tbody tr:last-child td { border-bottom:0; }
    tbody tr:hover { background:#1c2029; }
    .document-name { font-weight:700; overflow-wrap:anywhere; }
    .document-path { margin-top:3px; color:var(--muted); font-size:.73rem; overflow-wrap:anywhere; }
    .pill { display:inline-flex; padding:4px 8px; border-radius:999px; font-size:.72rem; font-weight:700; }
    .pill.indexed { background:#203b31; color:var(--green); }
    .pill.pending,.pill.not_indexed { background:#44371f; color:var(--amber); }
    .empty { padding:34px; color:var(--muted); text-align:center; }
    details { margin-top:16px; }
    summary { cursor:pointer; color:#d8deea; font-weight:700; }
    .trash-list { display:grid; gap:8px; margin-top:12px; }
    .trash-item { display:flex; justify-content:space-between; align-items:center; gap:14px; padding:11px 12px; border:1px solid #303643; border-radius:10px; background:#1c2029; }
    .status { min-height:1.2em; margin-top:9px; color:var(--green); font-size:.82rem; }
    @media (max-width:780px) { .topbar,.hero,.panel-head,.trash-item{align-items:stretch;flex-direction:column}.nav,.actions{width:100%}.button,.actions button{flex:1}.shell{width:min(100% - 20px,1500px);padding-top:18px}.stats{grid-template-columns:repeat(2,1fr)}.upload-grid,.toolbar{grid-template-columns:1fr} }
  </style>
</head>
<body>
  <header class="topbar">
    <strong>Document library</strong>
    <nav class="nav"><a class="button primary" href="/">Search documents</a><a class="button" href="/workspace">Research workspace</a><a class="button" href="/quality">Reference quality</a></nav>
  </header>
  <main class="shell">
    <section class="hero">
      <h1>Manage your source documents</h1>
      <div class="actions"><button id="syncButton" class="primary" type="button">Apply pending changes</button><button id="reindexButton" type="button">Reindex all</button></div>
    </section>
    <section class="stats">
      <div class="stat"><span>Documents</span><strong id="documentCount">—</strong></div>
      <div class="stat"><span>Indexed</span><strong id="indexedCount">—</strong></div>
      <div class="stat"><span>Pending changes</span><strong id="pendingCount">—</strong></div>
      <div class="stat"><span>In trash</span><strong id="trashCount">—</strong></div>
    </section>
    <section class="panel">
      <div class="panel-head"><h2>Add PDFs</h2></div>
      <div class="upload-grid">
        <div id="dropZone" class="drop field"><label for="fileInput">PDF files</label><input id="fileInput" type="file" accept="application/pdf,.pdf" multiple></div>
        <div class="field"><label for="folderInput">Destination folder (optional)</label><input id="folderInput" placeholder="e.g. standards/monorail"></div>
        <label class="replace"><input id="replaceInput" type="checkbox"> Replace matching files</label>
      </div>
      <div id="uploadStatus" class="status" aria-live="polite"></div>
      <section id="jobPanel" class="job" aria-live="polite"><div class="job-head"><strong id="jobTitle">Preparing index…</strong><span id="jobState" class="muted"></span></div><div class="progress"><span></span></div><div id="jobMessage" class="muted"></div><pre id="jobLog" class="log"></pre><div class="job-actions"><button id="restartButton" class="primary" type="button" hidden>Reload search models</button></div></section>
    </section>
    <section class="panel">
      <div class="panel-head"><h2>Library</h2></div>
      <div class="toolbar"><input id="filterInput" type="search" placeholder="Filter by file name, folder, or status"><button id="refreshButton" type="button">Refresh</button></div>
      <div class="table-wrap"><table><thead><tr><th>Document</th><th>Pages</th><th>Size</th><th>Chunks</th><th>Modified</th><th>Status</th><th>Actions</th></tr></thead><tbody id="documentRows"></tbody></table><div id="emptyLibrary" class="empty" hidden>No PDFs in the library yet.</div></div>
      <details><summary>Trash <span id="trashSummary"></span></summary><div id="trashList" class="trash-list"></div></details>
    </section>
  </main>
  <script>
    const byId = id => document.getElementById(id);
    const state = {documents:[],trash:[],job:null,poll:null};
    const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
    const humanSize = bytes => { const units=['B','KB','MB','GB']; let value=Number(bytes)||0,index=0; while(value>=1024&&index<units.length-1){value/=1024;index+=1} return `${value.toFixed(index?1:0)} ${units[index]}`; };
    const dateLabel = value => value ? new Date(value).toLocaleString() : '—';
    const api = async (url,options={}) => { const response=await fetch(url,options); const payload=await response.json().catch(()=>({})); if(!response.ok) throw new Error(payload.detail||'Request failed'); return payload; };

    function renderDocuments() {
      const query=byId('filterInput').value.trim().toLocaleLowerCase();
      const rows=byId('documentRows'); rows.replaceChildren();
      const visible=state.documents.filter(item=>!query||[item.source_id,item.status,item.folder].join(' ').toLocaleLowerCase().includes(query));
      byId('emptyLibrary').hidden=visible.length>0;
      visible.forEach(item=>{
        const row=document.createElement('tr');
        row.innerHTML=`<td><div class="document-name">${escapeHtml(item.name)}</div><div class="document-path">${escapeHtml(item.source_id)}</div></td><td>${item.pages??'—'}</td><td>${humanSize(item.size)}</td><td>${item.chunks||'—'}</td><td>${escapeHtml(dateLabel(item.modified_at))}</td><td><span class="pill ${escapeHtml(item.status)}">${item.status==='indexed'?'Indexed':item.status==='pending'?'Pending':'Not indexed'}</span></td><td><div class="row-actions"><button type="button" data-move="${escapeHtml(item.source_id)}">Move / rename</button><button class="danger" type="button" data-delete="${escapeHtml(item.source_id)}">Delete</button></div></td>`;
        rows.appendChild(row);
      });
      rows.querySelectorAll('[data-move]').forEach(button=>button.addEventListener('click',()=>moveDocument(button.dataset.move)));
      rows.querySelectorAll('[data-delete]').forEach(button=>button.addEventListener('click',()=>trashDocument(button.dataset.delete)));
    }

    function renderTrash() {
      byId('trashSummary').textContent=state.trash.length?`(${state.trash.length})`:'';
      const list=byId('trashList'); list.replaceChildren();
      if(!state.trash.length){ list.textContent='Trash is empty.'; list.className='trash-list muted'; return; }
      list.className='trash-list';
      state.trash.forEach(item=>{
        const row=document.createElement('div'); row.className='trash-item';
        row.innerHTML=`<div><div class="document-name">${escapeHtml(item.source_id)}</div><div class="document-path">Deleted ${escapeHtml(dateLabel(item.deleted_at))} · ${humanSize(item.size)}</div></div><div class="row-actions"><button type="button" data-restore="${escapeHtml(item.trash_id)}">Restore</button><button class="danger" type="button" data-purge="${escapeHtml(item.trash_id)}">Delete forever</button></div>`;
        list.appendChild(row);
      });
      list.querySelectorAll('[data-restore]').forEach(button=>button.addEventListener('click',()=>restoreDocument(button.dataset.restore)));
      list.querySelectorAll('[data-purge]').forEach(button=>button.addEventListener('click',()=>purgeDocument(button.dataset.purge)));
    }

    async function refresh() {
      const payload=await api('/documents/api/list');
      state.documents=payload.documents||[]; state.trash=payload.trash||[]; state.job=payload.job||null;
      byId('documentCount').textContent=payload.counts.documents;
      byId('indexedCount').textContent=payload.counts.indexed;
      byId('pendingCount').textContent=payload.counts.pending;
      byId('trashCount').textContent=payload.counts.trash;
      byId('syncButton').disabled=!payload.counts.pending||Boolean(payload.job?.running);
      byId('reindexButton').disabled=Boolean(payload.job?.running);
      renderDocuments(); renderTrash(); renderJob(payload.job);
    }

    async function uploadFiles(files) {
      const folder=byId('folderInput').value.trim().replaceAll('\\','/').replace(/^[/]+|[/]+$/g,'');
      const replace=byId('replaceInput').checked;
      for(let index=0;index<files.length;index+=1){
        const file=files[index]; const path=folder?`${folder}/${file.name}`:file.name;
        byId('uploadStatus').textContent=`Uploading ${index+1} of ${files.length}: ${file.name}`;
        await api(`/documents/api/upload?path=${encodeURIComponent(path)}&replace=${replace?'true':'false'}`,{method:'POST',headers:{'Content-Type':'application/pdf'},body:file});
      }
      byId('fileInput').value=''; byId('uploadStatus').textContent=`Added ${files.length} PDF${files.length===1?'':'s'}. Apply pending changes when ready.`; await refresh();
    }

    async function moveDocument(source) {
      const target=prompt('New relative path inside the library:',source); if(!target||target===source)return;
      try{await api(`/documents/api/item/${encodeURIComponent(source)}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({target})});await refresh();}catch(error){alert(error.message);}
    }
    async function trashDocument(source) {
      if(!confirm(`Move ${source} to the recoverable trash?`))return;
      try{await api(`/documents/api/item/${encodeURIComponent(source)}`,{method:'DELETE'});await refresh();}catch(error){alert(error.message);}
    }
    async function restoreDocument(id) { try{await api(`/documents/api/trash/${encodeURIComponent(id)}/restore`,{method:'POST'});await refresh();}catch(error){alert(error.message);} }
    async function purgeDocument(id) { if(!confirm('Permanently delete this PDF? This cannot be undone.'))return; try{await api(`/documents/api/trash/${encodeURIComponent(id)}`,{method:'DELETE'});await refresh();}catch(error){alert(error.message);} }

    function renderJob(job) {
      const panel=byId('jobPanel');
      const restartButton=byId('restartButton');
      if(!job||(!job.running&&!['complete','failed'].includes(job.state))){panel.className='job';restartButton.hidden=true;return;}
      panel.className=`job visible ${job.state==='complete'?'done':job.state==='failed'?'failed':''}`;
      byId('jobTitle').textContent=job.state==='complete'?'Index updated':job.state==='failed'?'Index update failed':'Updating document index';
      byId('jobState').textContent=job.state||''; byId('jobMessage').textContent=job.message||''; byId('jobLog').textContent=(job.log||[]).join('\n');
      restartButton.hidden=job.state!=='complete';
      if(!restartButton.hidden)restartButton.disabled=false;
      if(job.running&&!state.poll)state.poll=setInterval(async()=>{try{await refresh();if(!state.job?.running){clearInterval(state.poll);state.poll=null;}}catch(_){}},1200);
    }
    async function startSync(force=false) { if(force&&!confirm('Reindex every document? This can take a long time.'))return; try{const job=await api('/documents/api/sync',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({force})});state.job=job;renderJob(job);await refresh();}catch(error){alert(error.message);} }

    async function restartApp() {
      if(!confirm('Reload the search models now? The page will reconnect automatically.'))return;
      const button=byId('restartButton'); button.disabled=true; button.textContent='Reloading…';
      byId('jobMessage').textContent='Reloading search models. This page will reconnect automatically.';
      let restart;
      try{restart=await api('/documents/api/restart',{method:'POST'});}catch(error){button.disabled=false;button.textContent='Reload search models';alert(error.message);return;}
      const previousInstance=restart.instance_id; let attempts=0;
      const reconnect=async()=>{
        attempts+=1;
        try{
          const response=await fetch(`/documents/api/list?restart=${Date.now()}`,{cache:'no-store'});
          if(response.ok){const payload=await response.json();if(payload.app_instance_id&&payload.app_instance_id!==previousInstance){location.reload();return;}}
        }catch(_){}
        if(attempts>=160){button.disabled=false;button.textContent='Try reload again';byId('jobMessage').textContent='Search models did not reload automatically. Try again.';return;}
        setTimeout(reconnect,750);
      };
      setTimeout(reconnect,900);
    }

    byId('fileInput').addEventListener('change',event=>{const files=[...event.target.files];if(files.length)uploadFiles(files).catch(error=>{byId('uploadStatus').textContent=error.message;});});
    const drop=byId('dropZone'); ['dragenter','dragover'].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.add('drag');})); ['dragleave','drop'].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.remove('drag');})); drop.addEventListener('drop',event=>{const files=[...event.dataTransfer.files].filter(file=>file.name.toLowerCase().endsWith('.pdf'));if(files.length)uploadFiles(files).catch(error=>{byId('uploadStatus').textContent=error.message;});});
    byId('filterInput').addEventListener('input',renderDocuments); byId('refreshButton').addEventListener('click',()=>refresh().catch(error=>alert(error.message))); byId('syncButton').addEventListener('click',()=>startSync(false)); byId('reindexButton').addEventListener('click',()=>startSync(true)); byId('restartButton').addEventListener('click',restartApp);
    refresh().catch(error=>{byId('uploadStatus').textContent=error.message;});
  </script>
</body>
</html>"""
