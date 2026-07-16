import sqlite3
from pathlib import Path

import pytest

from corpus_scale import CorpusScaleManager
from document_manager import DuplicateDocumentError, DocumentRepository


def repository(tmp_path: Path) -> DocumentRepository:
    return DocumentRepository(
        pdf_dir=tmp_path / "docs",
        manifest_path=tmp_path / "db" / "manifest.json",
        state_path=tmp_path / "db" / "manager.json",
        trash_dir=tmp_path / "trash",
        quarantine_dir=tmp_path / "quarantine",
        revision_dir=tmp_path / "revisions",
    )


def manager(tmp_path: Path, repo: DocumentRepository) -> CorpusScaleManager:
    return CorpusScaleManager(
        repo,
        state_path=tmp_path / "db" / "corpus-state.json",
        db_dir=tmp_path / "db",
        workspace_db=tmp_path / "workspace.sqlite3",
        backup_dir=tmp_path / "backups",
        debug_dir=tmp_path / "debug",
    )


def pdf(tmp_path: Path, name: str, payload: bytes = b"original") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4\n" + payload + b"\n%%EOF\n")
    return path


def test_queue_is_persistent_and_pauses_between_documents(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.commit_upload(pdf(tmp_path, "one.part"), "one.pdf")
    repo.commit_upload(pdf(tmp_path, "two.part", b"second"), "two.pdf")
    corpus = manager(tmp_path, repo)

    queue = corpus.prepare_queue(repo.summary(), force=False)
    assert queue["remaining"] == 2
    corpus.mark_event("one.pdf", "processing")
    corpus.mark_event("one.pdf", "complete")
    corpus.request_pause()

    restarted = manager(tmp_path, repo)
    snapshot = restarted.snapshot()
    assert snapshot["paused"] is True
    assert snapshot["remaining"] == 1
    assert restarted.queued_sources() == ["two.pdf"]


def test_duplicate_detection_revision_history_and_rollback(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    original = pdf(tmp_path, "original.part")
    original_bytes = original.read_bytes()
    repo.commit_upload(original, "manual.pdf")

    with pytest.raises(DuplicateDocumentError) as duplicate:
        repo.commit_upload(pdf(tmp_path, "duplicate.part"), "copy.pdf")
    assert duplicate.value.duplicate_of == "manual.pdf"

    replaced = repo.commit_upload(
        pdf(tmp_path, "replacement.part", b"revised"),
        "manual.pdf",
        replace=True,
    )
    assert replaced["revision_id"]
    assert len(repo.list_revisions()) == 1

    repo.restore_revision(str(replaced["revision_id"]))
    assert repo.resolve_source("manual.pdf").read_bytes() == original_bytes
    assert len(repo.list_revisions()) == 2


def test_failed_document_quarantine_is_recoverable(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.commit_upload(pdf(tmp_path, "broken.part"), "broken.pdf")

    quarantined = repo.quarantine("broken.pdf", "RuntimeError: CUDA out of memory")
    assert not repo.resolve_source("broken.pdf").exists()
    assert repo.summary()["counts"]["quarantine"] == 1
    assert "CUDA out of memory" in repo.list_quarantine()[0]["error"]

    repo.restore_quarantine(str(quarantined["quarantine_id"]))
    assert repo.resolve_source("broken.pdf").is_file()
    assert repo.summary()["counts"]["quarantine"] == 0
    assert "broken.pdf" in repo.summary()["pending_sources"]


def test_health_reports_storage_and_preexisting_duplicates(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.pdf_dir.mkdir(parents=True)
    same = b"%PDF-1.4\nsame\n%%EOF\n"
    (repo.pdf_dir / "one.pdf").write_bytes(same)
    (repo.pdf_dir / "two.pdf").write_bytes(same)
    (tmp_path / "db").mkdir(exist_ok=True)
    (tmp_path / "db" / "index.bin").write_bytes(b"index-data")

    health = manager(tmp_path, repo).health(refresh=True)

    assert health["documents"] == 2
    assert len(health["duplicate_groups"]) == 1
    assert health["duplicate_groups"][0]["sources"] == ["one.pdf", "two.pdf"]
    assert health["storage"]["documents"] == len(same) * 2
    assert health["storage"]["index"] >= len(b"index-data")


def test_backup_and_restore_replace_the_full_corpus_snapshot(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    source = pdf(tmp_path, "manual.part")
    original_pdf = source.read_bytes()
    repo.commit_upload(source, "manual.pdf")
    repo.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    index_file = repo.manifest_path.parent / "index.bin"
    index_file.write_bytes(b"original-index")
    workspace = tmp_path / "workspace.sqlite3"
    with sqlite3.connect(workspace) as connection:
        connection.execute("CREATE TABLE notes (value TEXT NOT NULL)")
        connection.execute("INSERT INTO notes VALUES ('original-note')")

    corpus = manager(tmp_path, repo)
    backup = corpus.create_backup("before changes")
    assert Path(tmp_path / "backups" / backup["filename"]).is_file()

    repo.resolve_source("manual.pdf").write_bytes(b"%PDF-1.4\nchanged\n%%EOF\n")
    index_file.write_bytes(b"changed-index")
    with sqlite3.connect(workspace) as connection:
        connection.execute("UPDATE notes SET value = 'changed-note'")

    restored = corpus.restore_backup(str(backup["backup_id"]))

    assert restored["restored"] is True
    assert repo.resolve_source("manual.pdf").read_bytes() == original_pdf
    assert index_file.read_bytes() == b"original-index"
    with sqlite3.connect(workspace) as connection:
        assert connection.execute("SELECT value FROM notes").fetchone()[0] == "original-note"

