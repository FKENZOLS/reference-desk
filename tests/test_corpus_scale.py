import json
import sqlite3
from pathlib import Path

import chromadb
import pytest
from langchain_core.documents import Document

from corpus_scale import CorpusScaleManager, debug_artifact_paths
from document_manager import DuplicateDocumentError, DocumentRepository
from lexical_index import replace_source


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


def test_queue_reconciles_document_that_is_already_indexed(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.commit_upload(pdf(tmp_path, "manual.part"), "manual.pdf")
    corpus = manager(tmp_path, repo)
    corpus.prepare_queue(repo.summary(), force=False)
    repo.clear_pending_sources(("manual.pdf",), ())
    summary = {
        "pending_sources": [],
        "documents": [{"source_id": "manual.pdf", "status": "indexed"}],
    }

    reconciled = corpus.reconcile_indexed_documents(summary)
    snapshot = corpus.snapshot()

    assert reconciled == ["manual.pdf"]
    assert snapshot["remaining"] == 0
    assert snapshot["counts"] == {"complete": 1}


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


def test_storage_optimization_rebuilds_and_verifies_only_active_index(
    tmp_path: Path,
) -> None:
    repo = repository(tmp_path)
    repo.commit_upload(pdf(tmp_path, "manual.part"), "manual.pdf")
    repo.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repo.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "sources": {
                    "manual.pdf": {
                        "complete": True,
                        "chunk_count": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    repo.clear_pending_sources(("manual.pdf",), ())

    dense_client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    dense = dense_client.get_or_create_collection(
        "technical_docs_qwen_v1",
        metadata={"hnsw:space": "cosine"},
    )
    dense.add(
        ids=["chunk-1", "chunk-2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["first passage", "second passage"],
        metadatas=[
            {"source_id": "manual.pdf"},
            {"source_id": "manual.pdf"},
        ],
    )
    dense_client.close()

    lexical_documents = [
        Document(
            page_content="first passage",
            metadata={"document_title": "Manual", "source_id": "manual.pdf"},
        ),
        Document(
            page_content="second passage",
            metadata={"document_title": "Manual", "source_id": "manual.pdf"},
        ),
    ]
    replace_source(
        path=tmp_path / "db" / "lexical.sqlite3",
        source_id="manual.pdf",
        documents=lexical_documents,
        ids=["chunk-1", "chunk-2"],
    )

    active_debug, active_chunks = debug_artifact_paths(
        tmp_path / "debug",
        "manual.pdf",
    )
    active_debug.parent.mkdir(parents=True, exist_ok=True)
    active_debug.write_text("active", encoding="utf-8")
    active_chunks.write_text("active", encoding="utf-8")
    stale_debug, stale_chunks = debug_artifact_paths(
        tmp_path / "debug",
        "deleted.pdf",
    )
    stale_debug.write_text("stale", encoding="utf-8")
    stale_chunks.write_text("stale", encoding="utf-8")

    orphan = tmp_path / "db" / "11111111-1111-1111-1111-111111111111"
    orphan.mkdir()
    (orphan / "header.bin").write_bytes(b"old")
    (orphan / "data_level0.bin").write_bytes(b"unused vectors")

    corpus = manager(tmp_path, repo)
    assert corpus._index_snapshot() == {
        "manifest_chunks": 2,
        "manifest_sources": {"manual.pdf"},
        "dense_chunks": 2,
        "dense_sources": {"manual.pdf"},
        "lexical_chunks": 2,
        "lexical_sources": {"manual.pdf"},
    }
    result = corpus.optimize_storage()

    assert result["optimized"] is True
    assert result["chunks_verified"] == 2
    assert result["sources_verified"] == 1
    assert result["dense_index_rebuilt"] is True
    assert result["orphan_segments_removed"] == 1
    assert result["debug_files_removed"] == 2
    assert not orphan.exists()
    assert active_debug.read_text(encoding="utf-8") == "active"
    assert active_chunks.read_text(encoding="utf-8") == "active"
    assert not stale_debug.exists()
    assert not stale_chunks.exists()
    assert (tmp_path / "backups" / result["backup"]["filename"]).is_file()

    reopened = chromadb.PersistentClient(path=str(tmp_path / "db"))
    try:
        rebuilt = reopened.get_collection("technical_docs_qwen_v1")
        assert rebuilt.count() == 2
        assert set(rebuilt.get(include=[])["ids"]) == {"chunk-1", "chunk-2"}
    finally:
        reopened.close()
