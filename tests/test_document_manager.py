import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
from fastapi.testclient import TestClient

import search_app
from document_manager import (
    DocumentPathError,
    DocumentRepository,
    document_manager_html,
)


def repository(tmp_path: Path) -> DocumentRepository:
    return DocumentRepository(
        pdf_dir=tmp_path / "docs",
        manifest_path=tmp_path / "db" / "manifest.json",
        state_path=tmp_path / "db" / "manager.json",
        trash_dir=tmp_path / "trash",
    )


def temporary_pdf(tmp_path: Path, name: str = "upload.part") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return path


def test_document_paths_are_relative_pdf_paths() -> None:
    assert DocumentRepository.normalize_source_id("standards\\manual.pdf") == (
        "standards/manual.pdf"
    )
    for unsafe in ("../manual.pdf", "/manual.pdf", "manual.txt", "CON.pdf"):
        with pytest.raises(DocumentPathError):
            DocumentRepository.normalize_source_id(unsafe)


def test_upload_marks_document_pending_and_replace_is_explicit(tmp_path) -> None:
    repo = repository(tmp_path)
    stored = repo.commit_upload(
        temporary_pdf(tmp_path),
        "standards/manual.pdf",
    )
    assert stored["source_id"] == "standards/manual.pdf"
    summary = repo.summary()
    assert summary["counts"]["documents"] == 1
    assert summary["counts"]["pending"] == 1
    assert summary["documents"][0]["status"] == "pending"

    with pytest.raises(FileExistsError):
        repo.commit_upload(temporary_pdf(tmp_path, "again.part"), "standards/manual.pdf")

    replaced = temporary_pdf(tmp_path, "replacement.part")
    replaced.write_bytes(b"%PDF-1.7\nreplacement\n%%EOF\n")
    repo.commit_upload(replaced, "standards/manual.pdf", replace=True)
    assert repo.resolve_source("standards/manual.pdf").read_bytes().startswith(b"%PDF-1.7")


def test_indexed_status_uses_manifest_and_sync_clears_pending(tmp_path) -> None:
    repo = repository(tmp_path)
    repo.commit_upload(temporary_pdf(tmp_path), "manual.pdf")
    repo.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repo.manifest_path.write_text(
        json.dumps(
            {
                "sources": {
                    "manual.pdf": {
                        "complete": True,
                        "chunk_count": 18,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert repo.list_documents()[0]["status"] == "pending"
    repo.clear_pending()
    document = repo.list_documents()[0]
    assert document["status"] == "indexed"
    assert document["chunks"] == 18


def test_move_delete_and_restore_are_recoverable(tmp_path) -> None:
    repo = repository(tmp_path)
    repo.commit_upload(temporary_pdf(tmp_path), "manual.pdf")
    moved = repo.move("manual.pdf", "standards/renamed.pdf")
    assert moved["source_id"] == "standards/renamed.pdf"
    assert not repo.resolve_source("manual.pdf").exists()

    deleted = repo.trash("standards/renamed.pdf")
    assert repo.summary()["counts"]["trash"] == 1
    assert not repo.resolve_source("standards/renamed.pdf").exists()

    restored = repo.restore(deleted["trash_id"])
    assert restored["source_id"] == "standards/renamed.pdf"
    assert repo.resolve_source("standards/renamed.pdf").is_file()
    assert repo.summary()["counts"]["trash"] == 0


def test_document_manager_html_has_valid_javascript() -> None:
    rendered = document_manager_html()
    assert "Apply pending changes" in rendered
    assert "recoverable trash" in rendered
    assert "Add, organize, replace" not in rendered
    assert "Files are copied first" not in rendered
    assert "Moving or renaming a file" not in rendered
    assert 'id="restartButton"' in rendered
    assert "/documents/api/restart" in rendered
    assert 'href="/workspace"' in rendered
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for generated JavaScript validation")
    scripts = re.findall(r"<script>(.*?)</script>", rendered, re.DOTALL)
    assert scripts
    result = subprocess.run(
        [node, "--check", "-"],
        input=scripts[-1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_document_management_routes_cover_upload_move_delete_and_restore(
    tmp_path,
    monkeypatch,
) -> None:
    repo = repository(tmp_path)
    monkeypatch.setattr(search_app, "DOCUMENT_REPOSITORY", repo)
    app = search_app.create_web_app(gr.Blocks())
    client = TestClient(app)

    assert client.get("/documents").status_code == 200
    upload = client.post(
        "/documents/api/upload",
        params={"path": "standards/manual.pdf"},
        content=b"%PDF-1.4\n%%EOF\n",
        headers={"Content-Type": "application/pdf"},
    )
    assert upload.status_code == 200
    assert client.get("/documents/api/list").json()["counts"]["pending"] == 1

    moved = client.patch(
        "/documents/api/item/standards/manual.pdf",
        json={"target": "reference/renamed.pdf"},
    )
    assert moved.status_code == 200
    deleted = client.delete("/documents/api/item/reference/renamed.pdf")
    assert deleted.status_code == 200
    trash_id = deleted.json()["trash_id"]
    assert client.post(f"/documents/api/trash/{trash_id}/restore").status_code == 200
    assert repo.resolve_source("reference/renamed.pdf").is_file()


def test_document_restart_route_schedules_restart(monkeypatch) -> None:
    monkeypatch.setattr(
        search_app,
        "schedule_application_restart",
        lambda: {"restarting": True, "instance_id": "old-instance"},
    )
    monkeypatch.setattr(
        search_app,
        "_DOCUMENT_JOB",
        {"state": "complete", "running": False, "message": "", "log": []},
    )
    client = TestClient(search_app.create_web_app(gr.Blocks()))

    response = client.post("/documents/api/restart")

    assert response.status_code == 200
    assert response.json() == {
        "restarting": True,
        "instance_id": "old-instance",
    }


def test_application_restart_resets_runtime_and_changes_instance(monkeypatch) -> None:
    calls = []

    class ImmediateThread:
        def __init__(self, *, target, name, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(search_app, "_APP_INSTANCE_ID", "current-instance")
    monkeypatch.setattr(search_app, "_release_search_runtime", lambda: calls.append("runtime"))
    monkeypatch.setattr(
        search_app,
        "_unload_ollama_embedding_model",
        lambda: calls.append("ollama"),
    )
    monkeypatch.setattr(search_app.threading, "Thread", ImmediateThread)
    search_app._APP_RESTARTING.clear()
    search_app._DOCUMENT_MAINTENANCE.clear()

    response = search_app.schedule_application_restart()

    assert response == {"restarting": True, "instance_id": "current-instance"}
    assert search_app._APP_INSTANCE_ID != "current-instance"
    assert calls == ["runtime", "ollama"]
    assert not search_app._APP_RESTARTING.is_set()
    assert not search_app._DOCUMENT_MAINTENANCE.is_set()


def test_search_is_paused_during_document_maintenance() -> None:
    search_app._DOCUMENT_MAINTENANCE.set()
    try:
        with pytest.raises(search_app.DocumentMaintenanceError):
            search_app.search_with_additional("test query")
        with pytest.raises(search_app.DocumentMaintenanceError):
            search_app.get_citation_collection()
    finally:
        search_app._DOCUMENT_MAINTENANCE.clear()


def test_concurrent_ingestion_uses_live_free_memory_not_total(monkeypatch) -> None:
    monkeypatch.setattr(search_app, "_RUNTIME", object())
    monkeypatch.setattr(search_app, "SEARCH_DURING_INGESTION", "auto")
    monkeypatch.setattr(search_app, "DOCLING_GPU_HEADROOM_MB", 3500)
    monkeypatch.setattr(search_app, "CONCURRENT_QUERY_RESERVE_MB", 768)
    monkeypatch.setattr(search_app, "_gpu_memory_mib", lambda: (4268, 6144))

    allowed, _ = search_app.concurrent_ingestion_policy()

    assert allowed is True
    monkeypatch.setattr(search_app, "_gpu_memory_mib", lambda: (4267, 24576))
    allowed, reason = search_app.concurrent_ingestion_policy()
    assert allowed is False
    assert "4267 MiB free" in reason


def test_concurrent_indexing_allows_a_query_when_headroom_remains(monkeypatch) -> None:
    monkeypatch.setattr(search_app, "_RUNTIME", object())
    monkeypatch.setattr(search_app, "SEARCH_DURING_INGESTION", "auto")
    monkeypatch.setattr(search_app, "CONCURRENT_QUERY_RESERVE_MB", 768)
    monkeypatch.setattr(search_app, "_gpu_memory_mib", lambda: (900, 6144))
    monkeypatch.setattr(
        search_app,
        "_DOCUMENT_JOB",
        {"running": True, "search_available": True, "log": []},
    )

    search_app._guard_search_during_ingestion()

    monkeypatch.setattr(search_app, "_gpu_memory_mib", lambda: (767, 6144))
    with pytest.raises(search_app.DocumentMaintenanceError, match="767 MiB"):
        search_app._guard_search_during_ingestion()


def test_index_commit_pauses_search_until_the_child_finishes(tmp_path, monkeypatch) -> None:
    gate = tmp_path / "commit-gate.json"
    monkeypatch.setattr(search_app, "_INDEX_COMMIT_GATE_PATH", gate)
    monkeypatch.setattr(search_app, "_INDEX_COMMIT_BARRIER", None)
    search_app._DOCUMENT_MAINTENANCE.clear()

    assert search_app._handle_corpus_event(
        'CORPUS_EVENT {"event":"commit_requested","source_id":"manual.pdf","token":"abc"}\n'
    )
    deadline = time.monotonic() + 1
    while (not gate.is_file() or not search_app._DOCUMENT_MAINTENANCE.is_set()) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert search_app._DOCUMENT_MAINTENANCE.is_set()
    assert json.loads(gate.read_text(encoding="utf-8")) == {"token": "abc"}
    assert not search_app._SEARCH_MAINTENANCE_LOCK.acquire(blocking=False)

    assert search_app._handle_corpus_event(
        'CORPUS_EVENT {"event":"commit_finished","source_id":"manual.pdf","token":"abc"}\n'
    )
    deadline = time.monotonic() + 1
    while search_app._DOCUMENT_MAINTENANCE.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not search_app._DOCUMENT_MAINTENANCE.is_set()
    assert search_app._SEARCH_MAINTENANCE_LOCK.acquire(blocking=False)
    search_app._SEARCH_MAINTENANCE_LOCK.release()


def test_successful_background_job_clears_pending_changes(monkeypatch) -> None:
    cleared = []
    fake_repository = SimpleNamespace(clear_pending=lambda: cleared.append(True))

    class FakeProcess:
        stdout = iter(["Processing: manual.pdf\n", "Ingestion complete\n"])

        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr(search_app, "DOCUMENT_REPOSITORY", fake_repository)
    monkeypatch.setattr(search_app, "_release_search_runtime", lambda: None)
    monkeypatch.setattr(search_app, "_unload_ollama_embedding_model", lambda: None)
    monkeypatch.setattr(search_app.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        search_app,
        "_DOCUMENT_JOB",
        {"state": "queued", "running": True, "message": "", "log": []},
    )
    search_app._DOCUMENT_MAINTENANCE.set()

    search_app._run_document_index_job(False)

    assert cleared == [True]
    assert search_app.document_job_snapshot()["state"] == "complete"
    assert not search_app._DOCUMENT_MAINTENANCE.is_set()
