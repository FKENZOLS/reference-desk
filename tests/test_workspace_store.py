import io
import re
import shutil
import subprocess
import zipfile

import gradio as gr
from fastapi.testclient import TestClient

import search_app
from workspace_store import WorkspaceStore, parse_export_ids
from workspace_ui import workspace_html


def bookmark_payload(**overrides):
    payload = {
        "chunk_id": "chunk-1",
        "source_id": "manual.pdf",
        "document_title": "Technical Manual",
        "page_start": 12,
        "page_end": 13,
        "section": "Braking requirements",
        "content_type": "text",
        "document_date": "2026",
        "excerpt": "The train shall stop within the specified distance.",
        "citation_label": "Technical Manual. Braking requirements. page 12-13.",
        "citation_url": "/viewer/manual.pdf?page=12&chunk=chunk-1",
        "query": "stopping distance",
        "note": "Use this in the safety comparison.",
    }
    payload.update(overrides)
    return payload


def test_workspace_persists_bookmarks_notes_collections_and_history(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    collection = store.create_collection("Safety review")
    bookmark = store.upsert_bookmark(
        bookmark_payload(collection_id=collection["id"])
    )
    assert bookmark["collection_name"] == "Safety review"
    assert bookmark["note"].startswith("Use this")

    updated = store.update_bookmark(bookmark["id"], note="Reviewed")
    assert updated["note"] == "Reviewed"
    assert store.list_bookmarks()[0]["id"] == bookmark["id"]

    history = store.record_search(
        {
            "query": "braking distance",
            "content_filter": "requirement",
            "result_count": 7,
        }
    )
    assert history["result_count"] == 7
    assert store.list_history()[0]["query"] == "braking distance"


def test_exports_selected_passages_to_markdown_and_valid_docx(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    bookmark = store.upsert_bookmark(bookmark_payload())
    markdown = store.markdown_export([bookmark["id"]])
    assert "# Research excerpts" in markdown
    assert "> The train shall stop" in markdown
    assert "Technical Manual. Braking requirements" in markdown

    document = store.docx_export([bookmark["id"]])
    with zipfile.ZipFile(io.BytesIO(document)) as archive:
        assert "word/document.xml" in archive.namelist()
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "The train shall stop" in xml
    assert "Use this in the safety comparison" in xml


def test_workspace_html_escapes_embedded_json_and_has_compare_controls() -> None:
    rendered = workspace_html(
        [bookmark_payload(id=1, document_title="</script><script>bad()</script>")],
        [],
        [],
    )
    assert "</script><script>bad()" not in rendered
    assert "\\u003c/script\\u003e" in rendered
    assert "Compare side by side" in rendered
    assert "Export Word" in rendered
    assert "Search within saved passages" in rendered


def test_generated_workspace_javascript_has_valid_syntax() -> None:
    node = shutil.which("node")
    if node is None:
        return
    rendered = workspace_html([], [], [])
    scripts = re.findall(r"<script>(.*?)</script>", rendered, re.DOTALL)
    result = subprocess.run(
        [node, "--check", "-"],
        input=scripts[-1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_workspace_routes_save_update_and_export(tmp_path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", store)
    app = search_app.create_web_app(gr.Blocks())
    client = TestClient(app)

    collection_response = client.post(
        "/workspace/api/collections",
        json={"name": "System requirements"},
    )
    assert collection_response.status_code == 200
    collection_id = collection_response.json()["id"]

    bookmark_response = client.post(
        "/workspace/api/bookmarks",
        json=bookmark_payload(collection_id=collection_id),
    )
    assert bookmark_response.status_code == 200
    bookmark_id = bookmark_response.json()["id"]

    assert client.get("/workspace").status_code == 200
    update_response = client.patch(
        f"/workspace/api/bookmarks/{bookmark_id}",
        json={"note": "Verified against revision B"},
    )
    assert update_response.json()["note"] == "Verified against revision B"
    assert client.get(
        f"/workspace/export/markdown?ids={bookmark_id}"
    ).status_code == 200
    word_response = client.get(f"/workspace/export/word?ids={bookmark_id}")
    assert word_response.status_code == 200
    assert word_response.content.startswith(b"PK")


def test_export_id_parser_is_ordered_deduplicated_and_safe() -> None:
    assert parse_export_ids("3,nope,2,3,-1") == [3, 2]
