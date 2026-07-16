import re
import shutil
import subprocess

import gradio as gr
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import search_app
from quality_ui import quality_dashboard_html
from search_app import Candidate, select_results
from workspace_store import WorkspaceStore


def feedback_payload(**overrides):
    payload = {
        "query": "Where is stopping distance defined?",
        "judgment": "relevant",
        "chunk_id": "chunk-1",
        "source_id": "manual.pdf",
        "document_title": "Technical Manual",
        "page_start": 12,
        "page_end": 12,
        "section": "Braking",
        "excerpt": "The train shall stop within the specified distance.",
        "result_rank": 1,
        "rerank_logit": 2.5,
        "final_score": 0.02,
    }
    payload.update(overrides)
    return payload


def candidate(score: float, chunk_id: str) -> Candidate:
    item = Candidate(
        document=Document(
            page_content="A reference passage",
            metadata={"source_id": "manual.pdf", "page_start": 12},
        ),
        chunk_id=chunk_id,
    )
    item.rerank_logit = score
    return item


def test_feedback_builds_only_unambiguous_benchmark_cases(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "quality.sqlite3")
    store.upsert_feedback(feedback_payload(), min_positive=99, min_negative=99)
    store.upsert_feedback(
        feedback_payload(
            query="What is the lunar warranty?",
            judgment="no_relevant_result",
            chunk_id="",
            source_id="",
            rerank_logit=-3.0,
        ),
        min_positive=99,
        min_negative=99,
    )
    store.upsert_feedback(
        feedback_payload(
            query="A query with one wrong hit",
            judgment="wrong_passage",
            chunk_id="wrong-only",
            rerank_logit=-1.0,
        ),
        min_positive=99,
        min_negative=99,
    )

    cases = store.benchmark_cases_from_feedback()
    assert len(cases) == 2
    assert cases[0]["answerable"] is True
    assert cases[0]["relevant_chunk_ids"] == ["chunk-1"]
    assert cases[0]["relevant_locations"] == [
        {"source_id": "manual.pdf", "page": 12}
    ]
    assert cases[1]["answerable"] is False
    assert "A query with one wrong hit" not in store.benchmark_jsonl()


def test_calibration_waits_for_both_classes_and_can_be_paused(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "quality.sqlite3")
    store.upsert_feedback(feedback_payload(), min_positive=2, min_negative=2)
    status = store.calibration_status(min_positive=2, min_negative=2)
    assert status["active"] is False

    store.upsert_feedback(
        feedback_payload(query="positive two", chunk_id="p2", rerank_logit=3.0),
        min_positive=2,
        min_negative=2,
    )
    for index, score in enumerate((-2.0, -1.0), start=1):
        store.upsert_feedback(
            feedback_payload(
                query=f"negative {index}",
                judgment="wrong_passage",
                chunk_id=f"n{index}",
                rerank_logit=score,
            ),
            min_positive=2,
            min_negative=2,
        )

    status = store.calibration_status(min_positive=2, min_negative=2)
    assert status["active"] is True
    assert status["threshold"] == 2.5
    assert status["positive_recall"] == 1.0
    assert status["specificity"] == 1.0
    store.set_calibration_enabled(False)
    assert store.calibration_status(min_positive=2, min_negative=2)["active"] is False


def test_feedback_calibration_drives_live_result_rejection(tmp_path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "quality.sqlite3")
    store.upsert_feedback(
        feedback_payload(rerank_logit=2.0),
        min_positive=1,
        min_negative=1,
    )
    store.upsert_feedback(
        feedback_payload(
            query="negative",
            judgment="wrong_document",
            chunk_id="negative",
            rerank_logit=-1.0,
        ),
        min_positive=1,
        min_negative=1,
    )
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", store)
    monkeypatch.setattr(search_app, "QUALITY_MIN_POSITIVE_LABELS", 1)
    monkeypatch.setattr(search_app, "QUALITY_MIN_NEGATIVE_LABELS", 1)
    monkeypatch.setattr(search_app, "ENABLE_RELEVANCE_GATE", False)

    assert select_results([candidate(2.2, "strong")])
    assert select_results([candidate(-2.0, "weak")]) == []


def test_quality_dashboard_and_routes(tmp_path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "quality.sqlite3")
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", store)
    app = search_app.create_web_app(gr.Blocks())
    client = TestClient(app)

    response = client.post("/quality/api/feedback", json=feedback_payload())
    assert response.status_code == 200
    assert response.json()["feedback"]["judgment"] == "relevant"
    page = client.get("/quality")
    assert page.status_code == 200
    assert '<div id="root"></div>' in page.text
    state = client.get("/quality/api/state")
    assert state.status_code == 200
    assert state.json()["summary"]["total"] == 1
    benchmark = client.get("/quality/export/benchmark")
    assert benchmark.status_code == 200
    assert '"relevant_chunk_ids": ["chunk-1"]' in benchmark.text


def test_quality_html_and_result_controls_are_safe_and_valid() -> None:
    rendered = quality_dashboard_html(
        {
            "total": 0,
            "counts": {
                "relevant": 0,
                "wrong_passage": 0,
                "wrong_document": 0,
                "no_relevant_result": 0,
            },
            "benchmark_cases": 0,
            "answerable_cases": 0,
            "unanswerable_cases": 0,
            "calibration": {
                "active": False,
                "ready": False,
                "enabled": True,
                "positive_count": 0,
                "negative_count": 0,
                "minimum_positive": 20,
                "minimum_negative": 20,
                "threshold": None,
                "positive_recall": None,
                "specificity": None,
                "balanced_accuracy": None,
            },
        },
        [{"query": "</script><script>bad()</script>"}],
    )
    assert "</script><script>bad()" not in rendered
    assert "\\u003c/script\\u003e" in rendered

    block = search_app.format_candidate_block(
        "stopping distance",
        candidate(2.0, "chunk-safe"),
    )
    assert "Wrong passage" in block
    assert "Wrong document" in block
    assert "rag-feedback-payload" in block
    assert "rag-feedback-relevant" in block

    node = shutil.which("node")
    if node is not None:
        scripts = re.findall(r"<script>(.*?)</script>", rendered, re.DOTALL)
        result = subprocess.run(
            [node, "--check", "-"],
            input=scripts[-1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
