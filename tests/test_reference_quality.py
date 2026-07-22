import re
import shutil
import subprocess
import io
import sqlite3
import zipfile
from types import SimpleNamespace

import json
import pytest

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
            judgment="wrong_passage",
            chunk_id="hard-negative",
            rerank_logit=-1.0,
        ),
        min_positive=99,
        min_negative=99,
    )
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
    assert cases[0]["relevant_targets"] == [
        {
            "chunk_id": "chunk-1",
            "source_id": "manual.pdf",
            "page": 12,
        }
    ]
    assert cases[0]["hard_negative_chunk_ids"] == ["hard-negative"]
    assert cases[1]["answerable"] is False
    assert "A query with one wrong hit" not in store.benchmark_jsonl()


def test_feedback_identity_is_isolated_by_reranker(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "model-quality.sqlite3")
    store.upsert_feedback(
        feedback_payload(
            reranker_model="gte",
            reranker_fingerprint="gte-fingerprint",
        ),
        min_positive=99,
        min_negative=99,
    )
    store.upsert_feedback(
        feedback_payload(
            judgment="wrong_passage",
            reranker_model="bge",
            reranker_fingerprint="bge-fingerprint",
        ),
        min_positive=99,
        min_negative=99,
    )

    assert len(store.list_feedback()) == 2
    assert len(
        store.list_feedback(reranker_fingerprint="gte-fingerprint")
    ) == 1
    assert store.quality_summary(
        reranker_fingerprint="gte-fingerprint"
    )["counts"]["relevant"] == 1
    assert store.quality_summary(
        reranker_fingerprint="bge-fingerprint"
    )["counts"]["wrong_passage"] == 1


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


def test_calibration_is_kept_separate_for_each_reranker(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "quality.sqlite3")
    for fingerprint, positive, negative in (
        ("gte-fingerprint", 2.0, -1.0),
        ("bge-fingerprint", 4.0, -3.0),
    ):
        model = "gte" if fingerprint.startswith("gte") else "bge"
        store.upsert_feedback(
            feedback_payload(
                query=f"{model} positive",
                chunk_id=f"{model}-positive",
                rerank_logit=positive,
                reranker_model=model,
                reranker_fingerprint=fingerprint,
            ),
            min_positive=1,
            min_negative=1,
        )
        store.upsert_feedback(
            feedback_payload(
                query=f"{model} negative",
                judgment="wrong_passage",
                chunk_id=f"{model}-negative",
                rerank_logit=negative,
                reranker_model=model,
                reranker_fingerprint=fingerprint,
            ),
            min_positive=1,
            min_negative=1,
        )

    gte = store.calibration_status(
        min_positive=1,
        min_negative=1,
        reranker_model="gte",
        reranker_fingerprint="gte-fingerprint",
    )
    bge = store.calibration_status(
        min_positive=1,
        min_negative=1,
        reranker_model="bge",
        reranker_fingerprint="bge-fingerprint",
    )
    assert gte["active"] is True
    assert gte["threshold"] == 2.0
    assert bge["active"] is True
    assert bge["threshold"] == 4.0


def test_feedback_calibration_drives_live_result_rejection(tmp_path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "quality.sqlite3")
    model_fields = {
        "reranker_model": search_app.RERANKER_MODEL,
        "reranker_fingerprint": search_app.reranker_fingerprint(),
    }
    store.upsert_feedback(
        feedback_payload(rerank_logit=2.0, **model_fields),
        min_positive=1,
        min_negative=1,
    )
    store.upsert_feedback(
        feedback_payload(
            query="negative",
            judgment="wrong_document",
            chunk_id="negative",
            rerank_logit=-1.0,
            **model_fields,
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


def test_benchmarks_and_experiments_are_versioned_and_promotable(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "experiments.sqlite3")
    content = json.dumps(
        {
            "query": "Where is the braking requirement?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk-1"],
        }
    )
    benchmark = store.save_benchmark("Safety", content)
    assert benchmark["version"] == 1
    assert benchmark["case_count"] == 1

    changed = store.save_benchmark(
        "Safety",
        content
        + "\n"
        + json.dumps({"query": "Missing answer", "answerable": False}),
    )
    assert changed["version"] == 2
    assert changed["metadata"]["splits"] == {"test": 2}
    first_version = store.get_benchmark_version(
        int(changed["id"]),
        1,
        include_content=True,
    )
    assert first_version["case_count"] == 1
    experiment = store.create_experiment(
        "GTE baseline",
        int(changed["id"]),
        "Safety v2",
        {
            "reranker": "gte",
            "candidate_count": 20,
            "rerank_weight": 0.6,
            "passage_mode": "metadata-child",
        },
    )
    store.update_experiment(
        int(experiment["id"]),
        status="complete",
        results={"models": {"gte": {"summary": {"ndcg_at_5": 0.8}}}},
    )
    production = store.set_production_experiment(int(experiment["id"]))

    assert production["production"] is True
    assert store.production_experiment()["config"]["candidate_count"] == 20


def test_expected_passages_ambiguous_queries_notifications_and_regression_gate(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "quality-v4.sqlite3")
    store.upsert_feedback(
        feedback_payload(
            query="Where is the omitted definition?",
            judgment="expected_passage",
            chunk_id="",
            expected_source_id="manual.pdf",
            expected_page=44,
            reason="The UI did not display it",
        ),
        min_positive=99,
        min_negative=99,
    )
    cases = store.benchmark_cases_from_feedback()
    assert cases[0]["relevant_locations"] == [
        {"source_id": "manual.pdf", "page": 44}
    ]

    store.upsert_feedback(
        feedback_payload(
            query="Where is the omitted definition?",
            judgment="ambiguous",
            chunk_id="",
            rerank_logit=None,
        ),
        min_positive=99,
        min_negative=99,
    )
    assert store.benchmark_cases_from_feedback() == []

    notice = store.add_notification(
        kind="indexing",
        title="Index complete",
        status="success",
    )
    assert store.list_notifications(unread_only=True)[0]["id"] == notice["id"]
    store.mark_notification_read(int(notice["id"]))
    assert store.list_notifications(unread_only=True) == []
    assert store.schema_status()["current_version"] == 4

    experiment = store.create_experiment(
        "Regression",
        None,
        "Feedback",
        {"reranker": "gte", "candidate_count": 20, "rerank_weight": 0.6},
    )
    store.update_experiment(
        int(experiment["id"]),
        status="complete",
        results={"regression": {"passed": False, "threshold": 0.03}},
    )
    with pytest.raises(ValueError, match="regression"):
        store.set_production_experiment(int(experiment["id"]))


def test_comparison_experiment_cannot_become_production(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "comparison.sqlite3")
    experiment = store.create_experiment(
        "GTE versus BGE",
        None,
        "Benchmark",
        {"reranker": "both", "candidate_count": 20, "rerank_weight": 0.6},
    )
    store.update_experiment(
        int(experiment["id"]),
        status="complete",
        results={"regression": {"passed": True}},
    )

    with pytest.raises(ValueError, match="exactly one reranker"):
        store.set_production_experiment(int(experiment["id"]))


def test_production_experiment_sets_the_reranker_api_default(tmp_path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "production-default.sqlite3")
    experiment = store.create_experiment(
        "BGE production",
        None,
        "Benchmark",
        {"reranker": "bge", "candidate_count": 20, "rerank_weight": 0.6},
    )
    store.update_experiment(
        int(experiment["id"]),
        status="complete",
        results={"regression": {"passed": True}},
    )
    store.set_production_experiment(int(experiment["id"]))
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", store)

    response = TestClient(search_app.create_web_app()).get("/api/rerankers")

    assert response.status_code == 200
    assert response.json()["default"] == "bge"


def test_incompatible_regression_baseline_is_rejected(tmp_path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "baseline.sqlite3")
    benchmark = store.save_benchmark(
        "Held out",
        json.dumps(
            {
                "query": "Where is braking defined?",
                "split": "test",
                "relevant_chunk_ids": ["chunk-1"],
            }
        ),
    )
    stored = store.get_benchmark(int(benchmark["id"]), include_content=True)
    cases = search_app._benchmark_cases_from_content(stored["content_jsonl"])
    baseline = store.create_experiment(
        "GTE baseline",
        int(benchmark["id"]),
        "Held out v1",
        {
            "reranker": "gte",
            "benchmark_signature": search_app._benchmark_signature(cases),
        },
    )
    store.update_experiment(
        int(baseline["id"]),
        status="complete",
        results={"models": {"gte": {"summary": {"ndcg_at_5": 0.8}}}},
    )
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", store)
    search_app._EXPERIMENT_ACTIVE.clear()

    with pytest.raises(ValueError, match="no compatible result for: bge"):
        search_app.start_quality_experiment(
            {
                "benchmark_key": f"stored:{benchmark['id']}",
                "reranker": "bge",
                "split": "all",
                "baseline_experiment_id": baseline["id"],
            }
        )


def test_workspace_migration_rolls_back_as_one_transaction(tmp_path, monkeypatch) -> None:
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE retrieval_feedback (
                id INTEGER PRIMARY KEY,
                judgment TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def fail_after_rename(connection):
        connection.execute(
            "ALTER TABLE retrieval_feedback RENAME TO retrieval_feedback_legacy"
        )
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(
        WorkspaceStore,
        "_migrate_feedback_identity",
        staticmethod(fail_after_rename),
    )
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        WorkspaceStore(database)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    assert "retrieval_feedback" in tables
    assert "retrieval_feedback_legacy" not in tables
    assert version == 0


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


def test_diagnostic_bundle_excludes_queries_paths_and_excerpts(tmp_path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "diagnostics.sqlite3")
    store.add_notification(
        kind="reranker",
        title="Reranker failed",
        message="Private query and C:/private/manual.pdf must not be exported",
        status="error",
    )
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", store)
    monkeypatch.setattr(
        search_app,
        "CORPUS_SCALE",
        SimpleNamespace(
            state_path=tmp_path / "corpus.json",
            health=lambda: {
                "status": "healthy",
                "generated_at": "now",
                "documents": 1,
                "pages": 2,
                "chunks": 3,
                "indexed": 1,
                "pending": 0,
                "quarantined": 0,
                "revisions": 0,
                "storage": {},
                "issues": [],
            },
            snapshot=lambda: {
                "paused": False,
                "remaining": 0,
                "counts": {"complete": 1},
                "updated_at": "now",
                "items": [{"source_id": "C:/private/manual.pdf"}],
            },
        ),
    )
    monkeypatch.setattr(
        search_app,
        "DOCUMENT_REPOSITORY",
        SimpleNamespace(
            state_path=tmp_path / "documents.json",
            _read_json=lambda _path, _default: {"version": 2},
        ),
    )
    monkeypatch.setattr(search_app, "HARDWARE", SimpleNamespace(as_dict=lambda: {}))

    archive = zipfile.ZipFile(io.BytesIO(search_app.diagnostic_bundle()))
    combined = "\n".join(
        archive.read(name).decode("utf-8") for name in archive.namelist()
    )

    assert "privacy.json" in archive.namelist()
    assert "C:/private/manual.pdf" not in combined
    assert "Private query" not in combined
    assert "passage excerpts" in combined
