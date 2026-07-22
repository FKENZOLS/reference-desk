import gradio as gr
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import search_app
from search_app import Candidate
from workspace_store import WorkspaceStore


def test_react_search_api_returns_structured_citations(tmp_path, monkeypatch) -> None:
    gte_fingerprint = search_app.reranker_fingerprint("gte")
    result = Candidate(
        document=Document(
            page_content="The station dwell time shall be 30 seconds.",
            metadata={
                "chunk_id": "chunk-1",
                "source_id": "standards/manual.pdf",
                "document_title": "Transit specification",
                "page_start": 51,
                "page_end": 51,
                "section_path": "5.1.3 Station dwell time",
                "content_labels": "requirement",
            },
        ),
        chunk_id="chunk-1",
        retrieval_rank=1,
        rerank_logit=3.5,
        rerank_probability=0.97,
        rerank_rank=1,
        final_score=0.91,
        final_rank=1,
        reranker_choice="gte",
        reranker_model="Alibaba-NLP/gte-multilingual-reranker-base",
        reranker_fingerprint=gte_fingerprint,
    )
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", WorkspaceStore(tmp_path / "workspace.sqlite3"))
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return (
            "",
            "",
            [result],
            [],
            {
                "dense_seconds": 0.1,
                "lexical_seconds": 0.05,
                "rerank_seconds": 0.2,
                "total_seconds": 0.35,
                "reranker_truncation_rate": 0.0,
                "best_rerank_logit": 3.5,
                "considered_count": 1,
                "reranker_choice": "gte",
                "reranker_model": "Alibaba-NLP/gte-multilingual-reranker-base",
                "reranker_fingerprint": gte_fingerprint,
            },
        )

    monkeypatch.setattr(search_app, "search_with_additional", fake_search)

    client = TestClient(search_app.create_web_app(gr.Blocks()))
    response = client.post(
        "/api/search",
        json={"query": "station dwell time", "reranker_choice": "gte"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["chunk_id"] == "chunk-1"
    assert payload["results"][0]["citation_url"].startswith(
        "/viewer/standards/manual.pdf?page=51"
    )
    assert payload["results"][0]["feedback"]["query"] == "station dwell time"
    assert payload["results"][0]["feedback"]["reranker_choice"] == "gte"
    assert payload["reranker"]["choice"] == "gte"
    assert captured["reranker_choice"] == "gte"
    assert payload["result_ids"] == ["chunk-1"]


def test_reranker_options_and_invalid_choice() -> None:
    client = TestClient(search_app.create_web_app(gr.Blocks()))

    options = client.get("/api/rerankers")
    assert options.status_code == 200
    assert options.json()["status"]["status"] in {
        "idle", "loading", "ready", "failed", "restart_required"
    }
    assert {item["value"] for item in options.json()["options"]} == {
        "bge",
        "gte",
    }
    invalid = client.post(
        "/api/search",
        json={"query": "station dwell time", "reranker_choice": "unknown"},
    )
    assert invalid.status_code == 400


def test_source_update_api_requires_fresh_confirmation(monkeypatch) -> None:
    status = {
        "available": True,
        "can_update": True,
        "blocked_reason": None,
        "supervised_restart": False,
        "restart_required": False,
    }
    monkeypatch.setattr(search_app, "source_update_status", lambda fetch=False: dict(status))
    monkeypatch.setattr(search_app, "apply_source_update", lambda: {**status, "updated": False})
    client = TestClient(search_app.create_web_app(gr.Blocks()))

    current = client.get("/updates/api/status")
    assert current.status_code == 200
    token = current.json()["action_token"]
    denied = client.post(
        "/updates/api/apply",
        json={"confirm": True, "action_token": "wrong"},
    )
    assert denied.status_code == 403
    accepted = client.post(
        "/updates/api/apply",
        json={"confirm": True, "action_token": token},
    )
    assert accepted.status_code == 200
    assert accepted.json()["updated"] is False


def test_source_choices_do_not_start_the_search_runtime(tmp_path, monkeypatch) -> None:
    class FakeCollection:
        @staticmethod
        def get(*, include):
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {
                        "source_id": "standards/manual.pdf",
                        "document_title": "Transit specification",
                    }
                ]
            }

    class FakeClient:
        @staticmethod
        def get_collection(name):
            assert name == search_app.COLLECTION_NAME
            return FakeCollection()

    database = tmp_path / "chroma"
    database.mkdir()
    monkeypatch.setattr(search_app, "DB_DIR", database)
    monkeypatch.setattr(
        search_app,
        "get_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must stay lazy")),
    )
    monkeypatch.setattr(
        search_app.chromadb,
        "PersistentClient",
        lambda **kwargs: FakeClient(),
    )
    search_app._DOCUMENT_MAINTENANCE.clear()

    assert search_app.source_choices() == [
        ("All documents", ""),
        ("Transit specification — standards/manual.pdf", "standards/manual.pdf"),
    ]


def test_server_schedules_reranker_warmup_without_blocking_launch(monkeypatch) -> None:
    events = []

    class ImmediateTimer:
        def __init__(self, interval, function):
            events.append(("scheduled", interval))
            self.function = function
            self.daemon = False

        def start(self):
            events.append(("timer_started", self.daemon))
            self.function()

    fake_app = object()
    monkeypatch.setattr(search_app, "create_web_app", lambda: fake_app)
    monkeypatch.setattr(search_app, "get_runtime", lambda: events.append(("warmed", True)))
    monkeypatch.setattr(
        search_app,
        "_set_reranker_state",
        lambda **values: events.append(("state", values)),
    )
    monkeypatch.setattr(search_app.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(search_app, "WARM_RERANKER_ON_START", True)
    monkeypatch.setattr(search_app, "STARTUP_WARM_DELAY_SECONDS", 2.0)
    monkeypatch.setattr(search_app, "OPEN_BROWSER", False)
    monkeypatch.setattr(
        search_app.uvicorn,
        "run",
        lambda app, host, port: events.append(("served", app, host, port)),
    )

    search_app.launch_gradio()

    assert events[0][0] == "state"
    assert events[0][1]["status"] == "loading"
    assert "background" in events[0][1]["message"]
    assert events[1:] == [
        ("scheduled", 2.0),
        ("timer_started", True),
        ("warmed", True),
        ("served", fake_app, search_app.SERVER_HOST, search_app.SERVER_PORT),
    ]
