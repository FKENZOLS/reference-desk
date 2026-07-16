import gradio as gr
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import search_app
from search_app import Candidate
from workspace_store import WorkspaceStore


def test_react_search_api_returns_structured_citations(tmp_path, monkeypatch) -> None:
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
    )
    monkeypatch.setattr(search_app, "WORKSPACE_STORE", WorkspaceStore(tmp_path / "workspace.sqlite3"))
    monkeypatch.setattr(
        search_app,
        "search_with_additional",
        lambda *args, **kwargs: (
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
            },
        ),
    )

    client = TestClient(search_app.create_web_app(gr.Blocks()))
    response = client.post("/api/search", json={"query": "station dwell time"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["chunk_id"] == "chunk-1"
    assert payload["results"][0]["citation_url"].startswith(
        "/viewer/standards/manual.pdf?page=51"
    )
    assert payload["results"][0]["feedback"]["query"] == "station dwell time"
    assert payload["result_ids"] == ["chunk-1"]
