from langchain_core.documents import Document

import re
import shutil
import subprocess
import threading
from types import SimpleNamespace

import pytest

import search_app
from search_app import (
    Candidate,
    RetrievalFilters,
    additional_button_label,
    bbox_to_percentages,
    first_existing,
    gradio_search,
    normalize_source_filter,
    register_source_navigation,
    rerank_candidates,
    resolve_source_path,
    resolve_source_navigation,
    resolve_torch_device,
    select_results,
    select_additional_results,
    source_citation,
    source_viewer_html,
    source_url,
    text_lines_from_pdfium_page,
    toggle_additional_sources,
    document_matches_filters,
)


def candidate(score: float) -> Candidate:
    item = Candidate(
        document=Document(page_content="content", metadata={}),
        chunk_id=str(score),
    )
    item.rerank_logit = score
    return item


def test_missing_page_sentinel_is_not_displayed() -> None:
    assert first_existing({"page": -1}, ["page"]) == "not available"


def test_all_documents_scope_has_no_source_filter() -> None:
    assert normalize_source_filter("") is None
    assert normalize_source_filter(None) is None
    assert normalize_source_filter("  bases.pdf  ") == "bases.pdf"


def test_metadata_filters_cover_section_table_requirement_and_date() -> None:
    table = Document(
        page_content="Maximum speed values",
        metadata={
            "section_path": "7.4 Vehicle performance",
            "content_labels": "table, text",
            "document_title": "Specification June 2026",
        },
    )
    requirement = Document(
        page_content="The vehicle shall stop safely.",
        metadata={"section_path": "Braking", "content_labels": "text"},
    )
    assert document_matches_filters(
        table,
        RetrievalFilters(section="vehicle", content_type="table", date="2026"),
    )
    assert document_matches_filters(
        requirement,
        RetrievalFilters(content_type="requirement"),
    )
    assert not document_matches_filters(
        requirement,
        RetrievalFilters(content_type="table"),
    )


def test_gate_is_disabled_until_scores_are_calibrated() -> None:
    selected = select_results([candidate(4.0), candidate(-10.0)])
    assert len(selected) == 2


def test_enabled_result_floor_does_not_force_a_second_result(monkeypatch) -> None:
    monkeypatch.setattr(search_app, "ENABLE_RELEVANCE_GATE", True)
    selected = select_results([candidate(4.0), candidate(-10.0)])
    assert len(selected) == 1


def test_conservative_blend_protects_strong_retrieval_rank(monkeypatch) -> None:
    first = candidate(0.0)
    first.retrieval_rank = 1
    second = candidate(0.0)
    second.retrieval_rank = 20

    class FakeReranker:
        @staticmethod
        def predict(pairs):
            assert len(pairs) == 2
            return [(-2.0, 0.1, 10, False), (2.0, 0.9, 10, False)]

    runtime = SimpleNamespace(
        reranker=FakeReranker(),
        inference_lock=threading.Lock(),
    )
    ranked = rerank_candidates(runtime, "question", [first, second])
    assert ranked[0] is first
    assert first.retrieval_rank == 1
    assert first.rerank_rank == 2


def test_cpu_device_is_always_available() -> None:
    assert resolve_torch_device("cpu").type == "cpu"


def test_requested_cuda_never_silently_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(search_app, "COMPUTE_BACKEND", "cpu")
    monkeypatch.setattr(search_app.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_torch_device("cuda")


def test_requested_rocm_reports_the_requested_backend(monkeypatch) -> None:
    monkeypatch.setattr(search_app, "COMPUTE_BACKEND", "cpu")
    monkeypatch.setattr(search_app.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="AMD ROCm was requested"):
        resolve_torch_device("rocm")


def test_source_url_opens_the_first_page() -> None:
    metadata = {
        "source_id": "manuals/My Manual.pdf",
        "page_start": 12,
        "page_end": 14,
    }
    assert source_url(metadata) == "/sources/manuals/My%20Manual.pdf#page=12"
    assert source_citation(metadata, 1) == (
        "[Source 1 — manuals/My Manual.pdf, page 12]"
        "(/sources/manuals/My%20Manual.pdf#page=12)"
    )


def test_question_link_uses_the_local_viewer_and_escapes_html() -> None:
    metadata = {"source_id": "manual.pdf", "page_start": 12}
    assert source_url(metadata, "maximum speed") == (
        "/viewer/manual.pdf?page=12&q=maximum+speed"
    )
    rendered = source_viewer_html("manual.pdf", 12, "<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "Previous matched region" in rendered
    assert "Copy citation" in rendered
    assert "Copy passage" in rendered
    assert "Show full passage" in rendered
    assert "Save passage or add a note" in rendered
    assert 'data-highlight-mode="subtle"' in rendered
    assert 'id="highlightToggle"' in rendered
    assert "Highlight: Subtle" in rendered
    assert "Highlight: Off" in rendered
    assert "if (stored === 'strong') return 'subtle'" in rendered
    assert "citation-highlight-mode" in rendered
    assert "event.key.toLowerCase() !== 'h'" in rendered
    assert 'data-highlight-mode="off"' in rendered
    assert "highlight-emphasis" not in rendered
    assert "emphasizeHighlights" not in rendered
    assert 'id="textLayer"' in rendered
    assert "/page-text/" in rendered
    assert "Click match to copy" in rendered
    assert "viewerScroll.scrollBy" in rendered
    assert "copyYellowRegionAtPoint" in rendered
    assert "currentPage - 2" in rendered
    assert "thumbnails.replaceChildren(fragment)" in rendered
    assert "pageImage.addEventListener('error'" in rendered


def test_generated_viewer_javascript_has_valid_syntax() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for generated JavaScript validation")
    rendered = source_viewer_html("manual.pdf", 12, "maximum speed")
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


def test_exact_citation_link_includes_the_retrieved_chunk() -> None:
    metadata = {
        "source_id": "manual.pdf",
        "page_start": 12,
        "chunk_id": "chunk-abc",
    }
    assert source_url(metadata, "maximum speed") == (
        "/viewer/manual.pdf?page=12&q=maximum+speed&chunk=chunk-abc"
    )


def test_ranked_sources_receive_previous_and_next_viewer_links() -> None:
    first = candidate(3.0)
    first.chunk_id = "chunk-1"
    first.document.metadata = {
        "source_id": "first.pdf",
        "chunk_id": "chunk-1",
        "page_start": 3,
        "document_title": "First manual",
    }
    second = candidate(2.0)
    second.chunk_id = "chunk-2"
    second.document.metadata = {
        "source_id": "second.pdf",
        "chunk_id": "chunk-2",
        "page_start": 8,
        "document_title": "Second manual",
    }
    third = candidate(1.0)
    third.chunk_id = "chunk-3"
    third.document.metadata = {
        "source_id": "third.pdf",
        "chunk_id": "chunk-3",
        "page_start": 13,
        "document_title": "Third manual",
    }

    token = register_source_navigation("braking distance", [first, second, third])
    assert token
    assert f"nav={token}" in source_url(second.document.metadata, "braking distance")
    assert "at=1" in source_url(second.document.metadata, "braking distance")

    navigation = resolve_source_navigation(
        token,
        1,
        "second.pdf",
        "chunk-2",
    )
    assert navigation["sourcePosition"] == "Source 2 of 3"
    assert navigation["previousSource"]["label"] == "First manual · page 3"
    assert "/viewer/first.pdf?" in navigation["previousSource"]["url"]
    assert navigation["nextSource"]["label"] == "Third manual · page 13"
    assert "/viewer/third.pdf?" in navigation["nextSource"]["url"]


def test_viewer_renders_source_navigation_controls() -> None:
    rendered = source_viewer_html(
        "second.pdf",
        8,
        "braking distance",
        {
            "previousSource": {"url": "/viewer/first.pdf?page=3", "label": "First"},
            "nextSource": {"url": "/viewer/third.pdf?page=13", "label": "Third"},
            "sourcePosition": "Source 2 of 3",
        },
    )
    assert "← Previous source" in rendered
    assert "Next source →" in rendered
    assert "Source 2 of 3" in rendered
    assert 'href="/viewer/first.pdf?page=3"' in rendered
    assert 'href="/viewer/third.pdf?page=13"' in rendered


def test_docling_bottom_left_bbox_becomes_css_percentages() -> None:
    rectangle = bbox_to_percentages(
        {"left": 10, "top": 180, "right": 60, "bottom": 160},
        100,
        200,
    )
    assert rectangle == {
        "left": pytest.approx(10),
        "top": pytest.approx(10),
        "width": pytest.approx(50),
        "height": pytest.approx(10),
    }


def test_top_left_bbox_origin_is_also_supported() -> None:
    rectangle = bbox_to_percentages(
        {
            "left": 10,
            "top": 20,
            "right": 60,
            "bottom": 40,
            "coord_origin": "TOPLEFT",
        },
        100,
        200,
    )
    assert rectangle is not None
    assert rectangle["top"] == pytest.approx(10)
    assert rectangle["height"] == pytest.approx(10)


def test_pdf_characters_are_grouped_into_positioned_selectable_lines() -> None:
    class FakeTextPage:
        text = "Hi\r\nBye"
        boxes = {
            0: (10, 160, 20, 180),
            1: (20, 160, 30, 180),
            4: (10, 120, 20, 140),
            5: (20, 120, 30, 140),
            6: (30, 120, 40, 140),
        }

        def count_chars(self) -> int:
            return len(self.text)

        def get_text_range(self, index: int, count: int) -> str:
            return self.text[index : index + count]

        def get_charbox(self, index: int):
            return self.boxes[index]

    lines = text_lines_from_pdfium_page(FakeTextPage(), 100, 200)
    assert [line["text"] for line in lines] == ["Hi", "Bye"]
    assert lines[0]["left"] == pytest.approx(10)
    assert lines[0]["top"] == pytest.approx(10)
    assert lines[0]["width"] == pytest.approx(20)
    assert lines[0]["height"] == pytest.approx(10)


def test_only_one_child_from_the_same_parent_is_selected() -> None:
    first = candidate(2.0)
    first.document.metadata = {
        "source_id": "manual.pdf",
        "parent_chunk_id": "parent-1",
    }
    second = candidate(1.0)
    second.document.metadata = {
        "source_id": "manual.pdf",
        "parent_chunk_id": "parent-1",
    }
    third = candidate(0.0)
    third.document.page_content = "different technical passage"
    third.document.metadata = {
        "source_id": "manual.pdf",
        "parent_chunk_id": "parent-2",
    }
    selected = select_results([first, second, third])
    assert selected == [first, third]


def test_results_are_diversified_across_source_pages() -> None:
    first = candidate(2.0)
    first.document.metadata = {"source_id": "manual.pdf", "page_start": 12}
    second = candidate(1.0)
    second.document.page_content = "another passage on the same page"
    second.document.metadata = {"source_id": "manual.pdf", "page_start": 12}
    third = candidate(0.0)
    third.document.page_content = "passage from the next page"
    third.document.metadata = {"source_id": "manual.pdf", "page_start": 13}
    selected = select_results([first, second, third])
    assert selected == [first, third]


def test_additional_results_exclude_selected_parents_and_keep_new_context() -> None:
    first = candidate(2.0)
    first.document.metadata = {
        "source_id": "manual.pdf",
        "parent_chunk_id": "parent-1",
    }
    repeated_parent = candidate(1.0)
    repeated_parent.document.page_content = "overlapping child from first parent"
    repeated_parent.document.metadata = {
        "source_id": "manual.pdf",
        "parent_chunk_id": "parent-1",
    }
    new_context = candidate(0.0)
    new_context.document.page_content = "independent context about another requirement"
    new_context.document.metadata = {
        "source_id": "manual.pdf",
        "parent_chunk_id": "parent-2",
    }
    additional = select_additional_results(
        [first, repeated_parent, new_context],
        [first],
    )
    assert additional == [new_context]


def test_additional_button_label_reflects_toggle_state() -> None:
    assert additional_button_label(3) == "Show 3 more reranked sources"
    assert additional_button_label(1, expanded=True) == "Hide 1 more reranked source"


def test_gradio_search_exposes_cached_additional_results(monkeypatch) -> None:
    extra = candidate(0.0)
    monkeypatch.setattr(
        search_app,
        "search_with_additional",
        lambda question, source: ("primary", "additional", [], [extra], {}),
    )
    primary, additional_update, button_update, expanded, count = gradio_search(
        "question",
        "",
    )
    assert primary == "primary"
    assert additional_update["value"] == "additional"
    assert additional_update["visible"] is False
    assert button_update["visible"] is True
    assert button_update["value"] == "Show 1 more reranked source"
    assert expanded is False
    assert count == 1

    panel_update, hide_button_update, next_expanded = toggle_additional_sources(
        expanded,
        count,
    )
    assert panel_update["visible"] is True
    assert hide_button_update["value"] == "Hide 1 more reranked source"
    assert next_expanded is True


def test_source_path_is_restricted_to_pdf_directory(tmp_path, monkeypatch) -> None:
    pdf_root = tmp_path / "docs"
    pdf_root.mkdir()
    pdf_path = pdf_root / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    outside = tmp_path / "private.pdf"
    outside.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(search_app, "PDF_DIR", pdf_root)

    assert resolve_source_path("manual.pdf") == pdf_path.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_source_path("../private.pdf")
    with pytest.raises(FileNotFoundError):
        resolve_source_path("notes.txt")
