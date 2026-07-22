import json
import threading
import time

from ingest import (
    AdaptivePageWindow,
    ChunkRecord,
    EmbeddingCacheStats,
    MachineResources,
    convert_page_range_resilient,
    docling_safe_pdf_path,
    embed_documents_in_batches,
    ids_fingerprint,
    infer_document_title,
    infer_retrieval_unit_type,
    ingestion_fingerprint,
    index_commit_window,
    is_cuda_out_of_memory,
    merge_short_chunks,
    missing_extractable_pages,
    page_coverage_warnings,
    prune_removed_sources,
    recommended_docling_batch_size,
    recommended_docling_threads,
    recommended_page_window,
    report_cuda_headroom,
    resolve_ingestion_tuning,
    split_text_for_retrieval,
    split_structural_record,
    source_is_current,
    stitch_window_headings,
)

from types import SimpleNamespace

import pytest
import ingest
from corpus_scale import debug_artifact_paths


class FakeRuntime:
    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())


def record(text: str, label: str, tokens: int) -> ChunkRecord:
    return ChunkRecord(
        original_indices=[0],
        raw_text=text,
        content=text,
        headings=[],
        labels=[label],
        locations=[],
        token_count=tokens,
    )


def test_consecutive_prefixes_keep_original_order() -> None:
    body = " ".join(f"word{index}" for index in range(30))
    merged = merge_short_chunks(
        [
            record("Document title", "title", 2),
            record("Section name", "section_header", 2),
            record(body, "text", 30),
        ],
        FakeRuntime(),
    )
    assert len(merged) == 1
    assert merged[0].content.startswith(
        "Document title\n\nSection name\n\n"
    )


def title_record(
    text: str,
    *,
    page: int,
    label: str = "text",
    headings: list[str] | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        original_indices=[0],
        raw_text=text,
        content=text,
        headings=list(headings or []),
        labels=[label],
        locations=[{"page": page, "label": label}],
        token_count=len(text.split()),
    )


def test_document_title_uses_page_one_instead_of_later_heading(tmp_path) -> None:
    records = [
        title_record("AI Engineering\nChip Huyen", page=1),
        title_record(
            "Praise for AI Engineering",
            page=2,
            label="title",
            headings=["Praise for AI Engineering"],
        ),
    ]

    assert infer_document_title(records, tmp_path / "fallback.pdf") == "AI Engineering"


def test_document_title_prefers_explicit_page_one_title(tmp_path) -> None:
    records = [
        title_record("Publisher catalogue", page=1),
        title_record("The Proper Book Title", page=1, label="title"),
    ]

    assert infer_document_title(records, tmp_path / "fallback.pdf") == "The Proper Book Title"


def test_document_title_falls_back_to_filename_not_later_chapter(tmp_path) -> None:
    records = [
        title_record(
            "Chapter 1: An Incorrect Document Name",
            page=3,
            label="title",
            headings=["Chapter 1: An Incorrect Document Name"],
        )
    ]

    assert infer_document_title(records, tmp_path / "Correct_Book_Name.pdf") == "Correct Book Name"


def test_later_title_label_cannot_leak_through_cross_page_merge(tmp_path) -> None:
    merged = title_record(
        "AI Engineering\nPraise for AI Engineering",
        page=1,
        label="text",
        headings=["Praise for AI Engineering"],
    )
    merged.labels.append("title")
    merged.locations.append({"page": 2, "label": "title"})

    assert infer_document_title([merged], tmp_path / "fallback.pdf") == "AI Engineering"


def test_page_window_inherits_previous_section_context() -> None:
    inherited, active = stitch_window_headings(
        [record("continued requirement", "text", 2)],
        ["5 Performance", "5.1 Speed"],
        FakeRuntime(),
    )
    assert active == ["5 Performance", "5.1 Speed"]
    assert inherited[0].headings == active
    assert inherited[0].content.startswith(
        "Section context: 5 Performance > 5.1 Speed"
    )


def test_parent_text_is_split_into_bounded_overlapping_children() -> None:
    text = " ".join(f"word{index}" for index in range(18))
    children = split_text_for_retrieval(
        text,
        FakeRuntime(),
        max_tokens=8,
        overlap_tokens=2,
    )
    assert len(children) == 3
    assert all(len(child.split()) <= 8 for child in children)
    assert set(children[0].split()[-2:]) <= set(children[1].split()[:2])


def test_table_children_repeat_column_headers_and_keep_structural_type() -> None:
    rows = [
        "| Requirement | Description |",
        "| --- | --- |",
        *[
            f"| R-{index} | " + " ".join(f"detail{index}_{word}" for word in range(120)) + " |"
            for index in range(4)
        ],
    ]
    item = record("\n".join(rows), "table", 500)

    children = split_structural_record(item, FakeRuntime())

    assert infer_retrieval_unit_type(item) == "table"
    assert len(children) > 1
    assert all(unit_type == "table" for _, unit_type, _ in children)
    assert all("| Requirement | Description |" in text for text, _, _ in children)
    assert all(header == "| Requirement | Description |" for _, _, header in children)


def test_incomplete_manifest_never_skips_source() -> None:
    ids = ["a", "b"]
    metadata = [
        {"file_hash": "file", "ingestion_fingerprint": "ingestion"},
        {"file_hash": "file", "ingestion_fingerprint": "ingestion"},
    ]
    assert not source_is_current(
        ids,
        metadata,
        "file",
        "ingestion",
        {"complete": False},
    )
    assert source_is_current(
        ids,
        metadata,
        "file",
        "ingestion",
        {
            "complete": True,
            "file_hash": "file",
            "ingestion_fingerprint": "ingestion",
            "chunk_count": 2,
            "ids_fingerprint": ids_fingerprint(ids),
        },
    )


def test_failed_page_window_is_split_until_it_succeeds(tmp_path) -> None:
    class FakeConverter:
        def convert(self, path, page_range, raises_on_error):
            start, end = page_range
            if end - start + 1 > 2:
                return SimpleNamespace(
                    document=SimpleNamespace(pages={start: object()}),
                    errors=[SimpleNamespace(error_message="std::bad_alloc")],
                )
            return SimpleNamespace(
                document=SimpleNamespace(
                    pages={page: object() for page in range(start, end + 1)}
                ),
                errors=[],
            )

    results = convert_page_range_resilient(
        FakeConverter(),
        tmp_path / "large.pdf",
        1,
        5,
    )
    pages = sorted(page for result in results for page in result.document.pages)
    assert pages == [1, 2, 3, 4, 5]


def test_allocation_failure_splits_before_parser_fallback(tmp_path) -> None:
    calls = []
    memory_pressure = []

    class PrimaryConverter:
        def convert(self, path, page_range, raises_on_error):
            calls.append(page_range)
            start, end = page_range
            if end - start + 1 > 2:
                return SimpleNamespace(
                    document=SimpleNamespace(pages={}),
                    errors=[SimpleNamespace(error_message="std::bad_alloc")],
                )
            return SimpleNamespace(
                document=SimpleNamespace(
                    pages={page: object() for page in range(start, end + 1)}
                ),
                errors=[],
            )

    class UnexpectedFallback:
        def convert(self, *args, **kwargs):
            raise AssertionError("fallback must not replace a parser that can split")

    results = convert_page_range_resilient(
        PrimaryConverter(),
        tmp_path / "large.pdf",
        1,
        6,
        fallback_converter=UnexpectedFallback(),
        on_memory_pressure=memory_pressure.append,
    )

    assert calls[0] == (1, 6)
    assert memory_pressure
    assert memory_pressure[0] == 6
    assert sorted(page for result in results for page in result.document.pages) == list(
        range(1, 7)
    )


def test_docling_tuning_is_conservative_and_hardware_aware() -> None:
    six_gb_machine = MachineResources(
        cpu_count=16,
        system_available_mb=15_315,
        system_total_mb=32_654,
        accelerator_available_mb=5_134,
        accelerator_total_mb=6_143,
    )
    tuning = resolve_ingestion_tuning(six_gb_machine)
    assert tuning.page_window == 6
    assert tuning.page_batch_size == 2
    assert tuning.model_batch_size == 2
    assert tuning.queue_max_size == 12
    assert tuning.num_threads == 4

    assert recommended_docling_batch_size(None) == 1
    assert recommended_docling_batch_size(3_757, 14_000) == 1
    assert recommended_docling_batch_size(5_134, 14_000) == 2
    assert recommended_docling_batch_size(14_000, 32_000) == 4
    assert recommended_page_window(5_134, 15_315) == 6
    assert recommended_page_window(14_000, 32_000) == 16
    assert recommended_docling_threads(2, 16_000) == 1
    assert recommended_docling_threads(8, 4_096) == 1
    assert recommended_docling_threads(64, 16_000) == 4


def test_explicit_ingestion_tuning_overrides_automatic_values(monkeypatch) -> None:
    resources = MachineResources(16, 32_000, 64_000, 14_000, 24_000)
    monkeypatch.setattr(ingest, "PDF_PAGE_WINDOW", 7)
    monkeypatch.setattr(ingest, "DOCLING_PAGE_BATCH_SIZE", 2)
    monkeypatch.setattr(ingest, "DOCLING_MODEL_BATCH_SIZE", 3)
    monkeypatch.setattr(ingest, "DOCLING_QUEUE_MAX_SIZE", 9)
    monkeypatch.setattr(ingest, "DOCLING_NUM_THREADS", 2)

    tuning = resolve_ingestion_tuning(resources)

    assert (
        tuning.page_window,
        tuning.page_batch_size,
        tuning.model_batch_size,
        tuning.queue_max_size,
        tuning.num_threads,
    ) == (7, 2, 3, 9, 2)


def test_machine_performance_tuning_is_not_part_of_semantic_fingerprint(
    monkeypatch,
) -> None:
    original = ingestion_fingerprint("tokenizer", "auto")
    monkeypatch.setattr(ingest, "PDF_PAGE_WINDOW", 2)
    monkeypatch.setattr(ingest, "DOCLING_PAGE_BATCH_SIZE", 1)
    monkeypatch.setattr(ingest, "DOCLING_MODEL_BATCH_SIZE", 1)
    monkeypatch.setattr(ingest, "DOCLING_QUEUE_MAX_SIZE", 4)
    monkeypatch.setattr(ingest, "DOCLING_NUM_THREADS", 1)

    assert ingestion_fingerprint("tokenizer", "auto") == original


def test_page_window_stays_reduced_after_memory_pressure() -> None:
    window = AdaptivePageWindow(10)
    window.observe_memory_pressure(10)
    assert window.current == 5
    window.observe_memory_pressure(3)
    assert window.current == 5
    window.observe_memory_pressure(5)
    assert window.current == 2


def test_only_unrepresented_source_pages_with_text_require_recovery(
    tmp_path,
    monkeypatch,
) -> None:
    records = [title_record("represented", page=2)]
    monkeypatch.setattr(
        ingest,
        "extractable_text_pages",
        lambda pdf_path, page_numbers: {1, 2, 4} & set(page_numbers),
    )

    assert missing_extractable_pages(tmp_path / "manual.pdf", records, 4) == [1, 4]


def test_page_coverage_requires_an_explicit_per_document_override() -> None:
    with pytest.raises(RuntimeError, match="missing from pages 2, 7"):
        page_coverage_warnings(
            [2, 7],
            allow_incomplete_index=False,
        )

    warnings = page_coverage_warnings(
        [2, 7],
        allow_incomplete_index=True,
    )

    assert len(warnings) == 1
    assert "explicit user override" in warnings[0]
    assert "pages 2, 7" in warnings[0]


def test_pruning_removed_source_also_removes_its_debug_exports(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeCollection:
        deleted = []

        @staticmethod
        def get(include):
            return {"metadatas": [{"source_id": "deleted.pdf"}]}

        def delete(self, *, where):
            self.deleted.append(where)

    markdown, chunks = debug_artifact_paths(tmp_path / "debug", "deleted.pdf")
    markdown.parent.mkdir(parents=True)
    markdown.write_text("old", encoding="utf-8")
    chunks.write_text("old", encoding="utf-8")
    manifest = {"sources": {"deleted.pdf": {"complete": True}}}
    lexical_deletions = []
    monkeypatch.setattr(ingest, "DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(
        ingest,
        "delete_lexical_source",
        lambda path, source_id: lexical_deletions.append(source_id),
    )
    monkeypatch.setattr(ingest, "save_manifest", lambda value: None)

    collection = FakeCollection()
    removed = prune_removed_sources(collection, manifest, set())

    assert removed == 1
    assert collection.deleted == [{"source_id": "deleted.pdf"}]
    assert lexical_deletions == ["deleted.pdf"]
    assert manifest["sources"] == {}
    assert not markdown.exists()
    assert not chunks.exists()


def test_embedding_cache_reuses_exact_title_aware_prompt(tmp_path, monkeypatch) -> None:
    from langchain_core.documents import Document

    class FakeEmbeddings:
        def __init__(self):
            self.calls = 0

        def embed_documents_with_titles(self, texts, titles):
            self.calls += 1
            return [
                [float(len(text)), float(len(title or ""))]
                for text, title in zip(texts, titles, strict=True)
            ]

    documents = [
        Document(page_content="alpha", metadata={"document_title": "Manual"}),
        Document(page_content="beta", metadata={"document_title": "Manual"}),
    ]
    cache_path = tmp_path / "embedding-cache.sqlite3"
    monkeypatch.setattr(ingest, "embedding_fingerprint", lambda: "model-revision")
    first_embeddings = FakeEmbeddings()
    first_stats = EmbeddingCacheStats()
    first = embed_documents_in_batches(
        documents,
        first_embeddings,
        cache_path=cache_path,
        stats=first_stats,
    )
    second_embeddings = FakeEmbeddings()
    second_stats = EmbeddingCacheStats()
    second = embed_documents_in_batches(
        documents,
        second_embeddings,
        cache_path=cache_path,
        stats=second_stats,
    )

    assert first == second
    assert first_embeddings.calls == 1
    assert second_embeddings.calls == 0
    assert (first_stats.hits, first_stats.misses) == (0, 2)
    assert (second_stats.hits, second_stats.misses) == (2, 0)

    changed_title = [
        Document(page_content="alpha", metadata={"document_title": "New title"})
    ]
    title_embeddings = FakeEmbeddings()
    title_stats = EmbeddingCacheStats()
    embed_documents_in_batches(
        changed_title,
        title_embeddings,
        cache_path=cache_path,
        stats=title_stats,
    )
    assert title_embeddings.calls == 1
    assert title_stats.misses == 1

    monkeypatch.setattr(ingest, "embedding_fingerprint", lambda: "new-model-revision")
    revised_embeddings = FakeEmbeddings()
    revised_stats = EmbeddingCacheStats()
    embed_documents_in_batches(
        documents,
        revised_embeddings,
        cache_path=cache_path,
        stats=revised_stats,
    )
    assert revised_embeddings.calls == 1
    assert (revised_stats.hits, revised_stats.misses) == (0, 2)


def test_unicode_pdf_name_is_staged_as_temporary_ascii_path(tmp_path) -> None:
    original_path = tmp_path / "Anna’s Archive.pdf"
    original_path.write_bytes(b"example-pdf-bytes")

    with docling_safe_pdf_path(original_path) as safe_path:
        assert safe_path != original_path.resolve()
        assert str(safe_path).isascii()
        assert safe_path.suffix == ".pdf"
        assert safe_path.read_bytes() == original_path.read_bytes()
        staged_path = safe_path

    assert original_path.exists()
    assert not staged_path.exists()


def test_pdfium_converter_is_used_after_default_parser_failure(tmp_path) -> None:
    class FailingConverter:
        def convert(self, path, page_range, raises_on_error):
            return SimpleNamespace(
                document=SimpleNamespace(pages={}),
                errors=[SimpleNamespace(error_message="backend could not parse")],
            )

    class FallbackConverter:
        def __init__(self):
            self.calls = 0

        def convert(self, path, page_range, raises_on_error):
            self.calls += 1
            start, end = page_range
            return SimpleNamespace(
                document=SimpleNamespace(
                    pages={page: object() for page in range(start, end + 1)}
                ),
                errors=[],
            )

    fallback = FallbackConverter()
    results = convert_page_range_resilient(
        FailingConverter(),
        tmp_path / "manual.pdf",
        1,
        3,
        fallback_converter=fallback,
        source_name="original’s title.pdf",
    )

    assert fallback.calls == 1
    assert sorted(results[0].document.pages) == [1, 2, 3]


def test_cuda_oom_is_recognized_and_reports_recovery(tmp_path) -> None:
    class OomConverter:
        def convert(self, path, page_range, raises_on_error):
            raise RuntimeError("CUDA error: out of memory")

    assert is_cuda_out_of_memory("CUDA error: out of memory")
    assert is_cuda_out_of_memory("HIP out of memory on AMD device")
    assert is_cuda_out_of_memory("HSA_STATUS_ERROR_OUT_OF_RESOURCES")
    assert not is_cuda_out_of_memory("std::bad_alloc")
    with pytest.raises(RuntimeError, match=r"ollama stop qwen3-embedding:0\.6b"):
        convert_page_range_resilient(
            OomConverter(),
            tmp_path / "large.pdf",
            1,
            1,
        )


def test_low_cuda_headroom_stops_before_conversion(monkeypatch, capsys) -> None:
    monkeypatch.setattr("ingest.DOCLING_DEVICE", "cuda")
    monkeypatch.setattr("ingest.CUDA_HEADROOM_WARNING_MB", 3500)
    monkeypatch.setattr("ingest.ALLOW_LOW_CUDA_HEADROOM", False)
    monkeypatch.setattr("ingest.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "ingest.torch.cuda.mem_get_info",
        lambda: (2800 * 1024 * 1024, 6144 * 1024 * 1024),
    )
    with pytest.raises(RuntimeError, match=r"ollama stop qwen3-embedding:0\.6b"):
        report_cuda_headroom()
    assert "RAG_GPU_HEADROOM_GUARD_FAILED free_mib=2800 required_mib=3500" in capsys.readouterr().out


def test_index_commit_window_waits_for_the_app_permission(tmp_path, monkeypatch) -> None:
    gate = tmp_path / "commit-gate.json"
    events = []
    entered = threading.Event()
    monkeypatch.setattr("ingest.token_urlsafe", lambda _: "commit-token")
    monkeypatch.setattr(
        "ingest.emit_corpus_event",
        lambda event, **values: events.append((event, values)),
    )

    def commit() -> None:
        with index_commit_window(gate, "manual.pdf"):
            entered.set()

    worker = threading.Thread(target=commit)
    worker.start()
    deadline = time.monotonic() + 1
    while not events and time.monotonic() < deadline:
        time.sleep(0.01)
    assert events == [("commit_requested", {"source_id": "manual.pdf", "token": "commit-token"})]
    assert not entered.is_set()

    gate.write_text(json.dumps({"token": "commit-token"}), encoding="utf-8")
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert entered.is_set()
    assert events[-1] == (
        "commit_finished",
        {"source_id": "manual.pdf", "token": "commit-token", "status": "complete"},
    )


def test_failed_pdf_does_not_stop_remaining_queue(tmp_path, monkeypatch) -> None:
    pdf_dir = tmp_path / "docs"
    pdf_dir.mkdir()
    (pdf_dir / "broken.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (pdf_dir / "working.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    processed = []
    events = []

    args = SimpleNamespace(
        source=[],
        queue_managed=False,
        prune=False,
        queue_control=None,
        ocr=False,
        no_auto_ocr=False,
        force=False,
        commit_gate=None,
    )
    monkeypatch.setattr(ingest, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(ingest, "DB_DIR", tmp_path / "db")
    monkeypatch.setattr(ingest, "DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(ingest, "EXPORT_DEBUG_FILES", False)
    monkeypatch.setattr(ingest, "parse_args", lambda: args)
    monkeypatch.setattr(ingest, "resolved_embedding_revision", lambda: "test-revision")
    monkeypatch.setattr(ingest, "report_cuda_headroom", lambda *_: None)
    monkeypatch.setattr(ingest, "create_collection", lambda: object())
    monkeypatch.setattr(ingest, "load_manifest", lambda: {})
    monkeypatch.setattr(ingest, "create_converter", lambda **kwargs: object())
    monkeypatch.setattr(
        ingest,
        "create_chunking_runtime",
        lambda: SimpleNamespace(tokenizer_name="test-tokenizer"),
    )
    monkeypatch.setattr(ingest, "create_embeddings", lambda: object())
    monkeypatch.setattr(ingest, "synchronize_lexical_index", lambda collection: None)
    monkeypatch.setattr(
        ingest,
        "emit_corpus_event",
        lambda event, **values: events.append((event, values)),
    )

    def process_pdf(**values):
        name = values["pdf_path"].name
        processed.append(name)
        if name == "broken.pdf":
            raise ValueError("damaged cross-reference table")
        return SimpleNamespace(status="indexed", chunks=3)

    monkeypatch.setattr(ingest, "process_pdf", process_pdf)

    with pytest.raises(SystemExit) as stopped:
        ingest.main()

    assert stopped.value.code == 1
    assert processed == ["broken.pdf", "working.pdf"]
    assert any(event == "failed" and values["source_id"] == "broken.pdf" for event, values in events)
    assert any(event == "completed" and values["source_id"] == "working.pdf" for event, values in events)
