import json
import threading
import time

from ingest import (
    ChunkRecord,
    convert_page_range_resilient,
    docling_safe_pdf_path,
    ids_fingerprint,
    infer_retrieval_unit_type,
    index_commit_window,
    is_cuda_out_of_memory,
    merge_short_chunks,
    report_cuda_headroom,
    split_text_for_retrieval,
    split_structural_record,
    source_is_current,
    stitch_window_headings,
)

from types import SimpleNamespace

import pytest


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


def test_low_cuda_headroom_stops_before_conversion(monkeypatch) -> None:
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
