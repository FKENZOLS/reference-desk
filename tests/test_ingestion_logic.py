from ingest import (
    ChunkRecord,
    convert_page_range_resilient,
    ids_fingerprint,
    is_cuda_out_of_memory,
    merge_short_chunks,
    report_cuda_headroom,
    split_text_for_retrieval,
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
