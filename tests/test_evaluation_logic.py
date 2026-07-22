from langchain_core.documents import Document

from evaluate import hard_negative_hit_rate, labeled_recall, ndcg, relevance_targets
from search_app import Candidate


def page_candidate(chunk_id: str, page: int) -> Candidate:
    return Candidate(
        document=Document(
            page_content=f"content for {chunk_id}",
            metadata={"source_id": "manual.pdf", "page_start": page, "page_end": page},
        ),
        chunk_id=chunk_id,
    )


def test_ndcg_counts_one_labeled_page_only_once() -> None:
    case = {
        "relevant_locations": [{"source_id": "manual.pdf", "page": 12}],
    }
    candidates = [page_candidate("first", 12), page_candidate("second", 12)]
    assert ndcg(candidates, case, k=5) == 1.0


def test_mixed_relevance_identifiers_are_grouped_as_passage_targets() -> None:
    case = {
        "relevant_targets": [
            {
                "chunk_id": "old-chunk-id",
                "source_id": "manual.pdf",
                "page": 12,
            },
            {"source_id": "manual.pdf", "page": 44},
        ],
    }
    candidates = [page_candidate("new-chunk-id", 12), page_candidate("other", 44)]

    assert len(relevance_targets(case)) == 2
    assert labeled_recall(candidates, case) == 1.0
    assert ndcg(candidates, case, k=5) == 1.0


def test_legacy_parallel_id_and_location_describe_one_passage() -> None:
    case = {
        "relevant_chunk_ids": ["chunk-12"],
        "relevant_locations": [{"source_id": "manual.pdf", "page": 12}],
    }

    assert len(relevance_targets(case)) == 1
    assert labeled_recall([page_candidate("chunk-12", 12)], case) == 1.0


def test_hard_negative_rate_uses_only_labeled_cases() -> None:
    rows = [
        {"has_hard_negatives": False, "hard_negative_hit": False},
        {"has_hard_negatives": True, "hard_negative_hit": True},
    ]

    assert hard_negative_hit_rate(rows) == 1.0
