from langchain_core.documents import Document

from evaluate import ndcg
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
