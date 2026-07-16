"""Offline retrieval evaluation for a labeled JSONL query set."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from search_app import (
    Candidate,
    RERANK_CANDIDATES,
    get_runtime,
    rerank_candidates,
    retrieve_candidate_pool,
    select_results,
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not str(case.get("query", "")).strip():
                raise ValueError(f"Missing query on line {line_number}")
            cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def matching_relevance_labels(
    candidate: Candidate,
    case: dict[str, Any],
) -> set[str]:
    """Return unique labels matched by a candidate for duplicate-safe metrics."""

    matched: set[str] = set()
    metadata = candidate.document.metadata
    relevant_ids = {str(value) for value in case.get("relevant_chunk_ids", [])}
    if relevant_ids and candidate.chunk_id in relevant_ids:
        matched.add(f"chunk:{candidate.chunk_id}")

    relevant_locations = case.get("relevant_locations", [])
    page_start = int(metadata.get("page_start", metadata.get("page", -1)) or -1)
    page_end = int(metadata.get("page_end", metadata.get("page", -1)) or -1)
    source_id = str(metadata.get("source_id", ""))
    for location in relevant_locations:
        if str(location.get("source_id", source_id)) != source_id:
            continue
        page = int(location.get("page", -1))
        if page >= 0 and page_start <= page <= page_end:
            matched.add(f"location:{source_id}:{page}")
    return matched


def candidate_is_relevant(candidate: Candidate, case: dict[str, Any]) -> bool:
    return bool(matching_relevance_labels(candidate, case))


def reciprocal_rank(candidates: Sequence[Candidate], case: dict[str, Any]) -> float:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate_is_relevant(candidate, case):
            return 1.0 / rank
    return 0.0


def labeled_recall(candidates: Sequence[Candidate], case: dict[str, Any]) -> float:
    relevant_ids = {str(value) for value in case.get("relevant_chunk_ids", [])}
    if relevant_ids:
        found = {candidate.chunk_id for candidate in candidates} & relevant_ids
        return len(found) / len(relevant_ids)

    locations = list(case.get("relevant_locations", []))
    if not locations:
        return 0.0
    matched = 0
    for location in locations:
        location_case = {"relevant_locations": [location]}
        if any(candidate_is_relevant(candidate, location_case) for candidate in candidates):
            matched += 1
    return matched / len(locations)


def ndcg(candidates: Sequence[Candidate], case: dict[str, Any], k: int = 5) -> float:
    gains: list[float] = []
    seen_labels: set[str] = set()
    for candidate in candidates[:k]:
        new_labels = matching_relevance_labels(candidate, case) - seen_labels
        gains.append(1.0 if new_labels else 0.0)
        seen_labels.update(new_labels)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    relevant_count = max(
        len(case.get("relevant_chunk_ids", [])),
        len(case.get("relevant_locations", [])),
        1,
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(relevant_count, k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate(cases: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = get_runtime()
    rows: list[dict[str, Any]] = []

    for case in cases:
        query = str(case["query"]).strip()
        source_filter = case.get("source_filter") or None
        started = perf_counter()
        pool, dense_seconds, lexical_seconds = retrieve_candidate_pool(
            runtime,
            query,
            source_filter,
        )
        dense_ranked = sorted(
            (candidate for candidate in pool if candidate.vector_rank is not None),
            key=lambda candidate: candidate.vector_rank or 10**9,
        )
        lexical_ranked = sorted(
            (candidate for candidate in pool if candidate.lexical_rank is not None),
            key=lambda candidate: candidate.lexical_rank or 10**9,
        )
        retrieved = pool[:RERANK_CANDIDATES]
        rerank_started = perf_counter()
        reranked = rerank_candidates(runtime, query, retrieved)
        rerank_seconds = perf_counter() - rerank_started
        selected = select_results(reranked)
        total_seconds = perf_counter() - started

        answerable = bool(case.get("answerable", True))
        dense_recall = labeled_recall(dense_ranked, case)
        lexical_recall = labeled_recall(lexical_ranked, case)
        retrieved_recall = labeled_recall(retrieved, case)
        selected_recall = labeled_recall(selected, case)
        rows.append(
            {
                "query": query,
                "answerable": answerable,
                "dense_recall": dense_recall,
                "lexical_recall": lexical_recall,
                "fusion_recall": retrieved_recall,
                "retrieval_recall": retrieved_recall,
                "selected_recall": selected_recall,
                "mrr_at_5": reciprocal_rank(reranked[:5], case),
                "ndcg_at_5": ndcg(reranked, case, 5),
                "returned_count": len(selected),
                "correct_rejection": (not answerable and not selected),
                "dense_seconds": dense_seconds,
                "lexical_seconds": lexical_seconds,
                "rerank_seconds": rerank_seconds,
                "total_seconds": total_seconds,
                "truncation_rate": (
                    sum(item.rerank_truncated for item in reranked) / len(reranked)
                    if reranked
                    else 0.0
                ),
                "top_results": [
                    {
                        "chunk_id": item.chunk_id,
                        "source_id": item.document.metadata.get("source_id"),
                        "page_start": item.document.metadata.get("page_start"),
                        "dense_rank": item.vector_rank,
                        "lexical_rank": item.lexical_rank,
                        "fusion_rank": item.retrieval_rank,
                        "rerank_rank": item.rerank_rank,
                        "final_rank": item.final_rank,
                        "rerank_logit": item.rerank_logit,
                        "rerank_probability": item.rerank_probability,
                        "relevant": candidate_is_relevant(item, case),
                    }
                    for item in reranked[:5]
                ],
            }
        )

    answerable_rows = [row for row in rows if row["answerable"]]
    negative_rows = [row for row in rows if not row["answerable"]]
    latencies = [float(row["total_seconds"]) for row in rows]
    summary = {
        "cases": len(rows),
        "answerable_cases": len(answerable_rows),
        "unanswerable_cases": len(negative_rows),
        "dense_recall": (
            statistics.fmean(row["dense_recall"] for row in answerable_rows)
            if answerable_rows
            else 0.0
        ),
        "lexical_recall": (
            statistics.fmean(row["lexical_recall"] for row in answerable_rows)
            if answerable_rows
            else 0.0
        ),
        "fusion_recall": (
            statistics.fmean(row["fusion_recall"] for row in answerable_rows)
            if answerable_rows
            else 0.0
        ),
        "retrieval_recall": (
            statistics.fmean(row["retrieval_recall"] for row in answerable_rows)
            if answerable_rows
            else 0.0
        ),
        "selected_recall": (
            statistics.fmean(row["selected_recall"] for row in answerable_rows)
            if answerable_rows
            else 0.0
        ),
        "mrr_at_5": (
            statistics.fmean(row["mrr_at_5"] for row in answerable_rows)
            if answerable_rows
            else 0.0
        ),
        "ndcg_at_5": (
            statistics.fmean(row["ndcg_at_5"] for row in answerable_rows)
            if answerable_rows
            else 0.0
        ),
        "no_answer_recall": (
            statistics.fmean(row["correct_rejection"] for row in negative_rows)
            if negative_rows
            else 0.0
        ),
        "no_answer_precision": (
            sum(1 for row in rows if not row["answerable"] and not row["returned_count"])
            / sum(1 for row in rows if not row["returned_count"])
            if any(not row["returned_count"] for row in rows)
            else 0.0
        ),
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "mean_truncation_rate": statistics.fmean(
            row["truncation_rate"] for row in rows
        ),
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval.")
    parser.add_argument("benchmark", type=Path, help="Labeled JSONL benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Where to write detailed JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.benchmark)
    summary, rows = evaluate(cases)
    args.output.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Detailed results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
