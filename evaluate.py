"""Offline retrieval evaluation for a labeled JSONL query set."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import replace
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
    select_runtime_reranker,
)


CachedCandidates = list[tuple[list[Candidate], float, float]]


def relevance_targets(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical acceptable passages, including alternative identifiers."""

    explicit = case.get("relevant_targets")
    raw_targets: list[dict[str, Any]] = []
    if isinstance(explicit, list) and explicit:
        raw_targets.extend(target for target in explicit if isinstance(target, dict))
    else:
        chunk_ids = [
            str(value)
            for value in case.get("relevant_chunk_ids", [])
            if str(value)
        ]
        locations = [
            value
            for value in case.get("relevant_locations", [])
            if isinstance(value, dict)
        ]
        paired = min(len(chunk_ids), len(locations))
        raw_targets.extend(
            {
                "chunk_id": chunk_ids[index],
                "source_id": str(locations[index].get("source_id") or ""),
                "page": locations[index].get("page"),
            }
            for index in range(paired)
        )
        raw_targets.extend({"chunk_id": value} for value in chunk_ids[paired:])
        raw_targets.extend(dict(value) for value in locations[paired:])

    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for raw in raw_targets:
        chunk_id = str(raw.get("chunk_id") or "")
        source_id = str(raw.get("source_id") or "")
        try:
            page = int(raw["page"]) if raw.get("page") is not None else None
        except (TypeError, ValueError):
            page = None
        key = (chunk_id, source_id, page)
        if key == ("", "", None) or key in seen:
            continue
        seen.add(key)
        targets.append(
            {"chunk_id": chunk_id, "source_id": source_id, "page": page}
        )
    return targets


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
    page_start = int(metadata.get("page_start", metadata.get("page", -1)) or -1)
    page_end = int(metadata.get("page_end", page_start) or page_start)
    source_id = str(metadata.get("source_id", ""))
    for index, target in enumerate(relevance_targets(case)):
        chunk_id = str(target.get("chunk_id") or "")
        target_source = str(target.get("source_id") or "")
        target_page = target.get("page")
        id_match = bool(chunk_id and candidate.chunk_id == chunk_id)
        location_match = bool(
            target_source
            and target_source == source_id
            and target_page is not None
            and page_start <= int(target_page) <= page_end
        )
        if id_match or location_match:
            matched.add(f"target:{index}")
    return matched


def candidate_is_relevant(candidate: Candidate, case: dict[str, Any]) -> bool:
    return bool(matching_relevance_labels(candidate, case))


def reciprocal_rank(candidates: Sequence[Candidate], case: dict[str, Any]) -> float:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate_is_relevant(candidate, case):
            return 1.0 / rank
    return 0.0


def labeled_recall(candidates: Sequence[Candidate], case: dict[str, Any]) -> float:
    targets = relevance_targets(case)
    if not targets:
        return 0.0
    matched: set[str] = set()
    for candidate in candidates:
        matched.update(matching_relevance_labels(candidate, case))
    return len(matched) / len(targets)


def ndcg(candidates: Sequence[Candidate], case: dict[str, Any], k: int = 5) -> float:
    gains: list[float] = []
    seen_labels: set[str] = set()
    for candidate in candidates[:k]:
        new_labels = matching_relevance_labels(candidate, case) - seen_labels
        gains.append(1.0 if new_labels else 0.0)
        seen_labels.update(new_labels)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    relevant_count = max(len(relevance_targets(case)), 1)
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


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    samples: int = 400,
    seed: int = 20260721,
) -> dict[str, float]:
    """Return a deterministic 95% bootstrap interval for a mean metric."""

    clean = [float(value) for value in values]
    mean = statistics.fmean(clean) if clean else 0.0
    if len(clean) < 2:
        return {"mean": mean, "lower": mean, "upper": mean, "confidence": 0.95}
    generator = random.Random(seed)
    estimates = [
        statistics.fmean(generator.choice(clean) for _ in clean)
        for _ in range(max(100, int(samples)))
    ]
    return {
        "mean": mean,
        "lower": percentile(estimates, 0.025),
        "upper": percentile(estimates, 0.975),
        "confidence": 0.95,
    }


def subgroup_metrics(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(field) or "unknown"), []).append(row)
    output: dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        answerable = [row for row in group if row["answerable"]]
        output[name] = {
            "cases": len(group),
            "answerable_cases": len(answerable),
            "ndcg_at_5": (
                statistics.fmean(row["ndcg_at_5"] for row in answerable)
                if answerable
                else 0.0
            ),
            "mrr_at_5": (
                statistics.fmean(row["mrr_at_5"] for row in answerable)
                if answerable
                else 0.0
            ),
            "retrieval_recall": (
                statistics.fmean(row["retrieval_recall"] for row in answerable)
                if answerable
                else 0.0
            ),
            "hard_negative_hit_rate": hard_negative_hit_rate(group),
        }
    return output


def hard_negative_hit_rate(rows: Sequence[dict[str, Any]]) -> float:
    labeled = [row for row in rows if row.get("has_hard_negatives")]
    return (
        statistics.fmean(float(row["hard_negative_hit"]) for row in labeled)
        if labeled
        else 0.0
    )


def retrieve_evaluation_candidates(
    cases: list[dict[str, Any]],
) -> CachedCandidates:
    """Retrieve once so multiple rerankers see identical candidate pools."""

    runtime = get_runtime()
    cached: CachedCandidates = []
    for case in cases:
        pool, dense_seconds, lexical_seconds = retrieve_candidate_pool(
            runtime,
            str(case["query"]).strip(),
            case.get("source_filter") or None,
        )
        cached.append((pool, dense_seconds, lexical_seconds))
    return cached


def evaluate(
    cases: list[dict[str, Any]],
    *,
    reranker_choice: str | None = None,
    candidate_count: int = RERANK_CANDIDATES,
    rerank_weight: float | None = None,
    passage_mode: str = "metadata-child",
    candidate_cache: CachedCandidates | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = get_runtime()
    active_reranker = select_runtime_reranker(runtime, reranker_choice)
    candidate_count = max(1, int(candidate_count))
    cached = candidate_cache or retrieve_evaluation_candidates(cases)
    if len(cached) != len(cases):
        raise ValueError("Candidate cache does not match the benchmark case count")
    rows: list[dict[str, Any]] = []

    for case, cached_item in zip(cases, cached, strict=True):
        query = str(case["query"]).strip()
        started = perf_counter()
        cached_pool, dense_seconds, lexical_seconds = cached_item
        pool = [replace(candidate) for candidate in cached_pool]
        dense_ranked = sorted(
            (candidate for candidate in pool if candidate.vector_rank is not None),
            key=lambda candidate: candidate.vector_rank or 10**9,
        )
        lexical_ranked = sorted(
            (candidate for candidate in pool if candidate.lexical_rank is not None),
            key=lambda candidate: candidate.lexical_rank or 10**9,
        )
        retrieved = pool[:candidate_count]
        rerank_started = perf_counter()
        reranked = rerank_candidates(
            runtime,
            query,
            retrieved,
            rerank_weight=rerank_weight,
            passage_mode=passage_mode,
        )
        rerank_seconds = perf_counter() - rerank_started
        selected = select_results(reranked, active_reranker.choice)
        total_seconds = dense_seconds + lexical_seconds + (perf_counter() - started)

        answerable = bool(case.get("answerable", True))
        dense_recall = labeled_recall(dense_ranked, case)
        lexical_recall = labeled_recall(lexical_ranked, case)
        retrieved_recall = labeled_recall(retrieved, case)
        selected_recall = labeled_recall(selected, case)
        hard_negative_ids = {
            str(value) for value in case.get("hard_negative_chunk_ids", [])
        }
        rows.append(
            {
                "case_id": str(case.get("id") or ""),
                "query": query,
                "answerable": answerable,
                "split": str(case.get("split") or "test"),
                "category": str(case.get("category") or "general"),
                "language": str(case.get("language") or "und"),
                "dense_recall": dense_recall,
                "lexical_recall": lexical_recall,
                "fusion_recall": retrieved_recall,
                "retrieval_recall": retrieved_recall,
                "fusion_recall_at": {
                    str(k): labeled_recall(pool[:k], case)
                    for k in (10, 20, 40, 80)
                },
                "selected_recall": selected_recall,
                "mrr_at_5": reciprocal_rank(reranked[:5], case),
                "ndcg_at_5": ndcg(reranked, case, 5),
                "returned_count": len(selected),
                "correct_rejection": (not answerable and not selected),
                "hard_negative_hit": bool(
                    hard_negative_ids
                    and any(item.chunk_id in hard_negative_ids for item in reranked[:5])
                ),
                "has_hard_negatives": bool(hard_negative_ids),
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
        "reranker_choice": active_reranker.choice,
        "reranker_model": active_reranker.model_name,
        "candidate_count": candidate_count,
        "rerank_weight": rerank_weight,
        "passage_mode": passage_mode,
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
        "hard_negative_cases": sum(
            1 for row in rows if row.get("has_hard_negatives")
        ),
        "hard_negative_hit_rate": hard_negative_hit_rate(rows),
        "fusion_recall_at": {
            str(k): (
                statistics.fmean(
                    row["fusion_recall_at"][str(k)] for row in answerable_rows
                )
                if answerable_rows
                else 0.0
            )
            for k in (10, 20, 40, 80)
        },
        "confidence_intervals": {
            metric: bootstrap_confidence_interval(
                [float(row[metric]) for row in answerable_rows],
                seed=20260721 + index,
            )
            for index, metric in enumerate(
                ("ndcg_at_5", "mrr_at_5", "retrieval_recall", "selected_recall")
            )
        },
        "subgroups": {
            field: subgroup_metrics(rows, field)
            for field in ("split", "category", "language")
        },
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
    parser.add_argument(
        "--reranker",
        choices=("gte", "bge", "both"),
        default="gte",
        help="Reranker to evaluate; 'both' reuses identical retrieval candidates",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=RERANK_CANDIDATES,
        help="Number of fused candidates sent to each reranker",
    )
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=None,
        help="Override the reranker/retrieval blend weight (0 through 1)",
    )
    parser.add_argument(
        "--passage-mode",
        choices=(
            "child",
            "metadata-child",
            "metadata-parent",
            "metadata-child-parent",
        ),
        default="metadata-child",
        help="Text supplied to the reranker for offline input ablations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.benchmark)
    if args.candidate_count < 1:
        raise SystemExit("--candidate-count must be at least 1")
    if args.rerank_weight is not None and not 0 <= args.rerank_weight <= 1:
        raise SystemExit("--rerank-weight must be between 0 and 1")
    candidate_cache = retrieve_evaluation_candidates(cases)
    choices = ("gte", "bge") if args.reranker == "both" else (args.reranker,)
    evaluations: dict[str, dict[str, Any]] = {}
    for choice in choices:
        summary, rows = evaluate(
            cases,
            reranker_choice=choice,
            candidate_count=args.candidate_count,
            rerank_weight=args.rerank_weight,
            passage_mode=args.passage_mode,
            candidate_cache=candidate_cache,
        )
        evaluations[choice] = {"summary": summary, "cases": rows}
    output_payload: dict[str, Any]
    if len(evaluations) == 1:
        output_payload = next(iter(evaluations.values()))
    else:
        output_payload = {"models": evaluations}
    args.output.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {choice: result["summary"] for choice, result in evaluations.items()},
            indent=2,
        )
    )
    print(f"Detailed results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
