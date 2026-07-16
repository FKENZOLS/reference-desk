"""Environment-backed settings for the search application.

Shared storage and embedding settings remain in ``rag_common``. This module
owns only search, reranking, and web-server configuration so that
``search_app`` can focus on application behavior.
"""

from __future__ import annotations

import os


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_flag(name: str, default: bool) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback) == "1"


# Retrieval and fusion
DENSE_CANDIDATES = env_int("RAG_DENSE_CANDIDATES", 40)
LEXICAL_CANDIDATES = env_int("RAG_LEXICAL_CANDIDATES", 40)
RERANK_CANDIDATES = env_int("RAG_RERANK_CANDIDATES", 20)
RRF_K = env_int("RAG_RRF_K", 60)
RERANK_WEIGHT = env_float("RAG_RERANK_WEIGHT", 0.60)

# Cross-encoder reranking
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)
RERANK_MAX_LENGTH = env_int("RAG_RERANK_MAX_LENGTH", 512)
RERANK_BATCH_SIZE = env_int("RAG_RERANK_BATCH_SIZE", 8)
RERANK_TOP_N = env_int("RAG_RERANK_TOP_N", 5)
ADDITIONAL_RESULTS = env_int("RAG_ADDITIONAL_RESULTS", 15)
MAX_RESULTS_PER_SECTION = env_int("RAG_MAX_RESULTS_PER_SECTION", 2)
MAX_RESULTS_PER_PAGE = env_int("RAG_MAX_RESULTS_PER_PAGE", 1)
MAX_RESULT_TEXT_SIMILARITY = env_float("RAG_MAX_RESULT_TEXT_SIMILARITY", 0.88)
RERANK_USE_FP16 = env_flag("RAG_RERANK_FP16", True)

# Raw-logit relevance gate; disabled until calibrated on a labeled benchmark.
MIN_BEST_RERANK_LOGIT = env_float("RAG_MIN_BEST_RERANK_LOGIT", -2.2)
MIN_RESULT_RERANK_LOGIT = env_float("RAG_MIN_RESULT_RERANK_LOGIT", -3.0)
ENABLE_RELEVANCE_GATE = env_flag("RAG_ENABLE_RELEVANCE_GATE", False)

# Feedback calibration. The automatic gate stays off until both classes have
# enough explicit human judgments to avoid fitting a threshold to a tiny sample.
QUALITY_MIN_POSITIVE_LABELS = env_int("RAG_QUALITY_MIN_POSITIVE_LABELS", 20)
QUALITY_MIN_NEGATIVE_LABELS = env_int("RAG_QUALITY_MIN_NEGATIVE_LABELS", 20)
QUALITY_MIN_RECALL = env_float("RAG_QUALITY_MIN_RECALL", 0.90)

# Presentation, diagnostics, and local server
MAX_CHARS_PER_RESULT = env_int("RAG_MAX_RESULT_CHARS", 6000)
DEBUG_RETRIEVAL = env_flag("RAG_DEBUG_RETRIEVAL", False)
OPEN_BROWSER = env_flag("RAG_OPEN_BROWSER", True)
SERVER_HOST = os.environ.get("RAG_SERVER_HOST", "127.0.0.1")
SERVER_PORT = env_int("RAG_SERVER_PORT", 7860)
