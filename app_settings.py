"""Environment-backed settings for the search application.

Shared storage and embedding settings remain in ``rag_common``. This module
owns only search, reranking, and web-server configuration so that
``search_app`` can focus on application behavior.
"""

from __future__ import annotations

import hashlib
import json
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

# Local reranking. ``auto`` selects Qwen's generative yes/no scorer for
# Qwen3-Reranker models and the former sequence-classifier path for BGE.
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL",
    "Qwen/Qwen3-Reranker-0.6B",
)
RERANKER_REVISION = os.environ.get("RAG_RERANKER_REVISION", "main")
RERANKER_BACKEND = os.environ.get("RAG_RERANKER_BACKEND", "auto")
RERANKER_USE_AUTH = env_flag("RAG_RERANKER_USE_AUTH", False)
RERANKER_PROMPT_VERSION = "qwen3-reranker-reference-v1"
RERANK_INSTRUCTION = os.environ.get(
    "RAG_RERANK_INSTRUCTION",
    (
        "Given a technical-document search query, determine whether the "
        "document passage directly defines, specifies, explains, or constrains "
        "the requested subject. Prefer exact, citable reference evidence."
    ),
)
RERANK_MAX_LENGTH = env_int("RAG_RERANK_MAX_LENGTH", 512)
RERANK_BATCH_SIZE = env_int("RAG_RERANK_BATCH_SIZE", 8)
RERANK_TOP_N = env_int("RAG_RERANK_TOP_N", 5)
ADDITIONAL_RESULTS = env_int("RAG_ADDITIONAL_RESULTS", 15)
MAX_RESULTS_PER_SECTION = env_int("RAG_MAX_RESULTS_PER_SECTION", 2)
MAX_RESULTS_PER_PAGE = env_int("RAG_MAX_RESULTS_PER_PAGE", 1)
MAX_RESULT_TEXT_SIMILARITY = env_float("RAG_MAX_RESULT_TEXT_SIMILARITY", 0.88)
RERANK_USE_FP16 = env_flag("RAG_RERANK_FP16", True)


def resolve_reranker_backend(
    model_name: str = RERANKER_MODEL,
    configured: str = RERANKER_BACKEND,
) -> str:
    value = configured.strip().lower().replace("_", "-")
    if value in {"", "auto"}:
        return "qwen-causal-lm" if "qwen3-reranker" in model_name.lower() else "classifier"
    aliases = {
        "qwen": "qwen-causal-lm",
        "qwen-causal": "qwen-causal-lm",
        "qwen-causal-lm": "qwen-causal-lm",
        "classifier": "classifier",
        "cross-encoder": "classifier",
        "bge": "classifier",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise ValueError(
            "RAG_RERANKER_BACKEND must be auto, qwen-causal-lm, or classifier."
        ) from error


def reranker_configuration() -> dict[str, object]:
    return {
        "model": RERANKER_MODEL,
        "revision": RERANKER_REVISION,
        "backend": resolve_reranker_backend(),
        "max_length": RERANK_MAX_LENGTH,
        "prompt_version": RERANKER_PROMPT_VERSION,
        "instruction": RERANK_INSTRUCTION,
    }


def reranker_fingerprint() -> str:
    payload = json.dumps(
        reranker_configuration(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# Raw-logit relevance gate; disabled until calibrated on a labeled benchmark.
MIN_BEST_RERANK_LOGIT = env_float("RAG_MIN_BEST_RERANK_LOGIT", -2.2)
MIN_RESULT_RERANK_LOGIT = env_float("RAG_MIN_RESULT_RERANK_LOGIT", -3.0)
ENABLE_RELEVANCE_GATE = env_flag("RAG_ENABLE_RELEVANCE_GATE", False)

# Feedback calibration. The automatic gate stays off until both classes have
# enough explicit human judgments to avoid fitting a threshold to a tiny sample.
QUALITY_MIN_POSITIVE_LABELS = env_int("RAG_QUALITY_MIN_POSITIVE_LABELS", 20)
QUALITY_MIN_NEGATIVE_LABELS = env_int("RAG_QUALITY_MIN_NEGATIVE_LABELS", 20)
QUALITY_MIN_RECALL = env_float("RAG_QUALITY_MIN_RECALL", 0.90)

# Concurrent search and ingestion. Auto mode measures *free* VRAM after the
# search runtime is loaded; total VRAM alone does not tell us whether both jobs
# fit at a particular moment.
SEARCH_DURING_INGESTION = os.environ.get(
    "RAG_SEARCH_DURING_INGESTION",
    "auto",
).strip().lower()
if SEARCH_DURING_INGESTION not in {"auto", "never", "always"}:
    raise ValueError(
        "RAG_SEARCH_DURING_INGESTION must be auto, never, or always."
    )
DOCLING_GPU_HEADROOM_MB = env_int(
    "RAG_GPU_HEADROOM_WARNING_MB",
    env_int("RAG_CUDA_HEADROOM_WARNING_MB", 3500),
)
CONCURRENT_QUERY_RESERVE_MB = env_int(
    "RAG_CONCURRENT_QUERY_RESERVE_MB",
    768,
)

# Presentation, diagnostics, and local server
MAX_CHARS_PER_RESULT = env_int("RAG_MAX_RESULT_CHARS", 6000)
DEBUG_RETRIEVAL = env_flag("RAG_DEBUG_RETRIEVAL", False)
OPEN_BROWSER = env_flag("RAG_OPEN_BROWSER", True)
SERVER_HOST = os.environ.get("RAG_SERVER_HOST", "127.0.0.1")
SERVER_PORT = env_int("RAG_SERVER_PORT", 7860)
