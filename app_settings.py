"""Environment-backed settings for the search application.

Shared storage and embedding settings remain in ``rag_common``. This module
owns only search, reranking, and web-server configuration so that
``search_app`` can focus on application behavior.

GTE is the default reranker.

Select BGE explicitly with:

    RAG_RERANKER_CHOICE=bge

Select GTE with:

    RAG_RERANKER_CHOICE=gte

The selected preset controls the reranker model and backend. This prevents old
``RAG_RERANKER_MODEL`` or ``RAG_RERANKER_BACKEND`` environment variables from
silently loading a different reranker.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


# =============================================================================
# Environment helpers
# =============================================================================


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable."""

    value = os.environ.get(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be an integer, but received {value!r}."
        ) from error


def env_float(name: str, default: float) -> float:
    """Read a floating-point environment variable."""

    value = os.environ.get(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a number, but received {value!r}."
        ) from error


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Accepted true values:

        1, true, yes, on

    Accepted false values:

        0, false, no, off
    """

    value = os.environ.get(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of: "
        "1, 0, true, false, yes, no, on, or off."
    )


# =============================================================================
# Retrieval and fusion
# =============================================================================


DENSE_CANDIDATES = env_int(
    "RAG_DENSE_CANDIDATES",
    40,
)

LEXICAL_CANDIDATES = env_int(
    "RAG_LEXICAL_CANDIDATES",
    40,
)

RERANK_CANDIDATES = env_int(
    "RAG_RERANK_CANDIDATES",
    20,
)

RRF_K = env_int(
    "RAG_RRF_K",
    60,
)

RERANK_WEIGHT = env_float(
    "RAG_RERANK_WEIGHT",
    0.60,
)


# =============================================================================
# Reranker presets
# =============================================================================


RERANKER_PRESETS: dict[str, dict[str, Any]] = {
    "bge": {
        "label": "BGE",
        "description": "Fast multilingual cross-encoder reranking.",
        "model": "BAAI/bge-reranker-v2-m3",
        # Immutable Hugging Face revisions keep releases reproducible and stop
        # an upstream branch update from changing executable model code.
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "backend": "classifier",
        "prompt_version": "bge-reranker-reference-v1",
        "instruction": (
            "Determine whether the technical-document passage directly "
            "answers, defines, specifies, explains, or constrains the search "
            "query. Prefer exact, specific, and citable evidence."
        ),
        "max_length": 512,
        "batch_size": 8,
        "use_fp16": True,
        "trust_remote_code": False,
    },
    "gte": {
        "label": "GTE",
        "description": "Compact multilingual cross-encoder for 70+ languages.",
        "model": "Alibaba-NLP/gte-multilingual-reranker-base",
        "revision": "8215cf04918ba6f7b6a62bb44238ce2953d8831c",
        "code_repository": "Alibaba-NLP/new-impl",
        "code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4",
        "backend": "classifier",
        "prompt_version": "gte-multilingual-reranker-base-v1",
        "instruction": (
            "Determine whether the multilingual document passage is relevant "
            "to the search query. Prefer exact and citable evidence."
        ),
        # Parent passages are normally below this limit; keeping the bounded
        # input avoids paying the model's full 8192-token context cost.
        "max_length": 512,
        "batch_size": 8,
        "use_fp16": True,
        # The official model config maps to Alibaba-NLP/new-impl.
        "trust_remote_code": True,
    },
}


def resolve_reranker_choice(configured: str) -> str:
    """Normalize and validate the selected reranker preset."""

    value = configured.strip().lower().replace("_", "-")

    aliases = {
        "bge": "bge",
        "bge-m3": "bge",
        "bge-v2-m3": "bge",
        "bge-reranker": "bge",
        "bge-reranker-v2-m3": "bge",
        "gte": "gte",
        "gte-multilingual": "gte",
        "gte-reranker": "gte",
        "gte-multilingual-reranker-base": "gte",
    }

    try:
        return aliases[value]
    except KeyError as error:
        valid_choices = ", ".join(sorted(RERANKER_PRESETS))

        raise ValueError(
            "RAG_RERANKER_CHOICE must select a supported reranker. "
            f"Valid choices: {valid_choices}. "
            f"Received: {configured!r}."
        ) from error


# GTE is the default when RAG_RERANKER_CHOICE is not set.
RERANKER_CHOICE = resolve_reranker_choice(
    os.environ.get(
        "RAG_RERANKER_CHOICE",
        "gte",
    )
)

RERANKER_PRESET = RERANKER_PRESETS[RERANKER_CHOICE]


# =============================================================================
# Local reranking
# =============================================================================


# The selected preset is authoritative. Old model/backend environment variables
# cannot silently replace the chosen reranker.
RERANKER_MODEL = str(
    RERANKER_PRESET["model"]
)

RERANKER_BACKEND = str(
    RERANKER_PRESET["backend"]
)

RERANKER_REVISION = os.environ.get(
    "RAG_RERANKER_REVISION",
    str(RERANKER_PRESET["revision"]),
)

RERANKER_CODE_REVISION = os.environ.get(
    "RAG_RERANKER_CODE_REVISION",
    str(RERANKER_PRESET.get("code_revision") or RERANKER_PRESET["revision"]),
)

RERANKER_USE_AUTH = env_flag(
    "RAG_RERANKER_USE_AUTH",
    False,
)

RERANKER_PROMPT_VERSION = os.environ.get(
    "RAG_RERANKER_PROMPT_VERSION",
    str(RERANKER_PRESET["prompt_version"]),
)

RERANK_INSTRUCTION = os.environ.get(
    "RAG_RERANK_INSTRUCTION",
    str(RERANKER_PRESET["instruction"]),
)

RERANK_MAX_LENGTH = env_int(
    "RAG_RERANK_MAX_LENGTH",
    int(RERANKER_PRESET["max_length"]),
)

RERANK_BATCH_SIZE = env_int(
    "RAG_RERANK_BATCH_SIZE",
    int(RERANKER_PRESET["batch_size"]),
)

RERANK_TOP_N = env_int(
    "RAG_RERANK_TOP_N",
    5,
)

ADDITIONAL_RESULTS = env_int(
    "RAG_ADDITIONAL_RESULTS",
    15,
)

MAX_RESULTS_PER_SECTION = env_int(
    "RAG_MAX_RESULTS_PER_SECTION",
    2,
)

MAX_RESULTS_PER_PAGE = env_int(
    "RAG_MAX_RESULTS_PER_PAGE",
    1,
)

MAX_RESULT_TEXT_SIMILARITY = env_float(
    "RAG_MAX_RESULT_TEXT_SIMILARITY",
    0.88,
)

RERANK_USE_FP16 = env_flag(
    "RAG_RERANK_FP16",
    bool(RERANKER_PRESET["use_fp16"]),
)


def resolve_reranker_backend(
    model_name: str = RERANKER_MODEL,
    configured: str = RERANKER_BACKEND,
) -> str:
    """Resolve and validate the reranker backend.

    The selectable rerankers use sequence-classification cross-encoders.
    """

    value = configured.strip().lower().replace("_", "-")

    aliases = {
        "classifier": "classifier",
        "cross-encoder": "classifier",
        "crossencoder": "classifier",
        "bge": "classifier",
    }

    try:
        resolved = aliases[value]
    except KeyError as error:
        raise ValueError(
            "Reranker backend must be classifier. "
            f"Received: {configured!r}."
        ) from error

    return resolved


def reranker_configuration(choice: str | None = None) -> dict[str, object]:
    """Return one preset's effective reranker configuration.

    Environment overrides apply to every selectable preset, while each preset
    keeps its own model, backend, and safe batch-size defaults.
    """

    selected = RERANKER_CHOICE if choice is None else resolve_reranker_choice(choice)
    preset = RERANKER_PRESETS[selected]
    revision = os.environ.get(f"RAG_{selected.upper()}_RERANKER_REVISION")
    code_revision = os.environ.get(
        f"RAG_{selected.upper()}_RERANKER_CODE_REVISION"
    )
    if selected == RERANKER_CHOICE:
        revision = revision or RERANKER_REVISION
        code_revision = code_revision or RERANKER_CODE_REVISION

    return {
        "choice": selected,
        "label": str(preset["label"]),
        "description": str(preset["description"]),
        "model": str(preset["model"]),
        "revision": revision or str(preset["revision"]),
        "code_repository": str(preset.get("code_repository") or ""),
        "code_revision": code_revision
        or str(preset.get("code_revision") or preset["revision"]),
        "backend": resolve_reranker_backend(
            str(preset["model"]),
            str(preset["backend"]),
        ),
        "use_auth": RERANKER_USE_AUTH,
        "trust_remote_code": bool(preset.get("trust_remote_code", False)),
        "max_length": env_int(
            "RAG_RERANK_MAX_LENGTH",
            int(preset["max_length"]),
        ),
        "batch_size": env_int(
            "RAG_RERANK_BATCH_SIZE",
            int(preset["batch_size"]),
        ),
        "use_fp16": env_flag(
            "RAG_RERANK_FP16",
            bool(preset["use_fp16"]),
        ),
        "prompt_version": os.environ.get(
            "RAG_RERANKER_PROMPT_VERSION",
            str(preset["prompt_version"]),
        ),
        "instruction": os.environ.get(
            "RAG_RERANK_INSTRUCTION",
            str(preset["instruction"]),
        ),
    }


def reranker_fingerprint(choice: str | None = None) -> str:
    """Produce a stable hash for reranker-dependent stored data."""

    configuration = reranker_configuration(choice)
    configuration.pop("label", None)
    configuration.pop("description", None)
    payload = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def reranker_summary(choice: str | None = None) -> str:
    """Return a readable description of the active reranker."""

    configuration = reranker_configuration(choice)

    return (
        f"{configuration['choice']} | "
        f"{configuration['model']} | "
        f"{configuration['backend']} | "
        f"batch={configuration['batch_size']} | "
        f"max_length={configuration['max_length']} | "
        f"fp16={configuration['use_fp16']}"
    )


# =============================================================================
# Relevance gate
# =============================================================================


# Raw-logit relevance gate. Disabled until calibrated on a labeled benchmark.
MIN_BEST_RERANK_LOGIT = env_float(
    "RAG_MIN_BEST_RERANK_LOGIT",
    -2.2,
)

MIN_RESULT_RERANK_LOGIT = env_float(
    "RAG_MIN_RESULT_RERANK_LOGIT",
    -3.0,
)

ENABLE_RELEVANCE_GATE = env_flag(
    "RAG_ENABLE_RELEVANCE_GATE",
    False,
)


# =============================================================================
# Feedback calibration
# =============================================================================


# The automatic gate stays off until both classes have enough explicit human
# judgments to avoid fitting a threshold to a tiny sample.
QUALITY_MIN_POSITIVE_LABELS = env_int(
    "RAG_QUALITY_MIN_POSITIVE_LABELS",
    20,
)

QUALITY_MIN_NEGATIVE_LABELS = env_int(
    "RAG_QUALITY_MIN_NEGATIVE_LABELS",
    20,
)

QUALITY_MIN_RECALL = env_float(
    "RAG_QUALITY_MIN_RECALL",
    0.90,
)


# =============================================================================
# Concurrent search and ingestion
# =============================================================================


# Auto mode measures free VRAM after the search runtime is loaded. Total VRAM
# alone does not tell us whether both jobs fit at a particular moment.
SEARCH_DURING_INGESTION = os.environ.get(
    "RAG_SEARCH_DURING_INGESTION",
    "auto",
).strip().lower()

if SEARCH_DURING_INGESTION not in {
    "auto",
    "never",
    "always",
}:
    raise ValueError(
        "RAG_SEARCH_DURING_INGESTION must be auto, never, or always."
    )


DOCLING_GPU_HEADROOM_MB = env_int(
    "RAG_GPU_HEADROOM_WARNING_MB",
    env_int(
        "RAG_CUDA_HEADROOM_WARNING_MB",
        3500,
    ),
)

CONCURRENT_QUERY_RESERVE_MB = env_int(
    "RAG_CONCURRENT_QUERY_RESERVE_MB",
    768,
)


# =============================================================================
# Presentation, diagnostics, and local server
# =============================================================================


MAX_CHARS_PER_RESULT = env_int(
    "RAG_MAX_RESULT_CHARS",
    6000,
)

DEBUG_RETRIEVAL = env_flag(
    "RAG_DEBUG_RETRIEVAL",
    False,
)

OPEN_BROWSER = env_flag(
    "RAG_OPEN_BROWSER",
    True,
)

WARM_RERANKER_ON_START = env_flag(
    "RAG_WARM_RERANKER_ON_START",
    True,
)

STARTUP_WARM_DELAY_SECONDS = env_float(
    "RAG_STARTUP_WARM_DELAY_SECONDS",
    0.5,
)

SERVER_HOST = os.environ.get(
    "RAG_SERVER_HOST",
    "127.0.0.1",
)

SERVER_PORT = env_int(
    "RAG_SERVER_PORT",
    7860,
)
