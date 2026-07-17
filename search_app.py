"""Local hybrid document retrieval and Qwen reranking application.

The application deliberately performs retrieval only. It does not generate an
answer and does not rewrite the user's query with an LLM.
"""

from __future__ import annotations

import html
import base64
import io
import json
import logging
import os
import re
import gc
import sqlite3
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import webbrowser
import zipfile
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from secrets import token_urlsafe
from time import perf_counter
from typing import Any, Iterable, Sequence
from urllib.parse import quote, urlencode

import chromadb
import gradio as gr
import pypdfium2 as pdfium
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app_settings import (
    ADDITIONAL_RESULTS,
    CONCURRENT_QUERY_RESERVE_MB,
    DEBUG_RETRIEVAL,
    DENSE_CANDIDATES,
    DOCLING_GPU_HEADROOM_MB,
    ENABLE_RELEVANCE_GATE,
    LEXICAL_CANDIDATES,
    MAX_CHARS_PER_RESULT,
    MAX_RESULT_TEXT_SIMILARITY,
    MAX_RESULTS_PER_PAGE,
    MAX_RESULTS_PER_SECTION,
    MIN_BEST_RERANK_LOGIT,
    MIN_RESULT_RERANK_LOGIT,
    OPEN_BROWSER,
    QUALITY_MIN_NEGATIVE_LABELS,
    QUALITY_MIN_POSITIVE_LABELS,
    QUALITY_MIN_RECALL,
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATES,
    RERANK_INSTRUCTION,
    RERANK_MAX_LENGTH,
    RERANK_TOP_N,
    RERANK_USE_FP16,
    RERANK_WEIGHT,
    RERANKER_MODEL,
    RERANKER_REVISION,
    RERANKER_USE_AUTH,
    RRF_K,
    SEARCH_DURING_INGESTION,
    SERVER_HOST,
    SERVER_PORT,
    reranker_fingerprint,
    resolve_reranker_backend,
)
from lexical_index import rebuild_from_collection
from lexical_index import fingerprint_ids as fingerprint_lexical_ids
from lexical_index import search as lexical_search
from lexical_index import state as lexical_state
from document_manager import (
    DuplicateDocumentError,
    DocumentPathError,
    DocumentRepository,
    document_manager_html,
)
from rag_common import (
    COLLECTION_NAME,
    COMPUTE_BACKEND,
    DB_DIR,
    DEBUG_DIR,
    EMBEDDING_MODEL,
    HARDWARE,
    LEXICAL_DB_PATH,
    MANIFEST_PATH,
    OLLAMA_BASE_URL,
    PDF_DIR,
    RERANKER_DEVICE,
    create_embeddings,
    embedding_fingerprint,
)
from hardware import backend_label, detected_accelerator, runtime_version
from workspace_store import WorkspaceStore, parse_export_ids
from workspace_ui import workspace_html
from quality_ui import quality_dashboard_html
from corpus_scale import CorpusScaleManager, PAUSED_EXIT_CODE


logging.basicConfig(
    level=os.environ.get("RAG_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("document-rag")


@dataclass
class Candidate:
    document: Document
    chunk_id: str
    vector_distance: float | None = None
    vector_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    fusion_score: float = 0.0
    retrieval_rank: int = 0
    rerank_logit: float = float("-inf")
    rerank_probability: float = 0.0
    rerank_rank: int = 0
    final_score: float = 0.0
    final_rank: int = 0
    rerank_token_count: int = 0
    rerank_truncated: bool = False


@dataclass(frozen=True)
class SearchTimings:
    dense: float
    lexical: float
    reranking: float
    total: float


@dataclass(frozen=True)
class RetrievalFilters:
    """Metadata/content filters applied before expensive cross-encoder scoring."""

    section: str = ""
    content_type: str = ""
    date: str = ""


@dataclass
class Runtime:
    collection: Any
    vectorstore: Chroma
    reranker: "LocalReranker"
    inference_lock: threading.Lock = field(default_factory=threading.Lock)


class LocalReranker:
    """Qwen yes/no reranker with an optional classifier compatibility path."""

    QWEN_PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" "
        "or \"no\".<|im_end|>\n<|im_start|>user\n"
    )
    QWEN_SUFFIX = (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        max_length: int = RERANK_MAX_LENGTH,
        batch_size: int = RERANK_BATCH_SIZE,
        use_fp16: bool = RERANK_USE_FP16,
        device: str = RERANKER_DEVICE,
        revision: str = RERANKER_REVISION,
        instruction: str = RERANK_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = resolve_torch_device(device)
        self.backend = resolve_reranker_backend(model_name)
        self.instruction = instruction
        model_token: bool = True if RERANKER_USE_AUTH else False

        LOGGER.info(
            "Loading reranker %s (%s) on %s",
            model_name,
            self.backend,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            padding_side="left" if self.backend == "qwen-causal-lm" else "right",
            token=model_token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_dtype = (
            torch.float16
            if self.device.type == "cuda" and use_fp16
            else torch.float32
        )
        model_class = (
            AutoModelForCausalLM
            if self.backend == "qwen-causal-lm"
            else AutoModelForSequenceClassification
        )
        self.model = model_class.from_pretrained(
            model_name,
            revision=revision,
            dtype=model_dtype,
            token=model_token,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.backend == "qwen-causal-lm":
            self._prefix_ids = self.tokenizer.encode(
                self.QWEN_PREFIX,
                add_special_tokens=False,
            )
            self._suffix_ids = self.tokenizer.encode(
                self.QWEN_SUFFIX,
                add_special_tokens=False,
            )
            self._no_token_id = self.tokenizer.convert_tokens_to_ids("no")
            self._yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
            if self._no_token_id == self.tokenizer.unk_token_id or self._yes_token_id == self.tokenizer.unk_token_id:
                raise RuntimeError("Qwen reranker tokenizer does not expose yes/no tokens.")
            body_budget = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
            if body_budget < 32:
                raise ValueError(
                    "RAG_RERANK_MAX_LENGTH is too small for the Qwen reranker prompt."
                )
            self._body_budget = body_budget

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[float, float, int, bool]]:
        if not pairs:
            return []

        if self.backend == "qwen-causal-lm":
            return self._predict_qwen(pairs)
        return self._predict_classifier(pairs)

    def _predict_qwen(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[float, float, int, bool]]:
        output: list[tuple[float, float, int, bool]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            bodies = [
                (
                    f"<Instruct>: {self.instruction}\n"
                    f"<Query>: {query}\n"
                    f"<Document>: {passage}"
                )
                for query, passage in batch
            ]
            untruncated = self.tokenizer(
                bodies,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            overhead = len(self._prefix_ids) + len(self._suffix_ids)
            token_counts = [overhead + len(input_ids) for input_ids in untruncated]
            body_ids = self.tokenizer(
                bodies,
                padding=False,
                truncation=True,
                max_length=self._body_budget,
                add_special_tokens=False,
            )["input_ids"]
            encoded = self.tokenizer.pad(
                {
                    "input_ids": [
                        self._prefix_ids + input_ids + self._suffix_ids
                        for input_ids in body_ids
                    ]
                },
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                final_logits = self.model(**encoded).logits[:, -1, :].float()
                no_logits = final_logits[:, self._no_token_id]
                yes_logits = final_logits[:, self._yes_token_id]
                log_odds = yes_logits - no_logits
                probabilities = torch.sigmoid(log_odds)

            for logit, probability, token_count in zip(
                log_odds.cpu().tolist(),
                probabilities.cpu().tolist(),
                token_counts,
                strict=True,
            ):
                output.append(
                    (
                        float(logit),
                        float(probability),
                        token_count,
                        token_count > self.max_length,
                    )
                )
        return output

    def _predict_classifier(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[float, float, int, bool]]:

        output: list[tuple[float, float, int, bool]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            queries = [query for query, _ in batch]
            passages = [passage for _, passage in batch]

            untruncated = self.tokenizer(
                queries,
                passages,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            token_counts = [len(input_ids) for input_ids in untruncated["input_ids"]]

            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                logits = self.model(**encoded).logits.view(-1).float()
                probabilities = torch.sigmoid(logits)

            for logit, probability, token_count in zip(
                logits.cpu().tolist(),
                probabilities.cpu().tolist(),
                token_counts,
                strict=True,
            ):
                output.append(
                    (
                        float(logit),
                        float(probability),
                        token_count,
                        token_count > self.max_length,
                    )
                )
        return output


# Compatibility for local extensions that imported the previous class name.
BGEReranker = LocalReranker


def resolve_torch_device(configured: str) -> torch.device:
    """Resolve a PyTorch device for either NVIDIA CUDA or AMD ROCm."""

    requested = configured.strip().lower()
    value = requested
    if value == "auto":
        value = "cuda:0" if COMPUTE_BACKEND in {"cuda", "rocm"} else "cpu"
    elif value in {"cuda", "nvidia", "rocm", "amd", "hip"}:
        value = "cuda:0"

    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            build = torch.version.hip or torch.version.cuda or "CPU-only"
            if requested in {"rocm", "amd", "hip"}:
                requested_backend = "rocm"
            elif requested in {"cuda", "nvidia"}:
                requested_backend = "cuda"
            elif COMPUTE_BACKEND in {"cuda", "rocm"}:
                # ROCm intentionally uses PyTorch's cuda:N device notation.
                requested_backend = COMPUTE_BACKEND
            else:
                requested_backend = "cuda"
            raise RuntimeError(
                f"{backend_label(requested_backend)} was requested for the reranker, "
                f"but this PyTorch build cannot use a GPU (runtime: {build}). "
                "Run the portable setup script for this machine or select CPU."
            )
        detected = detected_accelerator(torch)
        if COMPUTE_BACKEND in {"cuda", "rocm"} and detected != COMPUTE_BACKEND:
            raise RuntimeError(
                f"The configured backend is {backend_label(COMPUTE_BACKEND)}, but "
                f"PyTorch provides {backend_label(detected)}. Re-run setup with "
                "the matching backend."
            )
        device = torch.device(value)
        torch.cuda.set_device(device)
        LOGGER.info(
            "%s ready: %s (runtime %s)",
            backend_label(COMPUTE_BACKEND),
            torch.cuda.get_device_name(device),
            runtime_version(COMPUTE_BACKEND, torch),
        )
        return device

    if value != "cpu":
        raise ValueError(
            "RAG_RERANKER_DEVICE must be auto, cpu, cuda:N, nvidia, or rocm; "
            f"received {configured!r}."
        )
    return torch.device("cpu")


_RUNTIME: Runtime | None = None
_RUNTIME_LOCK = threading.Lock()
_CITATION_COLLECTION: Any | None = None
_CITATION_COLLECTION_LOCK = threading.Lock()
_PDFIUM_LOCK = threading.RLock()
_SOURCE_NAVIGATION_LOCK = threading.Lock()
_SOURCE_NAVIGATION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
SOURCE_NAVIGATION_CACHE_SIZE = 64
SOURCE_NAVIGATION_TTL_SECONDS = 2 * 60 * 60
WORKSPACE_STORE = WorkspaceStore()
FRONTEND_DIST_DIR = Path(__file__).resolve().with_name("frontend") / "dist"
DOCUMENT_REPOSITORY = DocumentRepository(
    pdf_dir=PDF_DIR,
    manifest_path=MANIFEST_PATH,
    state_path=DB_DIR / "document_manager_state.json",
    trash_dir=PDF_DIR.parent / "document_trash",
    quarantine_dir=PDF_DIR.parent / "document_quarantine",
    revision_dir=PDF_DIR.parent / "document_revisions",
)
CORPUS_SCALE = CorpusScaleManager(
    DOCUMENT_REPOSITORY,
    state_path=DB_DIR / "corpus_scale_state.json",
    db_dir=DB_DIR,
    workspace_db=WORKSPACE_STORE.path,
    backup_dir=PDF_DIR.parent / "corpus_backups",
    debug_dir=DEBUG_DIR,
)
MAX_DOCUMENT_UPLOAD_BYTES = int(
    os.environ.get("RAG_MAX_DOCUMENT_UPLOAD_BYTES", str(1024 * 1024 * 1024))
)
_SEARCH_MAINTENANCE_LOCK = threading.Lock()
_DOCUMENT_MAINTENANCE = threading.Event()
_DOCUMENT_JOB_LOCK = threading.Lock()
_APP_RESTARTING = threading.Event()
_APP_INSTANCE_ID = token_urlsafe(8)
_DOCUMENT_JOB: dict[str, Any] = {
    "state": "idle",
    "running": False,
    "message": "",
    "log": [],
    "started_at": None,
    "finished_at": None,
    "force": False,
    "pause_requested": False,
    "search_available": False,
    "concurrency_reason": "No indexing job is running.",
}


class DocumentMaintenanceError(RuntimeError):
    """Raised when a search is attempted during an index update."""


def _gpu_memory_mib() -> tuple[int, int] | None:
    """Return currently free and total accelerator memory, when available."""

    if COMPUTE_BACKEND not in {"cuda", "rocm"} or not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, OSError):
        return None
    return (
        int(free_bytes // (1024 * 1024)),
        int(total_bytes // (1024 * 1024)),
    )


def concurrent_ingestion_policy() -> tuple[bool, str]:
    """Decide whether a loaded search runtime can remain active for ingestion."""

    if SEARCH_DURING_INGESTION == "never":
        return False, "Concurrent search is disabled by configuration."
    with _RUNTIME_LOCK:
        runtime_loaded = _RUNTIME is not None
    if not runtime_loaded:
        return False, "Search is not loaded yet, so indexing uses exclusive mode."
    if SEARCH_DURING_INGESTION == "always":
        return True, "Concurrent search was explicitly forced by configuration."

    memory = _gpu_memory_mib()
    if memory is None:
        return False, "Auto mode requires a supported GPU with memory reporting."
    free_mb, total_mb = memory
    required_mb = DOCLING_GPU_HEADROOM_MB + CONCURRENT_QUERY_RESERVE_MB
    if free_mb < required_mb:
        return (
            False,
            f"GPU has {free_mb} MiB free; Docling and a live query need "
            f"about {required_mb} MiB (Docling {DOCLING_GPU_HEADROOM_MB} + "
            f"query reserve {CONCURRENT_QUERY_RESERVE_MB}).",
        )
    return (
        True,
        f"GPU has {free_mb} MiB free of {total_mb} MiB; concurrent reserve is "
        f"{required_mb} MiB.",
    )


def _concurrent_search_is_safe() -> tuple[bool, str]:
    """Avoid a new rerank batch when ingestion has exhausted GPU headroom."""

    if SEARCH_DURING_INGESTION == "always":
        return True, "Concurrent search was explicitly forced by configuration."
    memory = _gpu_memory_mib()
    if memory is None:
        return False, "GPU memory is not available for a concurrent query."
    free_mb, _ = memory
    if free_mb < CONCURRENT_QUERY_RESERVE_MB:
        return (
            False,
            f"Indexing currently leaves only {free_mb} MiB of GPU memory free.",
        )
    return True, "GPU headroom is available."


def _guard_search_during_ingestion() -> None:
    job = document_job_snapshot()
    if not job.get("running"):
        return
    if not job.get("search_available"):
        raise DocumentMaintenanceError(
            "The document index is being updated. Search will resume when it finishes."
        )
    with _RUNTIME_LOCK:
        runtime_loaded = _RUNTIME is not None
    if not runtime_loaded:
        raise DocumentMaintenanceError(
            "The search runtime is restarting during indexing. Try again shortly."
        )
    safe, reason = _concurrent_search_is_safe()
    if not safe:
        raise DocumentMaintenanceError(
            f"Search is temporarily paused while indexing: {reason}"
        )


def document_job_snapshot() -> dict[str, Any]:
    with _DOCUMENT_JOB_LOCK:
        return {
            **_DOCUMENT_JOB,
            "log": list(_DOCUMENT_JOB.get("log") or []),
        }


def _update_document_job(**changes: Any) -> None:
    with _DOCUMENT_JOB_LOCK:
        _DOCUMENT_JOB.update(changes)


def _append_document_job_log(line: str) -> None:
    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line).strip()
    if not cleaned:
        return
    with _DOCUMENT_JOB_LOCK:
        lines = deque(_DOCUMENT_JOB.get("log") or [], maxlen=80)
        lines.append(cleaned[-1200:])
        _DOCUMENT_JOB["log"] = list(lines)
        if cleaned.startswith("Processing:"):
            _DOCUMENT_JOB["message"] = cleaned
        elif cleaned.startswith("Converting pages") or "Converting pages" in cleaned:
            _DOCUMENT_JOB["message"] = cleaned


def _handle_corpus_event(line: str) -> bool:
    marker = "CORPUS_EVENT "
    stripped = line.strip()
    if not stripped.startswith(marker):
        return False
    try:
        event = json.loads(stripped[len(marker) :])
    except json.JSONDecodeError:
        return False
    kind = str(event.get("event") or "")
    source_id = str(event.get("source_id") or "")
    if kind == "started" and source_id:
        CORPUS_SCALE.mark_event(source_id, "processing")
        _update_document_job(message=f"Processing {source_id}")
    elif kind == "completed" and source_id:
        CORPUS_SCALE.mark_event(source_id, "complete")
        if hasattr(DOCUMENT_REPOSITORY, "clear_pending_sources"):
            DOCUMENT_REPOSITORY.clear_pending_sources((source_id,), ())
        CORPUS_SCALE.invalidate_health()
    elif kind == "failed" and source_id:
        error = f"{event.get('error_type') or 'Error'}: {event.get('error') or 'Ingestion failed'}"
        try:
            DOCUMENT_REPOSITORY.quarantine(source_id, error)
        except (FileNotFoundError, OSError, AttributeError) as quarantine_error:
            CORPUS_SCALE.mark_event(
                source_id,
                "failed",
                error=f"{error} Quarantine failed: {quarantine_error}",
            )
        else:
            CORPUS_SCALE.mark_event(source_id, "quarantined", error=error)
        CORPUS_SCALE.invalidate_health()
    elif kind == "paused":
        _update_document_job(
            message="Paused after the current document. Search is available again.",
            pause_requested=False,
        )
    return True


def _release_search_runtime() -> None:
    """Release GPU and database handles before the ingestion subprocess."""

    global _RUNTIME, _CITATION_COLLECTION
    with _RUNTIME_LOCK:
        runtime = _RUNTIME
        _RUNTIME = None
    with _CITATION_COLLECTION_LOCK:
        _CITATION_COLLECTION = None

    if runtime is not None:
        reranker = runtime.reranker
        if hasattr(reranker, "model"):
            del reranker.model
        if hasattr(reranker, "tokenizer"):
            del reranker.tokenizer
        del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except (RuntimeError, OSError):
            pass


def _subprocess_creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _unload_ollama_embedding_model() -> None:
    """Unload the retained embedding model without launching the Ollama app."""

    base_url = OLLAMA_BASE_URL.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        loaded_names = {
            str(item.get("name") or item.get("model") or "")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        requested_base = EMBEDDING_MODEL.split(":", 1)[0]
        if not any(
            name == EMBEDDING_MODEL or name.split(":", 1)[0] == requested_base
            for name in loaded_names
        ):
            return
        body = json.dumps(
            {
                "model": EMBEDDING_MODEL,
                "input": "",
                "keep_alive": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            response.read(1)
        _append_document_job_log("Released the retained Ollama embedding model.")
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        _append_document_job_log(
            f"Could not unload Ollama automatically ({type(error).__name__}); continuing with the GPU memory guard."
        )


def schedule_application_restart() -> dict[str, Any]:
    """Reset the search runtime while keeping the local web server supervised."""

    if _APP_RESTARTING.is_set():
        return {"restarting": True, "instance_id": _APP_INSTANCE_ID}
    _APP_RESTARTING.set()
    previous_instance_id = _APP_INSTANCE_ID

    def reset_application_runtime() -> None:
        global _APP_INSTANCE_ID

        _DOCUMENT_MAINTENANCE.set()
        try:
            with _SEARCH_MAINTENANCE_LOCK:
                _release_search_runtime()
                _unload_ollama_embedding_model()
            _APP_INSTANCE_ID = token_urlsafe(8)
        except Exception:
            LOGGER.exception("The application runtime could not be restarted")
        finally:
            _DOCUMENT_MAINTENANCE.clear()
            _APP_RESTARTING.clear()

    threading.Thread(
        target=reset_application_runtime,
        name="application-runtime-restart",
        daemon=True,
    ).start()
    return {"restarting": True, "instance_id": previous_instance_id}


def _run_document_index_job(
    force: bool,
    search_available: bool = False,
) -> None:
    _update_document_job(
        state="waiting",
        message=(
            "Preparing concurrent indexing while search stays available."
            if search_available
            else "Waiting for active searches to finish."
        ),
    )
    exit_code = -1
    try:
        if search_available:
            _update_document_job(
                state="preparing",
                message="Keeping the loaded search runtime; checking GPU headroom.",
            )
        else:
            with _SEARCH_MAINTENANCE_LOCK:
                _update_document_job(
                    state="preparing",
                    message="Releasing search models and GPU memory.",
                )
                _release_search_runtime()
                _unload_ollama_embedding_model()

        managed_queue = (
            hasattr(DOCUMENT_REPOSITORY, "clear_pending_sources")
            and CORPUS_SCALE.repository is DOCUMENT_REPOSITORY
        )
        command = [sys.executable, str(Path(__file__).with_name("ingest.py")), "--prune"]
        if managed_queue:
            command.extend(
                ("--queue-managed", "--queue-control", str(CORPUS_SCALE.state_path))
            )
            prune_command = list(command)
            for source_id in CORPUS_SCALE.queued_sources():
                command.extend(("--source", source_id))
        if force:
            command.append("--force")
        environment = os.environ.copy()
        environment["RAG_OPEN_BROWSER"] = "0"
        if search_available:
            # Docling independently refuses to start if real free VRAM falls
            # below this combined reserve after the child process starts.
            environment["RAG_GPU_HEADROOM_WARNING_MB"] = str(
                DOCLING_GPU_HEADROOM_MB + CONCURRENT_QUERY_RESERVE_MB
            )
        _update_document_job(
            state="running",
            message=(
                "Indexing documents. Search remains available while GPU headroom permits."
                if search_available
                else "Starting document ingestion."
            ),
        )
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=_subprocess_creation_flags(),
        )
        assert process.stdout is not None
        structured_events = False
        for line in process.stdout:
            if _handle_corpus_event(line):
                structured_events = True
            else:
                _append_document_job_log(line)
        exit_code = process.wait()
        if exit_code != 0 and managed_queue:
            queue_after_run = CORPUS_SCALE.snapshot()
            quarantine_count = int(
                queue_after_run.get("counts", {}).get("quarantined", 0)
            )
            if quarantine_count and (
                not queue_after_run.get("remaining")
                or exit_code == PAUSED_EXIT_CODE
            ):
                _append_document_job_log(
                    "Finalizing the index after quarantining failed documents."
                )
                prune_process = subprocess.Popen(
                    prune_command,
                    cwd=str(Path(__file__).resolve().parent),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=_subprocess_creation_flags(),
                )
                assert prune_process.stdout is not None
                for line in prune_process.stdout:
                    if not _handle_corpus_event(line):
                        _append_document_job_log(line)
                if prune_process.wait() == 0 and exit_code != PAUSED_EXIT_CODE:
                    exit_code = 0

        if exit_code == 0:
            completed_deletions = (
                CORPUS_SCALE.mark_deletions_complete() if managed_queue else []
            )
            if managed_queue:
                DOCUMENT_REPOSITORY.clear_pending_sources((), tuple(completed_deletions))
            if not managed_queue or not structured_events:
                DOCUMENT_REPOSITORY.clear_pending()
            queue = CORPUS_SCALE.snapshot() if managed_queue else {"counts": {}}
            quarantined = int(queue.get("counts", {}).get("quarantined", 0))
            _update_document_job(
                state="complete_with_failures" if quarantined else "complete",
                running=False,
                message=(
                    f"Index updated. {quarantined} failed document(s) were moved to quarantine."
                    if quarantined
                    else "The document index is current. Search models will reload on the next query."
                ),
                finished_at=datetime.now(tz=UTC).isoformat(),
            )
        elif exit_code == PAUSED_EXIT_CODE:
            _update_document_job(
                state="paused",
                running=False,
                pause_requested=False,
                message="Queue paused. Search is available; resume when you are ready.",
                finished_at=datetime.now(tz=UTC).isoformat(),
            )
        else:
            queue = CORPUS_SCALE.snapshot() if managed_queue else {"remaining": 1, "counts": {}}
            remaining = int(queue.get("remaining") or 0)
            quarantined = int(queue.get("counts", {}).get("quarantined", 0))
            _update_document_job(
                state="failed" if remaining else "complete_with_failures",
                running=False,
                message=(
                    "Ingestion stopped before the queue completed. Pending work was preserved."
                    if remaining
                    else f"Queue completed with {quarantined} quarantined document(s)."
                ),
                finished_at=datetime.now(tz=UTC).isoformat(),
            )
    except Exception as error:
        LOGGER.exception("Document index job failed")
        _append_document_job_log(f"{type(error).__name__}: {error}")
        _update_document_job(
            state="failed",
            running=False,
            message="Index update failed. Pending changes were preserved.",
            finished_at=datetime.now(tz=UTC).isoformat(),
        )
    finally:
        _DOCUMENT_MAINTENANCE.clear()


def start_document_index_job(
    force: bool = False,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    search_available, concurrency_reason = concurrent_ingestion_policy()
    with _DOCUMENT_JOB_LOCK:
        if _DOCUMENT_JOB.get("running"):
            raise RuntimeError("A document index update is already running.")
        if resume:
            queue = CORPUS_SCALE.resume()
        else:
            queue = CORPUS_SCALE.prepare_queue(
                DOCUMENT_REPOSITORY.summary(),
                force=bool(force),
            )
        if not queue.get("remaining"):
            raise RuntimeError("There is no queued corpus work to run.")
        _DOCUMENT_JOB.update(
            {
                "state": "queued",
                "running": True,
                "message": "Index update queued",
                "log": [],
                "started_at": datetime.now(tz=UTC).isoformat(),
                "finished_at": None,
                "force": bool(force),
                "pause_requested": False,
                "search_available": search_available,
                "concurrency_reason": concurrency_reason,
            }
        )
        if search_available:
            _DOCUMENT_MAINTENANCE.clear()
        else:
            _DOCUMENT_MAINTENANCE.set()
    threading.Thread(
        target=_run_document_index_job,
        args=(bool(force), search_available),
        name="document-index-job",
        daemon=True,
    ).start()
    return document_job_snapshot()


def build_runtime() -> Runtime:
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Chroma database not found: {DB_DIR}. Run ingest.py first."
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError(
            f"Collection {COLLECTION_NAME!r} does not exist. Run ingest.py first."
        ) from error

    if collection.count() == 0:
        raise RuntimeError(f"Collection {COLLECTION_NAME!r} is empty.")

    actual_fingerprint = (collection.metadata or {}).get("embedding_fingerprint")
    if actual_fingerprint != embedding_fingerprint():
        raise RuntimeError(
            "Embedding configuration does not match the indexed collection. "
            "Use the same model revision and prompts or rebuild the index."
        )

    current_lexical_state = lexical_state(LEXICAL_DB_PATH)
    collection_ids = list(collection.get(include=[]).get("ids") or [])
    if (
        current_lexical_state.get("embedding_fingerprint")
        != embedding_fingerprint()
        or current_lexical_state.get("chunk_count") != str(collection.count())
        or current_lexical_state.get("ids_fingerprint")
        != fingerprint_lexical_ids(collection_ids)
    ):
        rebuilt = rebuild_from_collection(
            path=LEXICAL_DB_PATH,
            collection=collection,
            fingerprint=embedding_fingerprint(),
        )
        LOGGER.info("Rebuilt lexical index with %d chunks", rebuilt)

    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
    )
    return Runtime(
        collection=collection,
        vectorstore=vectorstore,
        reranker=LocalReranker(),
    )


def get_runtime() -> Runtime:
    global _RUNTIME
    if _DOCUMENT_MAINTENANCE.is_set():
        raise DocumentMaintenanceError(
            "The document index is being updated. Search will resume automatically when it finishes."
        )
    _guard_search_during_ingestion()
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _DOCUMENT_MAINTENANCE.is_set():
                raise DocumentMaintenanceError(
                    "The document index is being updated. Search will resume automatically when it finishes."
                )
            _guard_search_during_ingestion()
            if _RUNTIME is None:
                _RUNTIME = build_runtime()
    return _RUNTIME


def first_existing(
    metadata: dict[str, Any],
    keys: Iterable[str],
    default: str = "not available",
) -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], -1, "-1"):
            return str(value)
    return default


def format_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    source = first_existing(
        metadata,
        ["source", "source_file", "file", "filename", "file_path", "path"],
        "unknown source",
    )
    source_id = first_existing(metadata, ["source_id"], source)
    document_title = first_existing(
        metadata,
        ["document_title", "title"],
        Path(source).stem,
    )
    page_start = first_existing(
        metadata,
        ["page_start", "page", "page_no", "page_number"],
    )
    page_end = first_existing(
        metadata,
        ["page_end", "page", "page_no", "page_number"],
    )
    section = first_existing(
        metadata,
        ["section_path", "section_title", "section", "headings", "heading"],
    )
    chunk_index = first_existing(metadata, ["chunk_index", "chunk", "index"])
    content_labels = first_existing(metadata, ["content_labels"], "")

    if page_start == page_end:
        page_display = page_start
    elif page_start == "not available" and page_end == "not available":
        page_display = "not available"
    else:
        page_display = f"{page_start}-{page_end}"

    return {
        "source": source,
        "source_id": source_id,
        "document_title": document_title,
        "page": page_display,
        "section": section,
        "chunk_index": chunk_index,
        "content_labels": content_labels,
    }


def candidate_id(document: Document) -> str:
    metadata = document.metadata
    return str(
        metadata.get("chunk_id")
        or metadata.get("content_hash")
        or f"{metadata.get('source_id')}:{metadata.get('chunk_index')}"
    )


def build_rerank_passage(document: Document) -> str:
    metadata = format_metadata(document.metadata)
    header_parts = [
        f"Document: {metadata['document_title']}",
        f"File: {metadata['source_id']}",
    ]
    if metadata["section"] != "not available":
        header_parts.append(f"Section: {metadata['section']}")
    if metadata["page"] != "not available":
        header_parts.append(f"Page: {metadata['page']}")
    if metadata["content_labels"]:
        header_parts.append(f"Content type: {metadata['content_labels']}")
    return "\n".join(header_parts) + "\n\n" + document.page_content.strip()


def documents_by_id(collection: Any, ids: Sequence[str]) -> dict[str, Document]:
    if not ids:
        return {}
    result = collection.get(ids=list(ids), include=["documents", "metadatas"])
    return {
        str(chunk_id): Document(
            page_content=str(content or ""),
            metadata=metadata or {},
        )
        for chunk_id, content, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
            strict=True,
        )
    }


REQUIREMENT_PATTERN = re.compile(
    r"\b(shall|must|required|requirement|requirements|requisito|requisitos|"
    r"deve|deverá|deverao|deverão|obrigatório|obrigatoria)\b",
    re.IGNORECASE,
)


def document_matches_filters(
    document: Document,
    filters: RetrievalFilters | None,
) -> bool:
    """Apply user-facing filters to a retrieved document."""

    if filters is None:
        return True
    metadata = document.metadata
    section = str(
        metadata.get("section_path") or metadata.get("section_title") or ""
    )
    if filters.section.strip().casefold() not in section.casefold():
        return False

    content_filter = filters.content_type.strip().casefold()
    labels = str(metadata.get("content_labels") or "")
    searchable = "\n".join(
        (
            labels,
            section,
            str(metadata.get("parent_content") or document.page_content),
        )
    )
    if content_filter == "table" and "table" not in labels.casefold():
        return False
    if content_filter == "requirement" and not REQUIREMENT_PATTERN.search(searchable):
        return False

    date_filter = filters.date.strip().casefold()
    if date_filter:
        date_values = " ".join(
            str(metadata.get(key) or "")
            for key in (
                "document_date",
                "date",
                "publication_date",
                "revision_date",
                "document_title",
                "source_id",
            )
        )
        # Front matter and headings commonly carry the only indexed date.
        date_values += " " + str(
            metadata.get("parent_content") or document.page_content
        )[:1200]
        if date_filter not in date_values.casefold():
            return False
    return True


def candidates_from_ids(
    collection: Any,
    chunk_ids: Sequence[str],
    filters: RetrievalFilters | None = None,
    source_filter: str | None = None,
) -> list[Candidate]:
    """Rebuild an ordered candidate set for search-within-results."""

    ordered_ids = list(dict.fromkeys(str(item) for item in chunk_ids if item))
    documents = documents_by_id(collection, ordered_ids)
    candidates: list[Candidate] = []
    for rank, chunk_id in enumerate(ordered_ids, start=1):
        document = documents.get(chunk_id)
        if document is None:
            continue
        if source_filter and str(document.metadata.get("source_id") or "") != source_filter:
            continue
        if not document_matches_filters(document, filters):
            continue
        candidates.append(
            Candidate(
                document=document,
                chunk_id=chunk_id,
                fusion_score=1.0 / (RRF_K + rank),
                retrieval_rank=rank,
            )
        )
    return candidates[:RERANK_CANDIDATES]


def retrieve_candidate_pool(
    runtime: Runtime,
    question: str,
    source_filter: str | None = None,
    filters: RetrievalFilters | None = None,
) -> tuple[list[Candidate], float, float]:
    chroma_filter = {"source_id": source_filter} if source_filter else None

    dense_start = perf_counter()
    dense_results = runtime.vectorstore.similarity_search_with_score(
        query=question,
        k=DENSE_CANDIDATES,
        filter=chroma_filter,
    )
    dense_time = perf_counter() - dense_start

    lexical_start = perf_counter()
    lexical_results = lexical_search(
        path=LEXICAL_DB_PATH,
        question=question,
        limit=LEXICAL_CANDIDATES,
        source_id=source_filter,
    )
    lexical_documents = documents_by_id(
        runtime.collection,
        [chunk_id for chunk_id, _ in lexical_results],
    )
    lexical_time = perf_counter() - lexical_start

    candidates: dict[str, Candidate] = {}
    for rank, (document, distance) in enumerate(dense_results, start=1):
        chunk_id = candidate_id(document)
        candidate = candidates.setdefault(
            chunk_id,
            Candidate(document=document, chunk_id=chunk_id),
        )
        candidate.vector_distance = float(distance)
        candidate.vector_rank = rank
        candidate.fusion_score += 1.0 / (RRF_K + rank)

    for rank, (chunk_id, lexical_score) in enumerate(lexical_results, start=1):
        document = lexical_documents.get(chunk_id)
        if document is None:
            continue
        candidate = candidates.setdefault(
            chunk_id,
            Candidate(document=document, chunk_id=chunk_id),
        )
        candidate.lexical_score = lexical_score
        candidate.lexical_rank = rank
        candidate.fusion_score += 1.0 / (RRF_K + rank)

    # Remove repeated content within one source while retaining the provenance
    # of whichever copy had the stronger fused retrieval score.
    deduplicated: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates.values():
        metadata = candidate.document.metadata
        key = (
            str(metadata.get("source_id", "")),
            str(metadata.get("content_hash") or candidate.chunk_id),
        )
        existing = deduplicated.get(key)
        if existing is None or candidate.fusion_score > existing.fusion_score:
            deduplicated[key] = candidate

    ranked = sorted(
        (
            candidate
            for candidate in deduplicated.values()
            if document_matches_filters(candidate.document, filters)
        ),
        key=lambda candidate: candidate.fusion_score,
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate.retrieval_rank = rank
    return ranked, dense_time, lexical_time


def retrieve_candidates(
    runtime: Runtime,
    question: str,
    source_filter: str | None = None,
    filters: RetrievalFilters | None = None,
) -> tuple[list[Candidate], float, float]:
    """Return the fused shortlist that will be scored by the cross-encoder."""

    pool, dense_time, lexical_time = retrieve_candidate_pool(
        runtime,
        question,
        source_filter,
        filters,
    )
    return pool[:RERANK_CANDIDATES], dense_time, lexical_time


def rerank_candidates(
    runtime: Runtime,
    question: str,
    candidates: list[Candidate],
) -> list[Candidate]:
    pairs = [
        (question, build_rerank_passage(candidate.document))
        for candidate in candidates
    ]
    with runtime.inference_lock:
        scores = runtime.reranker.predict(pairs)

    for candidate, (logit, probability, token_count, truncated) in zip(
        candidates,
        scores,
        strict=True,
    ):
        candidate.rerank_logit = logit
        candidate.rerank_probability = probability
        candidate.rerank_token_count = token_count
        candidate.rerank_truncated = truncated

    cross_encoder_ranked = sorted(
        candidates,
        key=lambda item: item.rerank_logit,
        reverse=True,
    )
    for rank, candidate in enumerate(cross_encoder_ranked, start=1):
        candidate.rerank_rank = rank
        rerank_component = 1.0 / (RRF_K + rank)
        retrieval_component = 1.0 / (RRF_K + candidate.retrieval_rank)
        candidate.final_score = (
            RERANK_WEIGHT * rerank_component
            + (1.0 - RERANK_WEIGHT) * retrieval_component
        )

    final_ranked = sorted(
        candidates,
        key=lambda item: item.final_score,
        reverse=True,
    )
    for rank, candidate in enumerate(final_ranked, start=1):
        candidate.final_rank = rank
    return final_ranked


def relevance_gate_thresholds() -> tuple[float, float, str] | None:
    """Return live best/result cutoffs and their source, if gating is active."""

    if ENABLE_RELEVANCE_GATE:
        return MIN_BEST_RERANK_LOGIT, MIN_RESULT_RERANK_LOGIT, "environment"
    try:
        calibration = WORKSPACE_STORE.calibration_status(
            min_positive=QUALITY_MIN_POSITIVE_LABELS,
            min_negative=QUALITY_MIN_NEGATIVE_LABELS,
            reranker_model=RERANKER_MODEL,
            reranker_fingerprint=reranker_fingerprint(),
        )
    except Exception:
        LOGGER.exception("Could not read the reference-quality calibration")
        return None
    threshold = calibration.get("threshold")
    if not calibration.get("active") or threshold is None:
        return None
    learned = float(threshold)
    return learned, learned, "feedback"


def select_results(ranked: list[Candidate]) -> list[Candidate]:
    if not ranked:
        return []
    gate = relevance_gate_thresholds()
    if (
        gate is not None
        and max(candidate.rerank_logit for candidate in ranked) < gate[0]
    ):
        return []

    selected: list[Candidate] = []
    section_counts: dict[tuple[str, str], int] = {}
    page_counts: dict[tuple[str, int], int] = {}
    for candidate in ranked:
        if (
            gate is not None
            and candidate.rerank_logit < gate[1]
        ):
            continue
        metadata = candidate.document.metadata
        section_key = (
            str(metadata.get("source_id", "")),
            str(metadata.get("section_path", "")),
        )
        if section_counts.get(section_key, 0) >= MAX_RESULTS_PER_SECTION:
            continue
        page = first_source_page(metadata)
        page_key = (str(metadata.get("source_id", "")), page or -1)
        if page is not None and page_counts.get(page_key, 0) >= MAX_RESULTS_PER_PAGE:
            continue
        if any(results_are_near_duplicates(candidate, item) for item in selected):
            continue
        section_counts[section_key] = section_counts.get(section_key, 0) + 1
        if page is not None:
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
        selected.append(candidate)
        if len(selected) >= RERANK_TOP_N:
            break
    return selected


def select_additional_results(
    ranked: Sequence[Candidate],
    selected: Sequence[Candidate],
) -> list[Candidate]:
    """Return further unique reranked passages without rerunning inference."""

    selected_ids = {candidate.chunk_id for candidate in selected}
    gate = relevance_gate_thresholds()
    accepted = list(selected)
    additional: list[Candidate] = []
    for candidate in ranked:
        if candidate.chunk_id in selected_ids:
            continue
        if (
            gate is not None
            and candidate.rerank_logit < gate[1]
        ):
            continue
        if any(results_are_near_duplicates(candidate, item) for item in accepted):
            continue
        accepted.append(candidate)
        additional.append(candidate)
        if len(additional) >= ADDITIONAL_RESULTS:
            break
    return additional


RESULT_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def context_text(document: Document) -> str:
    """Return the larger parent context when child retrieval is available."""

    parent = str(document.metadata.get("parent_content") or "").strip()
    return parent or document.page_content.strip()


def results_are_near_duplicates(left: Candidate, right: Candidate) -> bool:
    """Suppress repeated child hits and very similar excerpts from one source."""

    left_metadata = left.document.metadata
    right_metadata = right.document.metadata
    left_source = str(left_metadata.get("source_id") or "")
    right_source = str(right_metadata.get("source_id") or "")
    if not left_source or left_source != right_source:
        return False

    left_parent = str(left_metadata.get("parent_chunk_id") or "")
    right_parent = str(right_metadata.get("parent_chunk_id") or "")
    if left_parent and left_parent == right_parent:
        return True

    left_words = {
        word.casefold()
        for word in RESULT_WORD_PATTERN.findall(context_text(left.document))
        if len(word) > 2
    }
    right_words = {
        word.casefold()
        for word in RESULT_WORD_PATTERN.findall(context_text(right.document))
        if len(word) > 2
    }
    if not left_words or not right_words:
        return False
    similarity = len(left_words & right_words) / len(left_words | right_words)
    return similarity >= MAX_RESULT_TEXT_SIMILARITY


def truncate_text(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_CHARS_PER_RESULT:
        return text
    return text[:MAX_CHARS_PER_RESULT].rstrip() + "\n...[excerpt truncated]"


def markdown_quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def inline_code(value: object) -> str:
    return str(value).replace("`", "\\`")


def first_source_page(metadata: dict[str, Any]) -> int | None:
    """Return the first valid one-based page stored for a result."""

    for key in ("page_start", "page", "page_no", "page_number"):
        value = metadata.get(key)
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
    return None


def register_source_navigation(
    question: str,
    candidates: Sequence[Candidate],
) -> str | None:
    """Store a small, temporary ordered citation list for viewer navigation."""

    usable: list[Candidate] = []
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        metadata = candidate.document.metadata
        source_id = str(metadata.get("source_id") or "").strip().replace("\\", "/")
        if not source_id:
            continue
        page = first_source_page(metadata) or 1
        items.append(
            {
                "source_id": source_id,
                "page": page,
                "chunk_id": str(metadata.get("chunk_id") or candidate.chunk_id)[:200],
                "title": str(
                    metadata.get("document_title") or Path(source_id).stem
                ).strip(),
                "section": str(
                    metadata.get("section_path")
                    or metadata.get("section_title")
                    or ""
                ).strip(),
            }
        )
        usable.append(candidate)
    if not items:
        return None

    token = token_urlsafe(12)
    now = perf_counter()
    with _SOURCE_NAVIGATION_LOCK:
        expired = [
            key
            for key, value in _SOURCE_NAVIGATION_CACHE.items()
            if now - float(value.get("created", 0.0)) > SOURCE_NAVIGATION_TTL_SECONDS
        ]
        for key in expired:
            _SOURCE_NAVIGATION_CACHE.pop(key, None)
        while len(_SOURCE_NAVIGATION_CACHE) >= SOURCE_NAVIGATION_CACHE_SIZE:
            _SOURCE_NAVIGATION_CACHE.popitem(last=False)
        _SOURCE_NAVIGATION_CACHE[token] = {
            "created": now,
            "question": question.strip()[:500],
            "items": items,
        }

    for index, candidate in enumerate(usable):
        candidate.document.metadata["_viewer_nav_token"] = token
        candidate.document.metadata["_viewer_nav_index"] = index
    return token


def _source_navigation_item_url(
    item: dict[str, Any],
    question: str,
    token: str,
    index: int,
) -> str:
    source_id = str(item["source_id"])
    parameters: dict[str, Any] = {
        "page": int(item.get("page") or 1),
        "q": question[:500],
        "chunk": str(item.get("chunk_id") or "")[:200],
        "nav": token,
        "at": index,
    }
    return f"/viewer/{quote(source_id, safe='/')}?{urlencode(parameters)}"


def resolve_source_navigation(
    token: str,
    index: int,
    source_id: str,
    chunk_id: str,
) -> dict[str, Any]:
    """Return verified previous/next links for one cached ranked result list."""

    empty = {
        "previousSource": None,
        "nextSource": None,
        "sourcePosition": "",
    }
    token = token.strip()[:100]
    if not token or index < 0:
        return empty
    with _SOURCE_NAVIGATION_LOCK:
        entry = _SOURCE_NAVIGATION_CACHE.get(token)
        if entry is None:
            return empty
        if perf_counter() - float(entry.get("created", 0.0)) > SOURCE_NAVIGATION_TTL_SECONDS:
            _SOURCE_NAVIGATION_CACHE.pop(token, None)
            return empty
        _SOURCE_NAVIGATION_CACHE.move_to_end(token)
        question = str(entry.get("question") or "")
        items = list(entry.get("items") or [])
    if index >= len(items):
        return empty
    current = items[index]
    if str(current.get("source_id") or "") != source_id.replace("\\", "/"):
        return empty
    stored_chunk = str(current.get("chunk_id") or "")
    if chunk_id and stored_chunk != chunk_id:
        return empty

    def target(target_index: int) -> dict[str, str] | None:
        if target_index < 0 or target_index >= len(items):
            return None
        item = items[target_index]
        page = int(item.get("page") or 1)
        title = str(item.get("title") or item.get("source_id") or "Source")
        return {
            "url": _source_navigation_item_url(
                item,
                question,
                token,
                target_index,
            ),
            "label": f"{title} · page {page}",
        }

    return {
        "previousSource": target(index - 1),
        "nextSource": target(index + 1),
        "sourcePosition": f"Source {index + 1} of {len(items)}",
    }


def source_url(
    metadata: dict[str, Any],
    question: str | None = None,
) -> str | None:
    """Build a stable in-app PDF URL without exposing an absolute disk path."""

    source_id = str(metadata.get("source_id") or "").strip().replace("\\", "/")
    if not source_id:
        return None
    encoded_source = quote(source_id, safe="/")
    page = first_source_page(metadata)
    if question:
        parameters = {"page": page or 1, "q": question[:500]}
        chunk_id = str(metadata.get("chunk_id") or "").strip()
        if chunk_id:
            parameters["chunk"] = chunk_id[:200]
        nav_token = str(metadata.get("_viewer_nav_token") or "").strip()
        try:
            nav_index = int(metadata.get("_viewer_nav_index"))
        except (TypeError, ValueError):
            nav_index = -1
        if nav_token and nav_index >= 0:
            parameters["nav"] = nav_token[:100]
            parameters["at"] = nav_index
        return f"/viewer/{encoded_source}?{urlencode(parameters)}"
    page_fragment = f"#page={page}" if page is not None else ""
    return f"/sources/{encoded_source}{page_fragment}"


def source_citation(
    metadata: dict[str, Any],
    number: int,
    question: str | None = None,
) -> str:
    """Format one NotebookLM-style source link for a result block."""

    url = source_url(metadata, question)
    if url is None:
        return "Source unavailable"
    source_id = str(metadata.get("source_id") or "source")
    page = first_source_page(metadata)
    page_label = f", page {page}" if page is not None else ""
    return f"[Source {number} — {source_id}{page_label}]({url})"


def get_citation_collection() -> Any:
    """Open Chroma without loading the embedding or reranker models."""

    global _CITATION_COLLECTION
    if _RUNTIME is not None:
        return _RUNTIME.collection
    if _CITATION_COLLECTION is None:
        with _CITATION_COLLECTION_LOCK:
            if _CITATION_COLLECTION is None:
                client = chromadb.PersistentClient(path=str(DB_DIR))
                _CITATION_COLLECTION = client.get_collection(COLLECTION_NAME)
    return _CITATION_COLLECTION


def load_citation_record(
    source_id: str,
    chunk_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Load one indexed passage and verify that it belongs to the URL source."""

    chunk_id = chunk_id.strip()[:200]
    if not chunk_id:
        return None
    try:
        result = get_citation_collection().get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
    except Exception as error:
        LOGGER.warning("Could not load citation chunk %s: %s", chunk_id, error)
        return None
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    if not documents or not metadatas or not isinstance(metadatas[0], dict):
        return None
    metadata = dict(metadatas[0])
    stored_source = str(metadata.get("source_id") or "").replace("\\", "/")
    requested_source = source_id.replace("\\", "/")
    if stored_source != requested_source:
        LOGGER.warning("Rejected a citation chunk belonging to another source")
        return None
    return str(documents[0] or ""), metadata


def parse_citation_locations(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw_locations = metadata.get("locations_json") or []
    if isinstance(raw_locations, str):
        try:
            raw_locations = json.loads(raw_locations)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(raw_locations, list):
        return []
    return [item for item in raw_locations[:1000] if isinstance(item, dict)]


def bbox_to_percentages(
    bbox: dict[str, Any],
    page_width: float,
    page_height: float,
) -> dict[str, float] | None:
    """Convert Docling/PDF coordinates into a browser overlay rectangle."""

    try:
        left = float(bbox["left"])
        right = float(bbox["right"])
        top = float(bbox["top"])
        bottom = float(bbox["bottom"])
    except (KeyError, TypeError, ValueError):
        return None
    if page_width <= 0 or page_height <= 0:
        return None

    x1, x2 = sorted((left, right))
    origin = str(bbox.get("coord_origin") or "").casefold()
    bottom_left = "bottom" in origin or (not origin and top >= bottom)
    if bottom_left:
        y1 = page_height - max(top, bottom)
        y2 = page_height - min(top, bottom)
    else:
        y1, y2 = sorted((top, bottom))

    x1 = min(max(x1, 0.0), page_width)
    x2 = min(max(x2, 0.0), page_width)
    y1 = min(max(y1, 0.0), page_height)
    y2 = min(max(y2, 0.0), page_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "left": 100.0 * x1 / page_width,
        "top": 100.0 * y1 / page_height,
        "width": 100.0 * (x2 - x1) / page_width,
        "height": 100.0 * (y2 - y1) / page_height,
    }


def _build_citation_view_data_unlocked(
    source_path: Path,
    source_id: str,
    page: int,
    question: str,
    chunk_id: str,
) -> dict[str, Any]:
    """Combine the stored passage, PDF geometry, and viewer metadata."""

    record = load_citation_record(source_id, chunk_id) if chunk_id else None
    document_text, metadata = record or ("", {})
    raw_locations = parse_citation_locations(metadata)
    document = pdfium.PdfDocument(str(source_path))
    page_sizes: dict[int, tuple[float, float]] = {}
    try:
        page_count = max(1, len(document))
        initial_page = min(max(1, int(page)), page_count)
        for location in raw_locations:
            try:
                location_page = int(location.get("page"))
            except (TypeError, ValueError):
                continue
            if location_page < 1 or location_page > page_count:
                continue
            if location_page not in page_sizes:
                pdf_page = document[location_page - 1]
                try:
                    page_sizes[location_page] = tuple(
                        float(value) for value in pdf_page.get_size()
                    )
                finally:
                    pdf_page.close()
    finally:
        document.close()

    locations: list[dict[str, Any]] = []
    for location in raw_locations:
        bbox = location.get("bbox")
        try:
            location_page = int(location.get("page"))
        except (TypeError, ValueError):
            continue
        if not isinstance(bbox, dict) or location_page not in page_sizes:
            continue
        rectangle = bbox_to_percentages(bbox, *page_sizes[location_page])
        if rectangle is None:
            continue
        locations.append(
            {
                "page": location_page,
                "label": str(location.get("label") or "passage")[:80],
                **rectangle,
            }
        )

    if locations and not any(item["page"] == initial_page for item in locations):
        initial_page = locations[0]["page"]
    title = str(metadata.get("document_title") or Path(source_id).stem).strip()
    section = str(
        metadata.get("section_path") or metadata.get("section_title") or ""
    ).strip()
    excerpt = str(metadata.get("parent_content") or document_text).strip()
    page_start = first_source_page(metadata) or initial_page
    try:
        page_end = int(metadata.get("page_end") or page_start)
    except (TypeError, ValueError):
        page_end = page_start
    page_label = (
        f"page {page_start}"
        if page_end == page_start
        else f"pages {page_start}-{page_end}"
    )
    citation_parts = [title]
    if section:
        citation_parts.append(section)
    citation_parts.extend([source_id, page_label])
    encoded_source = quote(source_id.replace("\\", "/"), safe="/")
    document_date = first_existing(
        metadata,
        ["document_date", "date", "publication_date", "revision_date"],
        "",
    )
    if not document_date:
        date_match = re.search(
            r"\b(?:19|20)\d{2}\b",
            " ".join((title, source_id, excerpt[:800])),
        )
        document_date = date_match.group(0) if date_match else ""
    return {
        "sourceId": source_id,
        "encodedSource": encoded_source,
        "title": title,
        "section": section,
        "excerpt": excerpt,
        "contentType": str(metadata.get("content_labels") or "").strip(),
        "documentDate": document_date,
        "question": question.strip()[:500],
        "chunkId": chunk_id[:200],
        "pageStart": page_start,
        "pageEnd": page_end,
        "pageCount": page_count,
        "initialPage": initial_page,
        "locations": locations,
        "citationLabel": (
            ". ".join(part.rstrip(".") for part in citation_parts) + "."
        ),
        "pdfUrl": f"/sources/{encoded_source}",
        "hasExactRegions": bool(locations),
    }


def build_citation_view_data(
    source_path: Path,
    source_id: str,
    page: int,
    question: str,
    chunk_id: str,
) -> dict[str, Any]:
    with _PDFIUM_LOCK:
        return _build_citation_view_data_unlocked(
            source_path,
            source_id,
            page,
            question,
            chunk_id,
        )


def _json_for_html(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def source_viewer_html(
    source_id: str,
    page: int,
    question: str,
    citation_data: dict[str, Any] | None = None,
) -> str:
    """Return the side-by-side passage and exact-region PDF workspace."""

    page = max(1, page)
    question = question.strip()[:500]
    encoded_source = quote(source_id.replace("\\", "/"), safe="/")
    data = dict(citation_data or {})
    data.setdefault("sourceId", source_id)
    data.setdefault("encodedSource", encoded_source)
    data.setdefault("title", Path(source_id).stem)
    data.setdefault("section", "")
    data.setdefault("excerpt", "")
    data.setdefault("contentType", "")
    data.setdefault("documentDate", "")
    data.setdefault("question", question)
    data.setdefault("chunkId", "")
    data.setdefault("pageStart", page)
    data.setdefault("pageEnd", page)
    data.setdefault("pageCount", page)
    data.setdefault("initialPage", page)
    data.setdefault("locations", [])
    data.setdefault("citationLabel", f"{source_id}, page {page}.")
    data.setdefault("pdfUrl", f"/sources/{encoded_source}")
    data.setdefault("hasExactRegions", False)
    data.setdefault("previousSource", None)
    data.setdefault("nextSource", None)
    data.setdefault("sourcePosition", "")

    safe_source = html.escape(str(data["sourceId"]))
    safe_title = html.escape(str(data["title"]))
    safe_section = html.escape(str(data["section"]))
    safe_excerpt = html.escape(str(data["excerpt"]))
    safe_question = html.escape(str(data["question"]))
    section_block = f'<p class="section">{safe_section}</p>' if safe_section else ""
    query_block = (
        f'<div class="query"><span>Search</span>{safe_question}</div>'
        if safe_question
        else ""
    )
    region_label = (
        "Exact source regions" if data["hasExactRegions"] else "Page-level source"
    )
    region_class = "exact" if data["hasExactRegions"] else "page-only"

    def source_navigation_button(
        target: Any,
        text: str,
        relation: str,
    ) -> str:
        if isinstance(target, dict) and target.get("url"):
            safe_url = html.escape(str(target["url"]), quote=True)
            safe_label = html.escape(str(target.get("label") or text), quote=True)
            return (
                f'<a class="source-nav-button" href="{safe_url}" '
                f'title="{safe_label}" rel="{relation}">{text}</a>'
            )
        return (
            f'<span class="source-nav-button disabled" aria-disabled="true">'
            f"{text}</span>"
        )

    has_source_navigation = any(
        isinstance(data.get(key), dict) and data[key].get("url")
        for key in ("previousSource", "nextSource")
    )
    source_navigation = ""
    if has_source_navigation:
        source_navigation = (
            '<div class="source-navigation" aria-label="Retrieved source navigation">'
            + source_navigation_button(data["previousSource"], "← Previous source", "prev")
            + f'<span class="source-position">{html.escape(str(data["sourcePosition"]))}</span>'
            + source_navigation_button(data["nextSource"], "Next source →", "next")
            + "</div>"
        )

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>@@TITLE@@ — citation viewer</title>
  <style>
    :root { color-scheme:dark; --panel:#171a21; --panel2:#20242d; --line:#343a46; --ink:#f7f7f8; --muted:#aeb5c2; --accent:#8fb7ff; --highlight:#ffd84d; }
    * { box-sizing:border-box; }
    html, body { height:100%; }
    body { margin:0; overflow:hidden; font-family:Inter,ui-sans-serif,system-ui,sans-serif; background:#101217; color:var(--ink); }
    button, a { font:inherit; }
    button { border:1px solid var(--line); border-radius:9px; background:#292e38; color:var(--ink); padding:8px 11px; cursor:pointer; }
    button:hover:not(:disabled), .link-button:hover { background:#353c49; }
    button:disabled { opacity:.42; cursor:not-allowed; }
    a { color:var(--accent); }
    .shell { height:100vh; display:grid; grid-template-columns:minmax(270px,320px) minmax(0,1fr) 116px; }
    .source-panel { min-width:0; overflow:auto; padding:17px 18px 24px; background:var(--panel); border-right:1px solid var(--line); scrollbar-gutter:stable; }
    .back { display:inline-flex; margin-bottom:13px; text-decoration:none; font-size:.86rem; font-weight:650; }
    .source-navigation { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:6px; margin:0 0 17px; }
    .source-nav-button { display:inline-flex; justify-content:center; align-items:center; min-height:34px; padding:6px 8px; border:1px solid var(--line); border-radius:8px; background:#292e38; color:#dce7ff; text-decoration:none; font-size:.72rem; white-space:nowrap; }
    .source-nav-button:hover { background:#353c49; }
    .source-nav-button.disabled { color:#727987; opacity:.55; cursor:not-allowed; }
    .source-position { color:var(--muted); font-size:.72rem; text-align:center; white-space:nowrap; }
    .eyebrow { margin:0 0 7px; color:var(--muted); font-size:.69rem; font-weight:750; letter-spacing:.09em; text-transform:uppercase; }
    h1 { margin:0; font-size:1.08rem; line-height:1.35; }
    .section { margin:7px 0 0; color:#d4d9e2; font-size:.9rem; line-height:1.4; }
    .source-name { margin:6px 0 0; color:var(--muted); font-size:.73rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .status { display:inline-flex; align-items:center; gap:6px; margin:12px 0 0; padding:5px 8px; border-radius:999px; font-size:.7rem; font-weight:700; }
    .status::before { content:""; width:8px; height:8px; border-radius:50%; background:currentColor; }
    .status.exact { background:#263e34; color:#8de4b7; }
    .status.page-only { background:#413928; color:#f0c779; }
    .query { margin-top:11px; padding:7px 9px; border:1px solid #343b48; border-radius:8px; background:#20242d; color:#e6e9ee; font-size:.78rem; line-height:1.4; }
    .query span { display:inline; margin-right:6px; color:var(--muted); font-size:.65rem; font-weight:750; text-transform:uppercase; }
    .panel-heading { margin:18px 0 8px; color:var(--muted); font-size:.7rem; font-weight:750; letter-spacing:.07em; text-transform:uppercase; }
    .excerpt-shell { position:relative; }
    .excerpt { margin:0; white-space:pre-wrap; color:#e2e5ea; font-size:.86rem; line-height:1.58; }
    .excerpt.collapsed { max-height:17.2em; overflow:hidden; }
    .excerpt.collapsed::after { content:""; position:absolute; right:0; bottom:0; left:0; height:54px; background:linear-gradient(transparent,var(--panel)); pointer-events:none; }
    .excerpt-toggle { display:block; width:100%; margin:5px 0 0; padding:6px; border:0; background:transparent; color:var(--accent); font-size:.76rem; }
    .excerpt-toggle:hover { background:#232832; }
    .copy-row { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin:0 0 11px; }
    .copy-row button { padding:7px 8px; font-size:.78rem; }
    .copy-feedback { grid-column:1/-1; min-height:0; color:#9de3bb; font-size:.72rem; text-align:left; }
    .save-panel { margin-top:14px; border:1px solid var(--line); border-radius:10px; background:#20242d; }
    .save-panel summary { padding:10px 11px; color:#dce7ff; font-size:.8rem; font-weight:650; cursor:pointer; list-style:none; }
    .save-panel summary::-webkit-details-marker { display:none; }
    .save-panel summary::after { content:"+"; float:right; color:var(--muted); }
    .save-panel[open] summary::after { content:"−"; }
    .save-panel-body { display:grid; gap:8px; padding:0 11px 11px; }
    .save-panel label { color:var(--muted); font-size:.66rem; font-weight:750; letter-spacing:.06em; text-transform:uppercase; }
    .save-panel select,.save-panel textarea { width:100%; border:1px solid var(--line); border-radius:8px; background:#292e38; color:var(--ink); padding:8px; font:inherit; font-size:.8rem; }
    .save-panel textarea { min-height:68px; resize:vertical; line-height:1.4; }
    .save-actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .save-actions a { display:flex; align-items:center; justify-content:center; padding:8px; border:1px solid var(--line); border-radius:9px; background:#292e38; color:var(--ink); text-decoration:none; text-align:center; }
    .save-status { min-height:15px; color:#9de3bb; font-size:.72rem; }
    .document-column { min-width:0; min-height:0; display:flex; flex-direction:column; background:#292d35; }
    .toolbar { min-height:56px; display:grid; grid-template-columns:auto minmax(0,1fr) auto; align-items:center; gap:10px; padding:9px 12px; background:var(--panel2); border-bottom:1px solid var(--line); }
    .toolbar-group { display:flex; align-items:center; gap:7px; min-width:0; }
    .toolbar button { min-width:36px; padding:7px 9px; }
    .page-controls { justify-self:start; }
    .passage-controls { justify-self:center; }
    .toolbar-actions { justify-self:end; }
    .toolbar-status { color:#d6dbe3; font-size:.8rem; text-align:center; white-space:nowrap; }
    .select-hint { max-width:150px; overflow:hidden; color:#aebbd1; font-size:.7rem; text-align:right; text-overflow:ellipsis; white-space:nowrap; }
    .highlight-toggle { min-width:112px !important; color:#dce7ff; font-size:.72rem; white-space:nowrap; }
    .highlight-toggle[aria-pressed="false"] { color:var(--muted); background:#222630; }
    .link-button { display:inline-flex; padding:7px 9px; border:1px solid var(--line); border-radius:8px; color:#dce7ff; font-size:.78rem; text-decoration:none; white-space:nowrap; }
    .viewer-scroll { flex:1; min-height:0; overflow:auto; padding:24px; scroll-behavior:auto; overscroll-behavior:contain; scrollbar-gutter:stable; touch-action:pan-x pan-y; }
    .paper { position:relative; width:min(100%,980px); margin:0 auto; background:white; box-shadow:0 12px 40px rgba(0,0,0,.38); }
    .paper img { display:block; width:100%; height:auto; min-height:500px; background:#f4f4f4; user-select:none; -webkit-user-drag:none; }
    .overlay-layer { position:absolute; z-index:1; inset:0; pointer-events:none; }
    .match-box { position:absolute; border:0; border-radius:2px; background:rgba(255,205,36,.10); box-shadow:none; opacity:1; pointer-events:none; transition:opacity .2s,background .22s; }
    .match-box.active { z-index:2; background:rgba(255,205,36,.15); }
    body[data-highlight-mode="off"] .match-box { opacity:0; }
    .empty-overlay { position:absolute; top:18px; left:50%; transform:translateX(-50%); padding:7px 10px; border-radius:8px; background:rgba(27,31,38,.88); color:#e8ebf0; font-size:.82rem; white-space:nowrap; }
    .text-layer { position:absolute; z-index:3; inset:0; overflow:hidden; cursor:text; user-select:text; -webkit-user-select:text; touch-action:pan-x pan-y; }
    .text-line { position:absolute; margin:0; padding:0; color:transparent; white-space:pre; line-height:1; font-family:Arial,sans-serif; transform-origin:left top; cursor:text; user-select:text; -webkit-user-select:text; }
    .text-line::selection { color:transparent; background:rgba(57,132,255,.48); }
    .text-line::-moz-selection { color:transparent; background:rgba(57,132,255,.48); }
    .thumb-rail { overflow:auto; padding:10px 8px 18px; background:#181b22; border-left:1px solid var(--line); }
    .thumb-title { position:sticky; top:-10px; z-index:3; margin:-10px -8px 9px; padding:12px 8px 8px; background:#181b22; color:var(--muted); font-size:.62rem; font-weight:750; letter-spacing:.06em; text-align:center; text-transform:uppercase; }
    .thumbnail { position:relative; display:block; width:100%; margin:0 0 9px; padding:4px; background:#242933; border:2px solid transparent; border-radius:8px; }
    .thumbnail.active { border-color:var(--accent); background:#30394a; }
    .thumbnail.has-match::after { content:""; position:absolute; top:8px; right:8px; width:10px; height:10px; border-radius:50%; background:var(--highlight); box-shadow:0 0 0 2px #3b3212; }
    .thumbnail img { display:block; width:100%; min-height:86px; object-fit:contain; background:white; }
    .thumbnail span { display:block; padding:5px 1px 1px; color:#d7dbe2; font-size:.72rem; text-align:center; }
    [hidden] { display:none !important; }
    @media (max-width:1180px) { .shell{grid-template-columns:minmax(270px,310px) minmax(0,1fr)}.thumb-rail{display:none}.select-hint{display:none} }
    @media (max-width:760px) { body{overflow:auto}.shell{height:auto;min-height:100vh;grid-template-columns:1fr}.source-panel{max-height:none;border-right:0;border-bottom:1px solid var(--line)}.document-column{min-height:85vh}.toolbar{grid-template-columns:1fr auto}.passage-controls{grid-column:1/-1;grid-row:2;justify-self:center}.viewer-scroll{padding:10px} }
  </style>
</head>
<body data-highlight-mode="subtle">
  <div class="shell">
    <aside class="source-panel">
      <a class="back" href="/">← Back to results</a>
      @@SOURCE_NAVIGATION@@
      <p class="eyebrow">Retrieved source</p>
      <h1>@@TITLE@@</h1>
      @@SECTION@@
      <p class="source-name">@@SOURCE@@</p>
      <div class="status @@REGION_CLASS@@">@@REGION_LABEL@@</div>
      @@QUERY@@
      <p class="panel-heading">Matched passage</p>
      <div class="copy-row">
        <button id="copyCitation" type="button">Copy citation</button>
        <button id="copyExcerpt" type="button">Copy passage</button>
        <div class="copy-feedback" id="copyFeedback" aria-live="polite"></div>
      </div>
      <div class="excerpt-shell">
        <p class="excerpt collapsed" id="matchedExcerpt" tabindex="0">@@EXCERPT@@</p>
      </div>
      <button class="excerpt-toggle" id="excerptToggle" type="button">Show full passage</button>
      <details class="save-panel" id="savePanel">
        <summary id="saveSummary">Save passage or add a note</summary>
        <div class="save-panel-body">
          <label for="bookmarkCollection">Collection</label>
          <select id="bookmarkCollection"><option value="">No collection</option></select>
          <label for="bookmarkNote">Note</label>
          <textarea id="bookmarkNote" placeholder="Why this passage matters…"></textarea>
          <div class="save-actions">
            <button id="savePassage" type="button">Save passage</button>
            <a href="/workspace">Workspace</a>
          </div>
          <div class="save-status" id="saveStatus" aria-live="polite"></div>
        </div>
      </details>
    </aside>
    <main class="document-column">
      <nav class="toolbar" aria-label="Citation navigation">
        <div class="toolbar-group page-controls">
          <button id="previousPage" type="button" aria-label="Previous page" title="Previous page">←</button>
          <span class="toolbar-status" id="pageStatus"></span>
          <button id="nextPage" type="button" aria-label="Next page" title="Next page">→</button>
        </div>
        <div class="toolbar-group passage-controls" id="passageControls">
          <button id="previousMatch" type="button" aria-label="Previous matched region">← Match</button>
          <span class="toolbar-status" id="matchStatus"></span>
          <button id="nextMatch" type="button" aria-label="Next matched region">Match →</button>
        </div>
        <div class="toolbar-group toolbar-actions">
          <span class="select-hint" id="textStatus" aria-live="polite">Click match to copy</span>
          <button class="highlight-toggle" id="highlightToggle" type="button" aria-pressed="true" title="Toggle highlight · press H">Highlight: Subtle</button>
          <a class="link-button" id="directPdf" target="_blank" rel="noopener">Original PDF ↗</a>
        </div>
      </nav>
      <div class="viewer-scroll" id="viewerScroll" tabindex="0" aria-label="Scrollable PDF page">
        <div class="paper">
          <img id="pageImage" alt="PDF page" draggable="false">
          <div class="overlay-layer" id="overlayLayer"></div>
          <div class="text-layer" id="textLayer" aria-label="Selectable PDF text"></div>
        </div>
      </div>
    </main>
    <aside class="thumb-rail" aria-label="Page thumbnails">
      <div class="thumb-title">Nearby pages<br>Yellow = match</div>
      <div id="thumbnails"></div>
    </aside>
  </div>
  <script id="citationData" type="application/json">@@DATA@@</script>
  <script>
    const data = JSON.parse(document.getElementById('citationData').textContent);
    const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
    let currentPage = clamp(Number(data.initialPage) || 1, 1, Number(data.pageCount) || 1);
    let activeMatch = data.locations.findIndex(item => item.page === currentPage);
    let currentTextLines = [];
    let textRequestNumber = 0;
    let pageImageRetries = 0;
    if (activeMatch < 0 && data.locations.length) activeMatch = 0;
    const byId = id => document.getElementById(id);
    const pageImage = byId('pageImage');
    const overlayLayer = byId('overlayLayer');
    const textLayer = byId('textLayer');
    const textStatus = byId('textStatus');
    const highlightToggle = byId('highlightToggle');
    const pageStatus = byId('pageStatus');
    const matchStatus = byId('matchStatus');
    const previousPage = byId('previousPage');
    const nextPage = byId('nextPage');
    const previousMatch = byId('previousMatch');
    const nextMatch = byId('nextMatch');
    const passageControls = byId('passageControls');
    const directPdf = byId('directPdf');
    const thumbnails = byId('thumbnails');
    const viewerScroll = byId('viewerScroll');
    const matchedExcerpt = byId('matchedExcerpt');
    const excerptToggle = byId('excerptToggle');
    const bookmarkCollection = byId('bookmarkCollection');
    const bookmarkNote = byId('bookmarkNote');
    const savePassage = byId('savePassage');
    const savePanel = byId('savePanel');
    const saveSummary = byId('saveSummary');
    const saveStatus = byId('saveStatus');
    const imageUrl = (page, width) => `/page-image/${data.encodedSource}?page=${page}&width=${width}`;
    const textUrl = page => `/page-text/${data.encodedSource}?page=${page}`;
    const highlightModes = ['subtle','off'];
    const highlightPreferenceKey = 'citation-highlight-mode';

    function preferredHighlightMode() {
      try {
        const stored = window.localStorage.getItem(highlightPreferenceKey);
        if (highlightModes.includes(stored)) return stored;
        if (stored === 'strong') return 'subtle';
      } catch (_) {}
      return 'subtle';
    }

    function applyHighlightMode(mode, persist = false) {
      const normalized = highlightModes.includes(mode) ? mode : 'subtle';
      document.body.dataset.highlightMode = normalized;
      highlightToggle.textContent = normalized === 'off' ? 'Highlight: Off' : 'Highlight: Subtle';
      highlightToggle.setAttribute('aria-pressed',String(normalized !== 'off'));
      highlightToggle.setAttribute('aria-label', `PDF highlight: ${normalized}. Click to toggle.`);
      if (persist) {
        try { window.localStorage.setItem(highlightPreferenceKey,normalized); } catch (_) {}
      }
      return normalized;
    }

    function announceHighlightMode(mode) {
      textStatus.textContent = `Highlights: ${mode}`;
      setTimeout(() => {
        if (document.body.dataset.highlightMode === mode) {
          textStatus.textContent = 'Click match to copy';
        }
      },1200);
    }

    async function loadBookmarkContext() {
      if (!data.chunkId || !data.excerpt) {
        savePassage.disabled = true;
        saveSummary.textContent = 'Passage saving unavailable';
        saveStatus.textContent = 'Open an exact search result to save it.';
        return;
      }
      try {
        const response = await fetch(`/workspace/api/context?chunk=${encodeURIComponent(data.chunkId)}`);
        if (!response.ok) throw new Error('Workspace request failed');
        const payload = await response.json();
        (payload.collections || []).forEach(item => {
          const option = document.createElement('option');
          option.value = item.id;
          option.textContent = item.name;
          bookmarkCollection.appendChild(option);
        });
        if (payload.bookmark) {
          bookmarkCollection.value = payload.bookmark.collection_id || '';
          bookmarkNote.value = payload.bookmark.note || '';
          savePassage.textContent = 'Update saved passage';
          saveSummary.textContent = 'Saved passage · edit note or collection';
        }
      } catch (_) {
        saveStatus.textContent = 'Workspace is temporarily unavailable.';
      }
    }

    async function saveCurrentPassage() {
      if (!data.chunkId || !data.excerpt) return;
      savePassage.disabled = true;
      saveStatus.textContent = 'Saving…';
      try {
        const response = await fetch('/workspace/api/bookmarks',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({
            chunk_id:data.chunkId,
            source_id:data.sourceId,
            document_title:data.title,
            page_start:data.pageStart,
            page_end:data.pageEnd,
            section:data.section,
            content_type:data.contentType,
            document_date:data.documentDate,
            excerpt:data.excerpt,
            citation_label:data.citationLabel,
            citation_url:window.location.href,
            query:data.question,
            collection_id:bookmarkCollection.value || null,
            note:bookmarkNote.value
          })
        });
        if (!response.ok) throw new Error('Save failed');
        savePassage.textContent = 'Update saved passage';
        saveSummary.textContent = 'Saved passage · edit note or collection';
        saveStatus.textContent = 'Saved to your workspace';
      } catch (_) {
        saveStatus.textContent = 'Could not save this passage.';
      } finally {
        savePassage.disabled = false;
      }
    }

    function initializeExcerpt() {
      requestAnimationFrame(() => {
        if (matchedExcerpt.scrollHeight <= matchedExcerpt.clientHeight + 3) {
          matchedExcerpt.classList.remove('collapsed');
          excerptToggle.hidden = true;
        }
      });
    }

    function requestPageImage(page, retry = false) {
      if (!retry) pageImageRetries = 0;
      const cacheBuster = retry ? `&retry=${Date.now()}` : '';
      pageImage.src = `${imageUrl(page,1600)}${cacheBuster}`;
      pageImage.alt = `${data.sourceId}, page ${page}`;
    }

    function renderOverlays() {
      overlayLayer.replaceChildren();
      const pageMatches = data.locations.map((item,index) => ({...item,index})).filter(item => item.page === currentPage);
      if (!pageMatches.length) {
        const note = document.createElement('div');
        note.className = 'empty-overlay';
        note.textContent = data.locations.length ? 'No matched region on this page' : 'This source only has page-level provenance';
        overlayLayer.appendChild(note);
        return;
      }
      pageMatches.forEach(item => {
        const box = document.createElement('div');
        box.className = `match-box${item.index === activeMatch ? ' active' : ''}`;
        box.dataset.matchIndex = String(item.index);
        Object.assign(box.style,{left:`${item.left}%`,top:`${item.top}%`,width:`${item.width}%`,height:`${item.height}%`});
        box.title = item.label || 'Matched source region';
        box.setAttribute('aria-hidden', 'true');
        overlayLayer.appendChild(box);
      });
    }

    function layoutTextLayer() {
      textLayer.replaceChildren();
      if (!currentTextLines.length) return;
      const layerWidth = textLayer.clientWidth;
      const layerHeight = textLayer.clientHeight;
      if (!layerWidth || !layerHeight) return;
      const spans = [];
      const fragment = document.createDocumentFragment();
      currentTextLines.forEach(line => {
        const span = document.createElement('span');
        span.className = 'text-line';
        span.textContent = `${line.text}\n`;
        span.style.left = `${line.left}%`;
        span.style.top = `${line.top}%`;
        span.style.fontSize = `${Math.max(1, line.height * layerHeight / 100)}px`;
        fragment.appendChild(span);
        spans.push({span, line});
      });
      textLayer.appendChild(fragment);
      spans.forEach(({span, line}) => {
        const measuredWidth = span.getBoundingClientRect().width;
        const targetWidth = line.width * layerWidth / 100;
        if (measuredWidth > 0 && targetWidth > 0) {
          span.style.transform = `scaleX(${targetWidth / measuredWidth})`;
        }
      });
    }

    async function loadSelectableText(page) {
      const requestNumber = ++textRequestNumber;
      currentTextLines = [];
      textLayer.replaceChildren();
      textStatus.textContent = 'Loading selectable text…';
      try {
        const response = await fetch(textUrl(page));
        if (!response.ok) throw new Error(`Text request failed: ${response.status}`);
        const payload = await response.json();
        if (requestNumber !== textRequestNumber || page !== currentPage) return;
        currentTextLines = Array.isArray(payload.lines) ? payload.lines : [];
        layoutTextLayer();
        textStatus.textContent = currentTextLines.length
          ? 'Click match to copy'
          : 'No selectable text on this page';
      } catch (_) {
        if (requestNumber !== textRequestNumber) return;
        textStatus.textContent = 'Selectable text unavailable';
      }
    }

    function lineOverlapsRegion(line, region) {
      const horizontal = Math.min(line.left + line.width, region.left + region.width) - Math.max(line.left, region.left);
      const vertical = Math.min(line.top + line.height, region.top + region.height) - Math.max(line.top, region.top);
      return horizontal > 0 && vertical > Math.min(line.height, region.height) * 0.2;
    }

    function regionAtPoint(event) {
      const layerBounds = textLayer.getBoundingClientRect();
      if (!layerBounds.width || !layerBounds.height) return null;
      const x = 100 * (event.clientX - layerBounds.left) / layerBounds.width;
      const y = 100 * (event.clientY - layerBounds.top) / layerBounds.height;
      return data.locations
        .map((item,index) => ({...item,index}))
        .filter(item => item.page === currentPage && x >= item.left && x <= item.left + item.width && y >= item.top && y <= item.top + item.height)
        .sort((left,right) => left.width * left.height - right.width * right.height)[0] || null;
    }

    async function copyYellowRegionAtPoint(event) {
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed && selection.toString().trim()) return;
      const region = regionAtPoint(event);
      if (!region) return;

      activeMatch = region.index;
      renderOverlays();
      updateNavigation(false);
      const regionText = currentTextLines
        .filter(line => lineOverlapsRegion(line, region))
        .map(line => line.text)
        .join('\\n')
        .trim();
      const copied = await copyText(regionText || data.excerpt, 'Highlighted PDF passage copied.');
      if (copied) {
        textStatus.textContent = 'Copied yellow passage';
        const copiedPage = currentPage;
        setTimeout(() => {
          if (currentPage === copiedPage) textStatus.textContent = 'Click match to copy';
        }, 1600);
      }
    }

    function updateNavigation(scrollThumbnail = false) {
      pageStatus.textContent = `Page ${currentPage} of ${data.pageCount}`;
      previousPage.disabled = currentPage <= 1;
      nextPage.disabled = currentPage >= data.pageCount;
      directPdf.href = `${data.pdfUrl}#page=${currentPage}`;
      passageControls.hidden = data.locations.length <= 1;
      matchStatus.textContent = data.locations.length ? `${activeMatch + 1} of ${data.locations.length}` : 'No matches';
      previousMatch.disabled = !data.locations.length || activeMatch <= 0;
      nextMatch.disabled = !data.locations.length || activeMatch >= data.locations.length - 1;
      document.querySelectorAll('.thumbnail').forEach(item => item.classList.toggle('active',Number(item.dataset.page) === currentPage));
      const activeThumbnail = document.querySelector(`.thumbnail[data-page="${currentPage}"]`);
      if (scrollThumbnail) activeThumbnail?.scrollIntoView({block:'nearest',behavior:'auto'});
    }

    function setPage(page, choosePageMatch = true, scrollThumbnail = true) {
      currentPage = clamp(Number(page) || 1, 1, data.pageCount);
      if (choosePageMatch) {
        const firstOnPage = data.locations.findIndex(item => item.page === currentPage);
        if (firstOnPage >= 0) activeMatch = firstOnPage;
      }
      requestPageImage(currentPage);
      loadSelectableText(currentPage);
      renderOverlays();
      buildThumbnails();
      updateNavigation(scrollThumbnail);
      viewerScroll.scrollTo({top:0,behavior:'auto'});
    }

    function setMatch(index) {
      if (!data.locations.length) return;
      activeMatch = clamp(Number(index) || 0,0,data.locations.length - 1);
      currentPage = data.locations[activeMatch].page;
      requestPageImage(currentPage);
      loadSelectableText(currentPage);
      renderOverlays();
      buildThumbnails();
      updateNavigation(true);
      requestAnimationFrame(() => document.querySelector('.match-box.active')?.scrollIntoView({block:'center',behavior:'auto'}));
    }

    function buildThumbnails() {
      const matchedPages = new Set(data.locations.map(item => item.page));
      const visiblePages = new Set(matchedPages);
      for (let page = currentPage - 2; page <= currentPage + 2; page += 1) {
        if (page >= 1 && page <= data.pageCount) visiblePages.add(page);
      }
      const fragment = document.createDocumentFragment();
      [...visiblePages].sort((left,right) => left - right).forEach(page => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `thumbnail${matchedPages.has(page) ? ' has-match' : ''}`;
        button.dataset.page = String(page);
        button.setAttribute('aria-label', `Open page ${page}${matchedPages.has(page) ? ', contains a match' : ''}`);
        const image = document.createElement('img');
        image.loading = 'lazy'; image.src = imageUrl(page,150); image.alt = '';
        const label = document.createElement('span'); label.textContent = `Page ${page}`;
        button.append(image,label);
        button.addEventListener('click', () => setPage(page));
        fragment.appendChild(button);
      });
      thumbnails.replaceChildren(fragment);
    }

    async function copyText(value, message) {
      try {
        if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(value);
        else {
          const helper = document.createElement('textarea'); helper.value = value;
          Object.assign(helper.style,{position:'fixed',opacity:'0'}); document.body.appendChild(helper);
          helper.select(); document.execCommand('copy'); helper.remove();
        }
        byId('copyFeedback').textContent = message;
        return true;
      } catch (_) {
        byId('copyFeedback').textContent = 'Copy failed — select the text manually.';
        return false;
      }
    }

    previousPage.addEventListener('click', () => setPage(currentPage - 1));
    nextPage.addEventListener('click', () => setPage(currentPage + 1));
    previousMatch.addEventListener('click', () => setMatch(activeMatch - 1));
    nextMatch.addEventListener('click', () => setMatch(activeMatch + 1));
    highlightToggle.addEventListener('click', () => {
      const mode = applyHighlightMode(document.body.dataset.highlightMode === 'off' ? 'subtle' : 'off',true);
      announceHighlightMode(mode);
    });
    document.addEventListener('keydown', event => {
      const target = event.target;
      const editing = target instanceof HTMLElement && (
        target.isContentEditable || ['INPUT','TEXTAREA','SELECT'].includes(target.tagName)
      );
      if (editing || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (event.key.toLowerCase() !== 'h') return;
      const currentIndex = highlightModes.indexOf(document.body.dataset.highlightMode);
      const mode = applyHighlightMode(highlightModes[(currentIndex + 1) % highlightModes.length],true);
      announceHighlightMode(mode);
    });
    byId('copyCitation').addEventListener('click', () => copyText(`${data.citationLabel}\n${window.location.href}`,'Citation copied.'));
    byId('copyExcerpt').addEventListener('click', () => copyText(data.excerpt,'Excerpt copied.'));
    excerptToggle.addEventListener('click', () => {
      const collapsed = matchedExcerpt.classList.toggle('collapsed');
      excerptToggle.textContent = collapsed ? 'Show full passage' : 'Show less';
    });
    savePassage.addEventListener('click',saveCurrentPassage);
    textLayer.addEventListener('wheel', event => {
      if (event.ctrlKey) return;
      const scale = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewerScroll.clientHeight : 1;
      const vertical = event.shiftKey ? 0 : event.deltaY * scale;
      const horizontal = (event.deltaX + (event.shiftKey ? event.deltaY : 0)) * scale;
      viewerScroll.scrollBy({top:vertical,left:horizontal,behavior:'auto'});
      event.preventDefault();
    }, {passive:false});
    let pagePointerStart = null;
    textLayer.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      pagePointerStart = {x:event.clientX,y:event.clientY};
    });
    textLayer.addEventListener('pointerup', event => {
      if (!pagePointerStart) return;
      const distance = Math.hypot(event.clientX - pagePointerStart.x,event.clientY - pagePointerStart.y);
      pagePointerStart = null;
      if (distance <= 6) copyYellowRegionAtPoint(event);
    });
    textLayer.addEventListener('pointercancel', () => { pagePointerStart = null; });
    pageImage.addEventListener('load', () => {
      pageImageRetries = 0;
      layoutTextLayer();
    });
    pageImage.addEventListener('error', () => {
      const failedPage = currentPage;
      if (pageImageRetries >= 2) {
        textStatus.textContent = 'Page image failed to load — change page to retry';
        return;
      }
      pageImageRetries += 1;
      setTimeout(() => {
        if (currentPage === failedPage) requestPageImage(failedPage,true);
      }, 250 * pageImageRetries);
    });
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(layoutTextLayer, 100);
    });
    applyHighlightMode(preferredHighlightMode());
    setPage(currentPage,false,true);
    initializeExcerpt();
    loadBookmarkContext();
  </script>
</body>
</html>"""
    replacements = {
        "@@DATA@@": _json_for_html(data),
        "@@TITLE@@": safe_title,
        "@@SOURCE@@": safe_source,
        "@@SOURCE_NAVIGATION@@": source_navigation,
        "@@SECTION@@": section_block,
        "@@QUERY@@": query_block,
        "@@REGION_CLASS@@": region_class,
        "@@REGION_LABEL@@": region_label,
        "@@EXCERPT@@": safe_excerpt,
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def resolve_source_path(source_id: str) -> Path:
    """Resolve a source ID under PDF_DIR and reject traversal/non-PDF paths."""

    root = PDF_DIR.resolve()
    candidate = (root / source_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FileNotFoundError(source_id) from error
    if candidate.suffix.lower() != ".pdf" or not candidate.is_file():
        raise FileNotFoundError(source_id)
    return candidate


@lru_cache(maxsize=256)
def _render_pdf_page_png_cached(
    source_path: str,
    modified_ns: int,
    page: int,
    width: int,
) -> bytes:
    """Render one PDF page; modified_ns keeps cached images file-safe."""

    del modified_ns
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(source_path)
        try:
            if page < 1 or page > len(document):
                raise ValueError(f"PDF page {page} is out of range")
            pdf_page = document[page - 1]
            try:
                page_width, _ = pdf_page.get_size()
                bitmap = pdf_page.render(scale=width / float(page_width))
                try:
                    image = bitmap.to_pil()
                    output = io.BytesIO()
                    image.save(output, format="PNG", optimize=True)
                    return output.getvalue()
                finally:
                    bitmap.close()
            finally:
                pdf_page.close()
        finally:
            document.close()


def render_pdf_page_png(source_path: Path, page: int, width: int) -> bytes:
    width = min(max(int(width), 120), 2200)
    return _render_pdf_page_png_cached(
        str(source_path),
        source_path.stat().st_mtime_ns,
        int(page),
        width,
    )


def text_lines_from_pdfium_page(
    text_page: Any,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    """Group PDFium characters into positioned, selectable browser lines."""

    lines: list[dict[str, Any]] = []
    characters: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []

    def flush_line() -> None:
        text = "".join(characters).rstrip()
        if text.strip() and boxes:
            left = min(box[0] for box in boxes)
            bottom = min(box[1] for box in boxes)
            right = max(box[2] for box in boxes)
            top = max(box[3] for box in boxes)
            rectangle = bbox_to_percentages(
                {
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top,
                    "coord_origin": "BOTTOMLEFT",
                },
                page_width,
                page_height,
            )
            if rectangle is not None:
                lines.append({"text": text, **rectangle})
        characters.clear()
        boxes.clear()

    for index in range(int(text_page.count_chars())):
        character = str(text_page.get_text_range(index, 1) or "")
        if character in {"\r", "\n"}:
            flush_line()
            continue
        characters.append(character)
        if character.isspace():
            continue
        try:
            left, bottom, right, top = (
                float(value) for value in text_page.get_charbox(index)
            )
        except Exception:
            continue
        if right > left and top > bottom:
            boxes.append((left, bottom, right, top))
    flush_line()
    return lines[:2000]


@lru_cache(maxsize=256)
def _extract_pdf_text_lines_cached(
    source_path: str,
    modified_ns: int,
    page: int,
) -> dict[str, Any]:
    """Extract one page's text and glyph geometry without model inference."""

    del modified_ns
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(source_path)
        try:
            if page < 1 or page > len(document):
                raise ValueError(f"PDF page {page} is out of range")
            pdf_page = document[page - 1]
            try:
                page_width, page_height = (
                    float(value) for value in pdf_page.get_size()
                )
                text_page = pdf_page.get_textpage()
                try:
                    lines = text_lines_from_pdfium_page(
                        text_page,
                        page_width,
                        page_height,
                    )
                finally:
                    text_page.close()
            finally:
                pdf_page.close()
        finally:
            document.close()
    return {"page": page, "lines": lines}


def extract_pdf_text_lines(source_path: Path, page: int) -> dict[str, Any]:
    return _extract_pdf_text_lines_cached(
        str(source_path),
        source_path.stat().st_mtime_ns,
        int(page),
    )


def feedback_context(
    source_filter: str | None = None,
    filters: RetrievalFilters | None = None,
) -> dict[str, str]:
    filters = filters or RetrievalFilters()
    return {
        "source_filter": normalize_source_filter(source_filter) or "",
        "section_filter": filters.section,
        "content_filter": filters.content_type,
        "date_filter": filters.date,
    }


def feedback_controls_html(
    payload: dict[str, Any],
    judgments: Sequence[tuple[str, str]],
) -> str:
    encoded = base64.b64encode(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).decode("ascii")
    buttons = "".join(
        (
            '<button type="button" '
            f'class="rag-feedback-button rag-feedback-{html.escape(judgment, quote=True)}">'
            f'{html.escape(label)}</button>'
        )
        for judgment, label in judgments
    )
    return (
        '<div class="rag-feedback">'
        f'<span class="rag-feedback-payload">{encoded}</span>'
        '<span class="rag-feedback-label">Was this useful?</span>'
        f'{buttons}<span class="rag-feedback-status" aria-live="polite"></span>'
        '</div>'
    )


def candidate_feedback_payload(
    question: str,
    candidate: Candidate,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw_metadata = candidate.document.metadata
    metadata = format_metadata(raw_metadata)
    try:
        page_end = int(raw_metadata.get("page_end"))
    except (TypeError, ValueError):
        page_end = first_source_page(raw_metadata)
    return {
        "query": question,
        "chunk_id": candidate.chunk_id,
        "source_id": metadata["source_id"],
        "document_title": metadata["document_title"],
        "page_start": first_source_page(raw_metadata),
        "page_end": page_end,
        "section": metadata["section"],
        "excerpt": context_text(candidate.document)[:20000],
        "result_rank": candidate.final_rank,
        "rerank_logit": candidate.rerank_logit,
        "final_score": candidate.final_score,
        "reranker_model": RERANKER_MODEL,
        "reranker_fingerprint": reranker_fingerprint(),
        **(context or {}),
    }


def format_candidate_block(
    question: str,
    candidate: Candidate,
    heading: str = "Result",
    feedback_filters: dict[str, str] | None = None,
) -> str:
    """Format one reusable primary or expanded reranked result block."""

    raw_metadata = candidate.document.metadata
    metadata = format_metadata(raw_metadata)
    vector_rank = candidate.vector_rank or "—"
    lexical_rank = candidate.lexical_rank or "—"
    truncation_note = "yes" if candidate.rerank_truncated else "no"
    citation = source_citation(raw_metadata, candidate.final_rank, question)
    displayed_text = context_text(candidate.document)
    child_note = (
        "- Retrieved child: "
        f"`{inline_code(raw_metadata.get('child_index', 0))}` of "
        f"`{inline_code(raw_metadata.get('child_count', 1))}`\n"
        if raw_metadata.get("parent_chunk_id")
        else ""
    )
    quality_controls = feedback_controls_html(
        candidate_feedback_payload(question, candidate, feedback_filters),
        (
            ("relevant", "Relevant"),
            ("wrong_passage", "Wrong passage"),
            ("wrong_document", "Wrong document"),
        ),
    )
    return (
        f"#### {heading} {candidate.final_rank}\n\n"
        f"- Citation: **{citation}**\n"
        f"- File: `{inline_code(metadata['source_id'])}`\n"
        f"- Title: `{inline_code(metadata['document_title'])}`\n"
        f"- Page(s): `{inline_code(metadata['page'])}`\n"
        f"- Section: `{inline_code(metadata['section'])}`\n"
        f"- Chunk: `{inline_code(metadata['chunk_index'])}`\n"
        f"{child_note}"
        f"- Rerank logit: `{candidate.rerank_logit:.4f}`\n"
        f"- Normalized score: `{candidate.rerank_probability:.4f}`\n"
        f"- Retrieval rank / cross-encoder rank: "
        f"`{candidate.retrieval_rank}` / `{candidate.rerank_rank}`\n"
        f"- Blended final score: `{candidate.final_score:.6f}`\n"
        f"- Dense rank / lexical rank: `{vector_rank}` / `{lexical_rank}`\n"
        f"- Reranker input truncated: `{truncation_note}`\n\n"
        f"{markdown_quote(truncate_text(displayed_text))}\n\n"
        f"{quality_controls}"
    )


def format_results(
    question: str,
    candidates: Sequence[Candidate],
    timings: SearchTimings,
    considered: Sequence[Candidate],
    feedback_filters: dict[str, str] | None = None,
) -> str:
    truncation_count = sum(candidate.rerank_truncated for candidate in considered)
    timing_block = (
        "### Timing\n\n"
        f"- Dense retrieval: `{timings.dense:.2f}s`\n"
        f"- Lexical retrieval: `{timings.lexical:.2f}s`\n"
        f"- Reranking: `{timings.reranking:.2f}s`\n"
        f"- Total: `{timings.total:.2f}s`\n"
        f"- Reranker truncations: `{truncation_count}/{len(considered)}`"
    )

    blocks = [f"### Query\n\n{inline_code(question)}"]
    best_score = (
        max(candidate.rerank_logit for candidate in considered)
        if considered
        else None
    )
    query_feedback = feedback_controls_html(
        {
            "query": question,
            "chunk_id": "",
            "rerank_logit": best_score,
            **(feedback_filters or {}),
        },
        (("no_relevant_result", "No relevant result"),),
    )
    if not candidates:
        if considered and relevance_gate_thresholds() is not None:
            blocks.append(
                "### Result\n\nNo strong evidence was found above the calibrated threshold."
            )
        else:
            blocks.append("### Result\n\nNo matching passages were found.")
        blocks.append(query_feedback)
        blocks.append(timing_block)
        return "\n\n---\n\n".join(blocks)

    blocks.append(
        f"### Results\n\nReturned **{len(candidates)}** "
        "potentially relevant passages."
    )
    blocks.append(query_feedback)
    blocks.extend(
        format_candidate_block(
            question,
            candidate,
            feedback_filters=feedback_filters,
        )
        for candidate in candidates
    )
    blocks.append(timing_block)
    return "\n\n---\n\n".join(blocks)


def format_additional_results(
    question: str,
    candidates: Sequence[Candidate],
    feedback_filters: dict[str, str] | None = None,
) -> str:
    if not candidates:
        return ""
    blocks = [
        "### Additional reranked sources\n\n"
        f"Showing **{len(candidates)}** additional passages from the existing "
        "reranking pass."
    ]
    blocks.extend(
        format_candidate_block(
            question,
            candidate,
            "Reranked source",
            feedback_filters,
        )
        for candidate in candidates
    )
    return "\n\n---\n\n".join(blocks)


def debug_candidates(question: str, candidates: Sequence[Candidate]) -> None:
    if not DEBUG_RETRIEVAL:
        return
    LOGGER.info("Query: %s", question)
    for candidate in candidates[:15]:
        metadata = format_metadata(candidate.document.metadata)
        LOGGER.info(
            "rank=%d logit=%.4f probability=%.4f dense=%s lexical=%s "
            "truncated=%s source=%s page=%s section=%s",
            candidate.final_rank,
            candidate.rerank_logit,
            candidate.rerank_probability,
            candidate.vector_rank,
            candidate.lexical_rank,
            candidate.rerank_truncated,
            metadata["source_id"],
            metadata["page"],
            metadata["section"],
        )


def _search_with_additional_unlocked(
    question: str,
    source_filter: str | None = None,
    filters: RetrievalFilters | None = None,
    within_result_ids: Sequence[str] | None = None,
) -> tuple[
    str,
    str,
    list[Candidate],
    list[Candidate],
    dict[str, Any],
]:
    question = question.strip()
    if not question:
        return "Enter a query.", "", [], [], {}

    runtime = get_runtime()
    total_start = perf_counter()
    normalized_source = normalize_source_filter(source_filter)
    if within_result_ids:
        candidates = candidates_from_ids(
            runtime.collection,
            within_result_ids,
            filters,
            normalized_source,
        )
        dense_time = 0.0
        lexical_time = 0.0
    else:
        candidates, dense_time, lexical_time = retrieve_candidates(
            runtime=runtime,
            question=question,
            source_filter=normalized_source,
            filters=filters,
        )

    rerank_start = perf_counter()
    ranked = rerank_candidates(runtime, question, candidates)
    rerank_time = perf_counter() - rerank_start
    selected = select_results(ranked)
    additional = select_additional_results(ranked, selected)
    total_time = perf_counter() - total_start

    timings = SearchTimings(
        dense=dense_time,
        lexical=lexical_time,
        reranking=rerank_time,
        total=total_time,
    )
    debug_candidates(question, ranked)
    register_source_navigation(question, [*selected, *additional])
    quality_context = feedback_context(normalized_source, filters)
    output = format_results(
        question,
        selected,
        timings,
        ranked,
        quality_context,
    )
    additional_output = format_additional_results(
        question,
        additional,
        quality_context,
    )
    metrics = {
        "dense_seconds": dense_time,
        "lexical_seconds": lexical_time,
        "rerank_seconds": rerank_time,
        "total_seconds": total_time,
        "reranker_truncation_rate": (
            sum(candidate.rerank_truncated for candidate in ranked) / len(ranked)
            if ranked
            else 0.0
        ),
        "best_rerank_logit": (
            max(candidate.rerank_logit for candidate in ranked)
            if ranked
            else None
        ),
        "considered_count": len(ranked),
    }
    return output, additional_output, selected, additional, metrics


def search_with_additional(
    question: str,
    source_filter: str | None = None,
    filters: RetrievalFilters | None = None,
    within_result_ids: Sequence[str] | None = None,
) -> tuple[
    str,
    str,
    list[Candidate],
    list[Candidate],
    dict[str, Any],
]:
    if _DOCUMENT_MAINTENANCE.is_set():
        raise DocumentMaintenanceError(
            "The document index is being updated. Search will resume when it finishes."
        )
    _guard_search_during_ingestion()
    with _SEARCH_MAINTENANCE_LOCK:
        if _DOCUMENT_MAINTENANCE.is_set():
            raise DocumentMaintenanceError(
                "The document index is being updated. Search will resume when it finishes."
            )
        _guard_search_during_ingestion()
        return _search_with_additional_unlocked(
            question,
            source_filter,
            filters,
            within_result_ids,
        )


def search(
    question: str,
    source_filter: str | None = None,
) -> tuple[str, list[Candidate], dict[str, Any]]:
    """Compatibility wrapper used by evaluation and programmatic callers."""

    output, _, selected, _, metrics = search_with_additional(
        question,
        source_filter,
    )
    return output, selected, metrics


def additional_button_label(count: int, expanded: bool = False) -> str:
    action = "Hide" if expanded else "Show"
    return f"{action} {count} more reranked source{'s' if count != 1 else ''}"


def gradio_search(
    question: str,
    source_filter: str | None,
    section_filter: str = "",
    content_filter: str = "",
    date_filter: str = "",
    within_results: bool = False,
    previous_result_ids: Sequence[str] | None = None,
    include_result_state: bool = False,
) -> tuple[Any, ...]:
    result_ids: list[str] = []
    try:
        filters = RetrievalFilters(
            section=(section_filter or "").strip(),
            content_type=(content_filter or "").strip(),
            date=(date_filter or "").strip(),
        )
        if not any(
            (
                filters.section,
                filters.content_type,
                filters.date,
                within_results,
                previous_result_ids,
            )
        ):
            # Retain the original two-argument call for compatibility with
            # evaluation helpers and third-party wrappers.
            output, additional_output, selected, additional, _ = search_with_additional(
                question,
                source_filter,
            )
        else:
            output, additional_output, selected, additional, _ = search_with_additional(
                question,
                source_filter,
                filters=filters,
                within_result_ids=(previous_result_ids if within_results else None),
            )
        count = len(additional)
        result_ids = [candidate.chunk_id for candidate in [*selected, *additional]]
        try:
            WORKSPACE_STORE.record_search(
                {
                    "query": question,
                    "source_filter": normalize_source_filter(source_filter) or "",
                    "section_filter": filters.section,
                    "content_filter": filters.content_type,
                    "date_filter": filters.date,
                    "within_results": within_results and bool(previous_result_ids),
                    "result_count": len(selected) + len(additional),
                }
            )
        except Exception:
            LOGGER.exception("Could not record search history")
        response: tuple[Any, ...] = (
            output,
            gr.update(value=additional_output, visible=False),
            gr.update(
                value=additional_button_label(count),
                visible=count > 0,
            ),
            False,
            count,
        )
        return (*response, result_ids) if include_result_state else response
    except DocumentMaintenanceError as error:
        response = (
            str(error),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            False,
            0,
        )
        return (*response, result_ids) if include_result_state else response
    except Exception:
        LOGGER.exception("Search failed")
        response = (
            "Search failed. Check the application log for details and verify "
            "that ingestion completed with the same embedding configuration.",
            gr.update(value="", visible=False),
            gr.update(visible=False),
            False,
            0,
        )
        return (*response, result_ids) if include_result_state else response


def toggle_additional_sources(
    expanded: bool,
    count: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    next_expanded = not bool(expanded)
    return (
        gr.update(visible=next_expanded),
        gr.update(
            value=additional_button_label(int(count), next_expanded),
            visible=int(count) > 0,
        ),
        next_expanded,
    )


def normalize_source_filter(source_filter: str | None) -> str | None:
    """Treat the visible All documents choice as an unrestricted search."""

    normalized = (source_filter or "").strip()
    return normalized or None


def source_choices() -> list[tuple[str, str]]:
    runtime = get_runtime()
    result = runtime.collection.get(include=["metadatas"])
    labels: dict[str, str] = {}
    for metadata in result.get("metadatas") or []:
        metadata = metadata or {}
        source_id = str(metadata.get("source_id", ""))
        if not source_id:
            continue
        title = str(metadata.get("document_title") or source_id)
        labels[source_id] = f"{title} — {source_id}"
    return [("All documents", "")] + [
        (labels[source_id], source_id) for source_id in sorted(labels)
    ]


def candidate_api_payload(
    question: str,
    candidate: Candidate,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Serialize a reranked passage for the React client."""

    raw_metadata = candidate.document.metadata
    metadata = format_metadata(raw_metadata)
    page_start = first_source_page(raw_metadata)
    try:
        page_end = int(raw_metadata.get("page_end"))
    except (TypeError, ValueError):
        page_end = page_start
    section = "" if metadata["section"] == "not available" else metadata["section"]
    page_label = ""
    if page_start is not None:
        page_label = f"page {page_start}"
        if page_end and page_end != page_start:
            page_label = f"pages {page_start}-{page_end}"
    citation_parts = [metadata["document_title"], section, page_label]
    return {
        "chunk_id": candidate.chunk_id,
        "source_id": metadata["source_id"],
        "document_title": metadata["document_title"],
        "page_start": page_start,
        "page_end": page_end,
        "page_label": page_label,
        "section": section,
        "content_type": str(raw_metadata.get("content_labels") or ""),
        "document_date": str(raw_metadata.get("document_date") or ""),
        "excerpt": context_text(candidate.document),
        "citation_label": ". ".join(part for part in citation_parts if part),
        "citation_url": source_url(raw_metadata, question) or "",
        "result_rank": candidate.final_rank,
        "rerank_logit": candidate.rerank_logit,
        "rerank_probability": candidate.rerank_probability,
        "final_score": candidate.final_score,
        "retrieval_rank": candidate.retrieval_rank,
        "rerank_rank": candidate.rerank_rank,
        "dense_rank": candidate.vector_rank,
        "lexical_rank": candidate.lexical_rank,
        "rerank_truncated": candidate.rerank_truncated,
        "feedback": candidate_feedback_payload(
            question,
            candidate,
            filters,
        ),
    }


def record_search_history(
    question: str,
    source_filter: str | None,
    filters: RetrievalFilters,
    within_results: bool,
    result_count: int,
) -> None:
    try:
        WORKSPACE_STORE.record_search(
            {
                "query": question,
                "source_filter": normalize_source_filter(source_filter) or "",
                "section_filter": filters.section,
                "content_filter": filters.content_type,
                "date_filter": filters.date,
                "within_results": within_results,
                "result_count": result_count,
            }
        )
    except Exception:
        LOGGER.exception("Could not record search history")


def refresh_source_dropdown() -> dict[str, Any]:
    try:
        return gr.update(choices=source_choices())
    except DocumentMaintenanceError:
        return gr.update()
    except Exception:
        LOGGER.warning("Document choices are unavailable until ingestion completes")
        return gr.update(choices=[("All documents", "")], value="")


def create_demo() -> gr.Blocks:
    try:
        choices = source_choices()
    except Exception:
        LOGGER.warning("Starting search UI without an available document index")
        choices = [("All documents", "")]
    with gr.Blocks(title="Document Retrieval and Reranking") as demo:
        gr.HTML(
            """
            <style>
              .rag-app-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 24px;
                margin: 4px 0 22px;
              }
              .rag-app-header h1 {
                margin: 0;
                font-size: clamp(1.55rem, 2.2vw, 2rem);
                line-height: 1.15;
              }
              .rag-header-actions { display:flex; align-items:center; gap:9px; }
              .rag-workspace-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 48px;
                padding: 12px 22px;
                border: 1px solid #60a5fa;
                border-radius: 10px;
                background: #2563eb;
                color: #fff !important;
                font-size: 1rem;
                font-weight: 700;
                line-height: 1;
                text-decoration: none !important;
                white-space: nowrap;
                box-shadow: 0 7px 20px rgba(37, 99, 235, .24);
                transition: background .16s ease, transform .16s ease, box-shadow .16s ease;
              }
              .rag-workspace-button:hover {
                background: #3b82f6;
                transform: translateY(-1px);
                box-shadow: 0 9px 24px rgba(37, 99, 235, .32);
              }
              .rag-workspace-button:focus-visible {
                outline: 3px solid rgba(96, 165, 250, .45);
                outline-offset: 3px;
              }
              .rag-document-button {
                display:inline-flex;
                align-items:center;
                justify-content:center;
                min-height:48px;
                padding:12px 18px;
                border:1px solid #4b5565;
                border-radius:10px;
                background:#252a35;
                color:#f3f5f8 !important;
                font-size:.94rem;
                font-weight:700;
                text-decoration:none !important;
                white-space:nowrap;
              }
              .rag-document-button:hover { background:#333a48; }
              .rag-quality-button {
                display:inline-flex;
                align-items:center;
                justify-content:center;
                min-height:48px;
                padding:12px 18px;
                border:1px solid #5d6c86;
                border-radius:10px;
                background:#30384a;
                color:#f3f5f8 !important;
                font-size:.94rem;
                font-weight:700;
                text-decoration:none !important;
                white-space:nowrap;
              }
              .rag-quality-button:hover { background:#3b465d; }
              .rag-feedback {
                display:flex;
                flex-wrap:wrap;
                align-items:center;
                gap:7px;
                margin:14px 0 2px;
                padding:9px 10px;
                border:1px solid var(--border-color-primary,#3b4250);
                border-radius:9px;
                background:rgba(35,39,49,.72);
              }
              .rag-feedback-label,.rag-feedback-status {
                color:var(--body-text-color-subdued,#aab2c0);
                font-size:.78rem;
              }
              .rag-feedback-label { margin-right:2px; font-weight:650; }
              .rag-feedback-payload { display:none !important; }
              .rag-feedback-button {
                min-height:30px !important;
                padding:4px 9px !important;
                border:1px solid #495264 !important;
                border-radius:7px !important;
                background:#282e3a !important;
                color:inherit !important;
                font-size:.76rem !important;
              }
              .rag-feedback-button:hover { background:#343d4d !important; }
              .rag-feedback-button.selected {
                border-color:#67a3ff !important;
                background:#254a7a !important;
              }
              .rag-feedback-status { min-width:54px; color:#8de4b7; }
              @media (max-width: 680px) {
                .rag-app-header { align-items: stretch; flex-direction: column; gap: 14px; }
                .rag-header-actions { align-items:stretch; flex-direction:column; }
                .rag-workspace-button,.rag-document-button,.rag-quality-button { width: 100%; }
              }
            </style>
            <header class="rag-app-header">
              <h1>Document Retrieval and Reranking</h1>
              <div class="rag-header-actions">
                <a class="rag-document-button" href="/documents">Manage documents</a>
                <a class="rag-quality-button" href="/quality">Reference quality</a>
                <a class="rag-workspace-button" href="/workspace">Open research workspace</a>
              </div>
            </header>
            """
        )
        with gr.Row():
            source = gr.Dropdown(
                choices=choices,
                value="",
                label="Search scope",
            )
            question = gr.Textbox(
                label="Query",
                placeholder="Where does the document define Station Dwell Reaction Time?",
                scale=3,
            )
        with gr.Accordion("Filters and search within results", open=False):
            with gr.Row():
                section_filter = gr.Textbox(
                    label="Section contains",
                    placeholder="e.g. braking or 7.4.2",
                )
                content_filter = gr.Dropdown(
                    choices=[
                        ("Any content", ""),
                        ("Tables", "table"),
                        ("Requirements", "requirement"),
                    ],
                    value="",
                    label="Passage type",
                )
                date_filter = gr.Textbox(
                    label="Document date or year",
                    placeholder="e.g. 2026",
                )
            within_results = gr.Checkbox(
                label="Search within the current results",
                info=(
                    "Rerank only the passages returned by your previous search. "
                    "Turn this off to search the full library again."
                ),
                value=False,
            )
        search_button = gr.Button("Search", variant="primary")
        output = gr.Markdown()
        show_more_button = gr.Button(
            "Show more reranked sources",
            variant="secondary",
            visible=False,
        )
        additional_output = gr.Markdown(visible=False)
        additional_expanded = gr.State(False)
        additional_count = gr.State(0)
        previous_result_ids = gr.State([])
        include_result_state = gr.State(True)

        search_outputs = [
            output,
            additional_output,
            show_more_button,
            additional_expanded,
            additional_count,
            previous_result_ids,
        ]

        search_inputs = [
            question,
            source,
            section_filter,
            content_filter,
            date_filter,
            within_results,
            previous_result_ids,
            include_result_state,
        ]

        search_button.click(
            fn=gradio_search,
            inputs=search_inputs,
            outputs=search_outputs,
            concurrency_limit=1,
        )
        question.submit(
            fn=gradio_search,
            inputs=search_inputs,
            outputs=search_outputs,
            concurrency_limit=1,
        )
        show_more_button.click(
            fn=toggle_additional_sources,
            inputs=[additional_expanded, additional_count],
            outputs=[additional_output, show_more_button, additional_expanded],
            queue=False,
        )
        demo.load(
            fn=None,
            inputs=None,
            outputs=[
                question,
                source,
                section_filter,
                content_filter,
                date_filter,
            ],
            js="""() => {
              const params = new URLSearchParams(window.location.search);
              return [
                params.get('q') || '',
                params.get('source') || '',
                params.get('section') || '',
                params.get('type') || '',
                params.get('date') || ''
              ];
            }""",
        )
        demo.load(
            fn=refresh_source_dropdown,
            inputs=None,
            outputs=source,
            queue=False,
        )
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
              if (window.__ragFeedbackReady) return [];
              window.__ragFeedbackReady = true;
              document.addEventListener('click', async event => {
                const button = event.target.closest('.rag-feedback-button');
                if (!button) return;
                const panel = button.closest('.rag-feedback');
                if (!panel) return;
                const status = panel.querySelector('.rag-feedback-status');
                const buttons = [...panel.querySelectorAll('.rag-feedback-button')];
                buttons.forEach(item => { item.disabled = true; });
                status.textContent = 'Saving...';
                try {
                  const encoded = panel.querySelector('.rag-feedback-payload')?.textContent || '';
                  const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
                  const payload = JSON.parse(new TextDecoder().decode(bytes));
                  const judgmentClass = [...button.classList].find(name => name.startsWith('rag-feedback-') && name !== 'rag-feedback-button');
                  payload.judgment = judgmentClass?.slice('rag-feedback-'.length) || '';
                  const response = await fetch('/quality/api/feedback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                  });
                  if (!response.ok) throw new Error('Feedback was not saved');
                  buttons.forEach(item => {
                    const selected = item === button;
                    item.classList.toggle('selected', selected);
                    item.setAttribute('aria-pressed', String(selected));
                  });
                  status.textContent = 'Saved';
                } catch (_) {
                  status.textContent = 'Could not save';
                } finally {
                  buttons.forEach(item => { item.disabled = false; });
                }
              });
              return [];
            }""",
        )

    return demo.queue(default_concurrency_limit=1)


def frontend_page(fallback_html: str) -> Response:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )
    return HTMLResponse(fallback_html, headers={"Cache-Control": "no-store"})


def create_web_app(demo: gr.Blocks | None = None) -> FastAPI:
    """Create the FastAPI backend and serve the React UI when built."""

    app = FastAPI(title="Document Retrieval and Reranking")

    @app.get("/api/sources", include_in_schema=False)
    def list_search_sources() -> dict[str, Any]:
        try:
            return {
                "sources": [
                    {"label": label, "value": value}
                    for label, value in source_choices()
                ]
            }
        except DocumentMaintenanceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/search", include_in_schema=False)
    async def search_documents_api(request: Request) -> dict[str, Any]:
        payload = await request.json()
        question = str(payload.get("query") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Enter a search query")
        source_filter = str(payload.get("source_filter") or "").strip() or None
        filters = RetrievalFilters(
            section=str(payload.get("section_filter") or "").strip(),
            content_type=str(payload.get("content_filter") or "").strip(),
            date=str(payload.get("date_filter") or "").strip(),
        )
        within_results = bool(payload.get("within_results"))
        previous_ids = [
            str(item)[:200]
            for item in (payload.get("previous_result_ids") or [])[:100]
            if str(item).strip()
        ]
        try:
            _, _, selected, additional, metrics = search_with_additional(
                question,
                source_filter,
                filters=filters,
                within_result_ids=(previous_ids if within_results else None),
            )
        except DocumentMaintenanceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            LOGGER.exception("Search API failed")
            raise HTTPException(
                status_code=500,
                detail=(
                    "Search failed. Verify that ingestion used the current "
                    "embedding configuration."
                ),
            ) from error

        result_ids = [candidate.chunk_id for candidate in [*selected, *additional]]
        record_search_history(
            question,
            source_filter,
            filters,
            within_results and bool(previous_ids),
            len(result_ids),
        )
        context = feedback_context(source_filter, filters)
        gate = relevance_gate_thresholds()
        return {
            "query": question,
            "results": [
                candidate_api_payload(question, candidate, context)
                for candidate in selected
            ],
            "additional_results": [
                candidate_api_payload(question, candidate, context)
                for candidate in additional
            ],
            "result_ids": result_ids,
            "metrics": metrics,
            "gate": {
                "active": gate is not None,
                "source": gate[2] if gate else "",
                "threshold": gate[0] if gate else None,
                "no_strong_evidence": bool(
                    gate is not None
                    and metrics.get("considered_count")
                    and not selected
                ),
            },
            "query_feedback": {
                "query": question,
                "chunk_id": "",
                "rerank_logit": metrics.get("best_rerank_logit"),
                **context,
            },
        }

    @app.get(
        "/documents",
        response_class=Response,
        include_in_schema=False,
    )
    def document_library() -> Response:
        return frontend_page(document_manager_html())

    @app.get("/documents/api/list", include_in_schema=False)
    def list_managed_documents() -> dict[str, Any]:
        payload = DOCUMENT_REPOSITORY.summary()
        payload["job"] = document_job_snapshot()
        payload["app_instance_id"] = _APP_INSTANCE_ID
        payload["hardware"] = HARDWARE.as_dict()
        if CORPUS_SCALE.repository is DOCUMENT_REPOSITORY:
            payload["queue"] = CORPUS_SCALE.snapshot()
            payload["health"] = CORPUS_SCALE.health()
            payload["backups"] = CORPUS_SCALE.list_backups()
        return payload

    @app.post("/documents/api/upload", include_in_schema=False)
    async def upload_managed_document(
        request: Request,
        path: str,
        replace: bool = False,
    ) -> dict[str, Any]:
        if document_job_snapshot().get("running"):
            raise HTTPException(
                status_code=409,
                detail="Wait for the current index update to finish.",
            )
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_DOCUMENT_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="The PDF is too large.")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid upload size.") from None

        PDF_DIR.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".rag-upload-",
            suffix=".part",
            dir=str(PDF_DIR.parent),
        )
        temporary_path = Path(temporary_name)
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_DOCUMENT_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="The PDF is too large.")
                    handle.write(chunk)
            result = DOCUMENT_REPOSITORY.commit_upload(
                temporary_path,
                path,
                replace=replace,
            )
            if CORPUS_SCALE.repository is DOCUMENT_REPOSITORY:
                CORPUS_SCALE.invalidate_health()
            return result
        except DuplicateDocumentError as error:
            raise HTTPException(
                status_code=409,
                detail=f"Duplicate PDF. It is already stored as {error.duplicate_of}.",
            ) from error
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            temporary_path.unlink(missing_ok=True)

    @app.patch("/documents/api/item/{source_id:path}", include_in_schema=False)
    async def move_managed_document(
        source_id: str,
        request: Request,
    ) -> dict[str, str]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        payload = await request.json()
        try:
            result = DOCUMENT_REPOSITORY.move(
                source_id,
                str(payload.get("target") or ""),
            )
            if CORPUS_SCALE.repository is DOCUMENT_REPOSITORY:
                CORPUS_SCALE.invalidate_health()
            return result
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Document not found.") from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.delete("/documents/api/item/{source_id:path}", include_in_schema=False)
    def trash_managed_document(source_id: str) -> dict[str, str]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        try:
            result = DOCUMENT_REPOSITORY.trash(source_id)
            if CORPUS_SCALE.repository is DOCUMENT_REPOSITORY:
                CORPUS_SCALE.invalidate_health()
            return result
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Document not found.") from error

    @app.post(
        "/documents/api/trash/{trash_id}/restore",
        include_in_schema=False,
    )
    def restore_managed_document(trash_id: str) -> dict[str, str]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        try:
            result = DOCUMENT_REPOSITORY.restore(trash_id)
            if CORPUS_SCALE.repository is DOCUMENT_REPOSITORY:
                CORPUS_SCALE.invalidate_health()
            return result
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Trash entry not found.") from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.delete(
        "/documents/api/trash/{trash_id}",
        include_in_schema=False,
    )
    def permanently_delete_managed_document(trash_id: str) -> dict[str, bool]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        try:
            if not DOCUMENT_REPOSITORY.delete_forever(trash_id):
                raise HTTPException(status_code=404, detail="Trash entry not found.")
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if CORPUS_SCALE.repository is DOCUMENT_REPOSITORY:
            CORPUS_SCALE.invalidate_health()
        return {"deleted": True}

    @app.post("/documents/api/sync", include_in_schema=False)
    async def synchronize_managed_documents(request: Request) -> dict[str, Any]:
        payload = await request.json()
        force = bool(payload.get("force", False))
        summary = DOCUMENT_REPOSITORY.summary()
        if not force and not summary["counts"]["pending"]:
            raise HTTPException(status_code=400, detail="There are no pending changes.")
        try:
            return start_document_index_job(force=force)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/documents/api/restart", include_in_schema=False)
    def restart_application() -> dict[str, Any]:
        if document_job_snapshot().get("running"):
            raise HTTPException(
                status_code=409,
                detail="Wait for the current index update to finish before restarting.",
            )
        try:
            return schedule_application_restart()
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail="The app could not schedule its restart.",
            ) from error

    @app.post("/documents/api/queue/pause", include_in_schema=False)
    def pause_corpus_queue() -> dict[str, Any]:
        job = document_job_snapshot()
        if not job.get("running"):
            raise HTTPException(status_code=409, detail="The ingestion queue is not running.")
        queue = CORPUS_SCALE.request_pause()
        _update_document_job(
            pause_requested=True,
            message="Pause requested. Finishing the current document safely.",
        )
        return {"queue": queue, "job": document_job_snapshot()}

    @app.post("/documents/api/queue/resume", include_in_schema=False)
    def resume_corpus_queue() -> dict[str, Any]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="The ingestion queue is already running.")
        queue = CORPUS_SCALE.snapshot()
        if not queue.get("remaining"):
            raise HTTPException(status_code=400, detail="There is no paused work to resume.")
        try:
            return start_document_index_job(
                force=bool(document_job_snapshot().get("force")),
                resume=True,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/documents/api/quarantine/{quarantine_id}/restore",
        include_in_schema=False,
    )
    def restore_quarantined_document(quarantine_id: str) -> dict[str, str]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        try:
            result = DOCUMENT_REPOSITORY.restore_quarantine(quarantine_id)
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Quarantine entry not found.") from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        CORPUS_SCALE.invalidate_health()
        return result

    @app.delete(
        "/documents/api/quarantine/{quarantine_id}",
        include_in_schema=False,
    )
    def delete_quarantined_document(quarantine_id: str) -> dict[str, bool]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        try:
            if not DOCUMENT_REPOSITORY.delete_quarantine(quarantine_id):
                raise HTTPException(status_code=404, detail="Quarantine entry not found.")
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        CORPUS_SCALE.invalidate_health()
        return {"deleted": True}

    @app.post(
        "/documents/api/revisions/{revision_id}/restore",
        include_in_schema=False,
    )
    def restore_document_revision(revision_id: str) -> dict[str, str]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        try:
            result = DOCUMENT_REPOSITORY.restore_revision(revision_id)
        except DocumentPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Revision not found.") from error
        CORPUS_SCALE.invalidate_health()
        return result

    @app.get("/documents/api/health", include_in_schema=False)
    def corpus_health(refresh: bool = False) -> dict[str, Any]:
        return CORPUS_SCALE.health(refresh=refresh)

    @app.get("/documents/api/backups", include_in_schema=False)
    def list_corpus_backups() -> dict[str, Any]:
        return {"backups": CORPUS_SCALE.list_backups()}

    @app.post("/documents/api/backups", include_in_schema=False)
    async def create_corpus_backup(request: Request) -> dict[str, Any]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Pause or finish ingestion first.")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError):
            payload = {}
        _DOCUMENT_MAINTENANCE.set()
        try:
            with _SEARCH_MAINTENANCE_LOCK:
                _release_search_runtime()
                return CORPUS_SCALE.create_backup(str(payload.get("label") or ""))
        except (OSError, sqlite3.Error, zipfile.BadZipFile) as error:
            LOGGER.exception("Corpus backup failed")
            raise HTTPException(status_code=500, detail=f"Backup failed: {error}") from error
        finally:
            _DOCUMENT_MAINTENANCE.clear()

    @app.post(
        "/documents/api/backups/{backup_id}/restore",
        include_in_schema=False,
    )
    def restore_corpus_backup(backup_id: str) -> dict[str, Any]:
        global _APP_INSTANCE_ID

        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Pause or finish ingestion first.")
        _DOCUMENT_MAINTENANCE.set()
        try:
            with _SEARCH_MAINTENANCE_LOCK:
                _release_search_runtime()
                result = CORPUS_SCALE.restore_backup(backup_id)
                _APP_INSTANCE_ID = token_urlsafe(8)
                return result
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Backup not found.") from error
        except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
            LOGGER.exception("Corpus restore failed")
            raise HTTPException(status_code=400, detail=f"Restore failed: {error}") from error
        finally:
            _DOCUMENT_MAINTENANCE.clear()

    @app.delete(
        "/documents/api/backups/{backup_id}",
        include_in_schema=False,
    )
    def delete_corpus_backup(backup_id: str) -> dict[str, bool]:
        if document_job_snapshot().get("running"):
            raise HTTPException(status_code=409, detail="Index update in progress.")
        try:
            if not CORPUS_SCALE.delete_backup(backup_id):
                raise HTTPException(status_code=404, detail="Backup not found.")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"deleted": True}

    @app.get(
        "/quality",
        response_class=Response,
        include_in_schema=False,
    )
    def reference_quality() -> Response:
        summary = WORKSPACE_STORE.quality_summary(
            min_positive=QUALITY_MIN_POSITIVE_LABELS,
            min_negative=QUALITY_MIN_NEGATIVE_LABELS,
            reranker_model=RERANKER_MODEL,
            reranker_fingerprint=reranker_fingerprint(),
        )
        return frontend_page(
            quality_dashboard_html(
                summary,
                WORKSPACE_STORE.list_feedback(limit=300),
            )
        )

    @app.get("/quality/api/state", include_in_schema=False)
    def reference_quality_state() -> dict[str, Any]:
        return {
            "summary": WORKSPACE_STORE.quality_summary(
                min_positive=QUALITY_MIN_POSITIVE_LABELS,
                min_negative=QUALITY_MIN_NEGATIVE_LABELS,
                reranker_model=RERANKER_MODEL,
                reranker_fingerprint=reranker_fingerprint(),
            ),
            "feedback": WORKSPACE_STORE.list_feedback(limit=300),
        }

    @app.post("/quality/api/feedback", include_in_schema=False)
    async def save_retrieval_feedback(request: Request) -> dict[str, Any]:
        payload = dict(await request.json())
        payload["reranker_model"] = RERANKER_MODEL
        payload["reranker_fingerprint"] = reranker_fingerprint()
        try:
            feedback = WORKSPACE_STORE.upsert_feedback(
                payload,
                min_positive=QUALITY_MIN_POSITIVE_LABELS,
                min_negative=QUALITY_MIN_NEGATIVE_LABELS,
                min_recall=QUALITY_MIN_RECALL,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "feedback": feedback,
            "calibration": WORKSPACE_STORE.calibration_status(
                min_positive=QUALITY_MIN_POSITIVE_LABELS,
                min_negative=QUALITY_MIN_NEGATIVE_LABELS,
                reranker_model=RERANKER_MODEL,
                reranker_fingerprint=reranker_fingerprint(),
            ),
        }

    @app.delete("/quality/api/feedback/{feedback_id}", include_in_schema=False)
    def delete_retrieval_feedback(feedback_id: int) -> dict[str, bool]:
        deleted = WORKSPACE_STORE.delete_feedback(
            feedback_id,
            min_positive=QUALITY_MIN_POSITIVE_LABELS,
            min_negative=QUALITY_MIN_NEGATIVE_LABELS,
            min_recall=QUALITY_MIN_RECALL,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return {"deleted": True}

    @app.post("/quality/api/calibrate", include_in_schema=False)
    def recalibrate_reference_quality() -> dict[str, Any]:
        return WORKSPACE_STORE.calibrate_feedback(
            min_positive=QUALITY_MIN_POSITIVE_LABELS,
            min_negative=QUALITY_MIN_NEGATIVE_LABELS,
            min_recall=QUALITY_MIN_RECALL,
            reranker_model=RERANKER_MODEL,
            reranker_fingerprint=reranker_fingerprint(),
        )

    @app.patch("/quality/api/calibration", include_in_schema=False)
    async def update_reference_quality_calibration(
        request: Request,
    ) -> dict[str, Any]:
        payload = await request.json()
        if "enabled" not in payload:
            raise HTTPException(status_code=400, detail="enabled is required")
        WORKSPACE_STORE.set_calibration_enabled(
            bool(payload["enabled"]),
            reranker_model=RERANKER_MODEL,
            reranker_fingerprint=reranker_fingerprint(),
        )
        return WORKSPACE_STORE.calibration_status(
            min_positive=QUALITY_MIN_POSITIVE_LABELS,
            min_negative=QUALITY_MIN_NEGATIVE_LABELS,
            reranker_model=RERANKER_MODEL,
            reranker_fingerprint=reranker_fingerprint(),
        )

    @app.get("/quality/export/benchmark", include_in_schema=False)
    def export_feedback_benchmark() -> Response:
        return Response(
            WORKSPACE_STORE.benchmark_jsonl(),
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="reference-quality.jsonl"'
            },
        )

    @app.get(
        "/workspace",
        response_class=Response,
        include_in_schema=False,
    )
    def research_workspace() -> Response:
        return frontend_page(
            workspace_html(
                WORKSPACE_STORE.list_bookmarks(),
                WORKSPACE_STORE.list_collections(),
                WORKSPACE_STORE.list_history(),
            )
        )

    @app.get("/workspace/api/state", include_in_schema=False)
    def workspace_state() -> dict[str, Any]:
        return {
            "bookmarks": WORKSPACE_STORE.list_bookmarks(),
            "collections": WORKSPACE_STORE.list_collections(),
            "history": WORKSPACE_STORE.list_history(),
        }

    @app.get("/workspace/api/context", include_in_schema=False)
    def workspace_context(chunk: str = "") -> dict[str, Any]:
        return {
            "bookmark": WORKSPACE_STORE.get_bookmark(chunk_id=chunk[:200]),
            "collections": WORKSPACE_STORE.list_collections(),
        }

    @app.post("/workspace/api/collections", include_in_schema=False)
    async def create_workspace_collection(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            return WORKSPACE_STORE.create_collection(
                str(payload.get("name") or ""),
                str(payload.get("description") or ""),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/workspace/api/bookmarks", include_in_schema=False)
    async def save_workspace_bookmark(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            return WORKSPACE_STORE.upsert_bookmark(dict(payload))
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.patch("/workspace/api/bookmarks/{bookmark_id}", include_in_schema=False)
    async def update_workspace_bookmark(
        bookmark_id: int,
        request: Request,
    ) -> dict[str, Any]:
        payload = await request.json()
        kwargs: dict[str, Any] = {}
        if "note" in payload:
            kwargs["note"] = str(payload.get("note") or "")
        if "collection_id" in payload:
            kwargs["collection_id"] = payload.get("collection_id")
        try:
            return WORKSPACE_STORE.update_bookmark(bookmark_id, **kwargs)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Bookmark not found") from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/workspace/api/bookmarks/{bookmark_id}", include_in_schema=False)
    def delete_workspace_bookmark(bookmark_id: int) -> dict[str, bool]:
        if not WORKSPACE_STORE.delete_bookmark(bookmark_id):
            raise HTTPException(status_code=404, detail="Bookmark not found")
        return {"deleted": True}

    @app.get("/workspace/export/markdown", include_in_schema=False)
    def export_workspace_markdown(ids: str = "") -> Response:
        bookmark_ids = parse_export_ids(ids)
        if not bookmark_ids:
            raise HTTPException(status_code=400, detail="Select at least one passage")
        return Response(
            WORKSPACE_STORE.markdown_export(bookmark_ids),
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="research-excerpts.md"'
            },
        )

    @app.get("/workspace/export/word", include_in_schema=False)
    def export_workspace_word(ids: str = "") -> Response:
        bookmark_ids = parse_export_ids(ids)
        if not bookmark_ids:
            raise HTTPException(status_code=400, detail="Select at least one passage")
        return Response(
            WORKSPACE_STORE.docx_export(bookmark_ids),
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            headers={
                "Content-Disposition": 'attachment; filename="research-excerpts.docx"'
            },
        )

    @app.get(
        "/sources/{source_id:path}",
        response_class=FileResponse,
        include_in_schema=False,
    )
    def serve_source(source_id: str) -> FileResponse:
        try:
            source_path = resolve_source_path(source_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="PDF source not found") from error
        return FileResponse(
            source_path,
            media_type="application/pdf",
            filename=source_path.name,
            content_disposition_type="inline",
        )

    @app.get(
        "/page-image/{source_id:path}",
        response_class=Response,
        include_in_schema=False,
    )
    def serve_page_image(
        source_id: str,
        page: int = 1,
        width: int = 1600,
    ) -> Response:
        try:
            source_path = resolve_source_path(source_id)
            png = render_pdf_page_png(source_path, page, width)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="PDF source not found") from error
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get(
        "/page-text/{source_id:path}",
        response_class=JSONResponse,
        include_in_schema=False,
    )
    def serve_page_text(source_id: str, page: int = 1) -> JSONResponse:
        try:
            source_path = resolve_source_path(source_id)
            payload = extract_pdf_text_lines(source_path, page)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="PDF source not found") from error
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return JSONResponse(
            content=payload,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get(
        "/viewer/{source_id:path}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def view_source(
        source_id: str,
        page: int = 1,
        q: str = "",
        chunk: str = "",
        nav: str = "",
        at: int = -1,
    ) -> HTMLResponse:
        try:
            source_path = resolve_source_path(source_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="PDF source not found") from error
        citation_data = build_citation_view_data(
            source_path,
            source_id,
            page,
            q,
            chunk,
        )
        citation_data.update(
            resolve_source_navigation(nav, at, source_id, chunk)
        )
        return HTMLResponse(
            source_viewer_html(source_id, page, q, citation_data),
            headers={"Cache-Control": "no-store"},
        )

    frontend_index = FRONTEND_DIST_DIR / "index.html"
    frontend_assets = FRONTEND_DIST_DIR / "assets"
    if frontend_index.is_file():
        if frontend_assets.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=frontend_assets),
                name="frontend-assets",
            )

        @app.get("/", response_class=FileResponse, include_in_schema=False)
        def frontend_home() -> FileResponse:
            return FileResponse(
                frontend_index,
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )

        return app

    fallback_demo = demo or create_demo()
    return gr.mount_gradio_app(app, fallback_demo, path="/")


def launch_gradio() -> None:
    app = create_web_app()
    if OPEN_BROWSER:
        browser_host = "127.0.0.1" if SERVER_HOST in {"0.0.0.0", "::"} else SERVER_HOST
        threading.Timer(
            1.5,
            lambda: webbrowser.open(f"http://{browser_host}:{SERVER_PORT}/"),
        ).start()
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    launch_gradio()
