"""Shared configuration and embedding utilities for the local RAG project."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from hardware import (
    accelerator_info,
    configured_accelerator,
    resolve_accelerator,
    torch_device_for,
)


PROJECT_DIR = Path(__file__).resolve().parent


def _configured_path(environment_name: str, default_name: str) -> Path:
    configured = Path(os.environ.get(environment_name, default_name)).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_DIR / configured
    return configured.resolve()


PDF_DIR = _configured_path("RAG_PDF_DIR", "docs")
DB_DIR = _configured_path("RAG_DB_DIR", "chroma_db")
DEBUG_DIR = _configured_path("RAG_DEBUG_DIR", "ingestion_debug")
MANIFEST_PATH = _configured_path("RAG_MANIFEST_PATH", "chroma_db/ingestion_manifest.json")
LEXICAL_DB_PATH = _configured_path("RAG_LEXICAL_DB_PATH", "chroma_db/lexical.sqlite3")

COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "technical_docs_v3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "embeddinggemma:latest")

# PyTorch ROCm intentionally uses the same ``torch.cuda`` device strings as
# NVIDIA CUDA. Keep the vendor backend separate from the physical device name.
REQUESTED_ACCELERATOR = configured_accelerator()
COMPUTE_BACKEND = resolve_accelerator(REQUESTED_ACCELERATOR)
COMPUTE_DEVICE = torch_device_for(COMPUTE_BACKEND)
HARDWARE = accelerator_info(REQUESTED_ACCELERATOR)


def _component_device(environment_name: str) -> str:
    requested = os.environ.get(environment_name, "auto").strip().lower()
    if requested in {"", "auto"}:
        return COMPUTE_DEVICE
    if requested in {"amd", "hip", "rocm"}:
        if COMPUTE_BACKEND != "rocm":
            raise RuntimeError(
                f"{environment_name}=rocm requires a ROCm PyTorch installation."
            )
        return "cuda:0"
    if requested in {"nvidia"}:
        if COMPUTE_BACKEND != "cuda":
            raise RuntimeError(
                f"{environment_name}=nvidia requires a CUDA PyTorch installation."
            )
        return "cuda:0"
    if requested == "cuda":
        requested = "cuda:0"
    if requested == "cpu" or re.fullmatch(r"cuda:\d+", requested):
        if requested.startswith("cuda") and COMPUTE_BACKEND == "cpu":
            raise RuntimeError(
                f"{environment_name} requested a GPU, but PyTorch is CPU-only."
            )
        return requested
    raise ValueError(
        f"{environment_name} must be auto, cpu, cuda[:N], nvidia, or rocm."
    )


DOCLING_DEVICE = _component_device("RAG_DOCLING_DEVICE")
RERANKER_DEVICE = _component_device("RAG_RERANKER_DEVICE")

# Ollama runs outside Python and selects NVIDIA/AMD/Vulkan independently. This
# must stay GPU-enabled even when the Python models intentionally use CPU.
OLLAMA_ACCELERATOR = os.environ.get("RAG_OLLAMA_ACCELERATOR", "auto").strip().lower()
_DEFAULT_OLLAMA_NUM_GPU = "0" if OLLAMA_ACCELERATOR == "cpu" else "-1"
OLLAMA_NUM_GPU = int(os.environ.get("RAG_OLLAMA_NUM_GPU", _DEFAULT_OLLAMA_NUM_GPU))
OLLAMA_KEEP_ALIVE = int(os.environ.get("RAG_OLLAMA_KEEP_ALIVE", "300"))

# Set this to a pinned tag or a deployment-specific digest. If the underlying
# model changes, update this value and rebuild the collection.
_EXPLICIT_EMBEDDING_REVISION = os.environ.get("RAG_EMBEDDING_MODEL_REVISION")
EMBEDDING_MODEL_REVISION = _EXPLICIT_EMBEDDING_REVISION or EMBEDDING_MODEL
EMBEDDING_DIMENSION = int(os.environ.get("RAG_EMBEDDING_DIMENSION", "768"))
EMBEDDING_PROMPT_VERSION = "embeddinggemma-retrieval-v1"

DOCUMENT_PREFIX_TEMPLATE = "title: {title} | text: {text}"
QUERY_PREFIX_TEMPLATE = "task: search result | query: {query}"


def stable_fingerprint(values: dict[str, object]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def resolved_embedding_revision() -> str:
    """Return an explicit revision or the digest reported by local Ollama."""

    if _EXPLICIT_EMBEDDING_REVISION:
        return _EXPLICIT_EMBEDDING_REVISION

    endpoint = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=3) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Cannot resolve the Ollama embedding-model digest. Start Ollama or "
            "set RAG_EMBEDDING_MODEL_REVISION explicitly."
        ) from error

    target = EMBEDDING_MODEL if ":" in EMBEDDING_MODEL else f"{EMBEDDING_MODEL}:latest"
    for model in payload.get("models", []):
        name = str(model.get("name") or model.get("model") or "")
        if name == target:
            digest = str(model.get("digest") or "")
            if digest:
                return digest
    raise RuntimeError(
        f"Ollama model {target!r} is not installed or did not report a digest."
    )


def embedding_configuration() -> dict[str, object]:
    return {
        "model": EMBEDDING_MODEL,
        "model_revision": resolved_embedding_revision(),
        "dimension": EMBEDDING_DIMENSION,
        "prompt_version": EMBEDDING_PROMPT_VERSION,
        "document_prefix": DOCUMENT_PREFIX_TEMPLATE,
        "query_prefix": QUERY_PREFIX_TEMPLATE,
    }


def embedding_fingerprint() -> str:
    return stable_fingerprint(embedding_configuration())


def clean_title(title: str | None) -> str:
    value = re.sub(r"\s+", " ", title or "").strip()
    return value[:300] or "none"


class EmbeddingGemmaOllamaEmbeddings(Embeddings):
    """EmbeddingGemma prompts shared by ingestion and retrieval."""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        num_gpu: int = OLLAMA_NUM_GPU,
        keep_alive: int = OLLAMA_KEEP_ALIVE,
    ) -> None:
        self._ollama = OllamaEmbeddings(
            model=model,
            base_url=base_url,
            num_gpu=num_gpu,
            keep_alive=keep_alive,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """LangChain-compatible fallback when document titles are unavailable."""

        return self.embed_documents_with_titles(texts, ["none"] * len(texts))

    def embed_documents_with_titles(
        self,
        texts: Iterable[str],
        titles: Iterable[str | None],
    ) -> list[list[float]]:
        text_list = list(texts)
        title_list = list(titles)
        if len(text_list) != len(title_list):
            raise ValueError("Every document must have exactly one title.")

        prompted = [
            DOCUMENT_PREFIX_TEMPLATE.format(title=clean_title(title), text=text)
            for text, title in zip(text_list, title_list, strict=True)
        ]
        vectors = self._ollama.embed_documents(prompted)
        self._validate_dimensions(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._ollama.embed_query(
            QUERY_PREFIX_TEMPLATE.format(query=text)
        )
        self._validate_dimensions([vector])
        return vector

    @staticmethod
    def _validate_dimensions(vectors: list[list[float]]) -> None:
        unexpected = {
            len(vector)
            for vector in vectors
            if len(vector) != EMBEDDING_DIMENSION
        }
        if unexpected:
            raise RuntimeError(
                "Embedding dimension mismatch: expected "
                f"{EMBEDDING_DIMENSION}, received {sorted(unexpected)}. "
                "Update RAG_EMBEDDING_DIMENSION and rebuild the index."
            )


def create_embeddings() -> EmbeddingGemmaOllamaEmbeddings:
    return EmbeddingGemmaOllamaEmbeddings()
