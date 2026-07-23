"""High-quality Docling ingestion for technical-document RAG.

Key changes from the original version:
- Uses HybridChunker with token-aware splitting and peer merging.
- Embeds Docling's contextualized chunk text, not only chunk.text.
- Preserves short titles/section headers by merging them forward.
- Uses Qwen3-Embedding's asymmetric document/query formatting.
- Repeats table headers and serializes tables as compact Markdown.
- Replaces older versions of a source by stable relative source_id.
- Skips unchanged files and writes debug Markdown/JSONL exports.

Use create_embeddings() in the retrieval application too. The document and
query prefixes must remain consistent between ingestion and search.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from secrets import token_urlsafe
from typing import Any, Callable, Iterable, Sequence

import chromadb
import pypdfium2 as pdfium
import torch
from docling.chunking import HybridChunker
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import settings
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from langchain_core.documents import Document
from transformers import AutoTokenizer

from corpus_scale import debug_artifact_paths, remove_debug_artifacts
from hardware import backend_label
from embedding_cache import EmbeddingCache, embedding_cache_key
from lexical_index import delete_source as delete_lexical_source
from lexical_index import count_chunks as count_lexical_chunks
from lexical_index import fingerprint_ids as fingerprint_lexical_ids
from lexical_index import rebuild_from_collection as rebuild_lexical_index
from lexical_index import replace_source as replace_lexical_source
from lexical_index import set_state as set_lexical_state
from lexical_index import state as lexical_state
from rag_common import (
    COLLECTION_NAME,
    COMPUTE_BACKEND,
    DB_DIR,
    DEBUG_DIR,
    DOCLING_DEVICE,
    EMBEDDING_MODEL,
    LEXICAL_DB_PATH,
    MANIFEST_PATH,
    PDF_DIR,
    create_embeddings,
    document_embedding_input,
    embedding_fingerprint,
    resolved_embedding_revision,
    stable_fingerprint,
)


# ============================================================
# Configuration
# ============================================================

# Qwen's public tokenizer matches the Ollama embedding model and avoids the
# gated-model authentication failure of the former EmbeddingGemma tokenizer.
PRIMARY_TOKENIZER_MODEL = os.environ.get(
    "RAG_CHUNK_TOKENIZER_MODEL",
    "Qwen/Qwen3-Embedding-0.6B",
)
FALLBACK_TOKENIZER_MODEL = os.environ.get(
    "RAG_CHUNK_TOKENIZER_FALLBACK",
    "intfloat/multilingual-e5-small",
)
TOKENIZER_USE_AUTH = os.environ.get("RAG_CHUNK_TOKENIZER_USE_AUTH", "0") == "1"

# Qwen supports a much larger context, but compact chunks remain preferable for
# precise technical-document retrieval and exact citations.
MAX_CHUNK_TOKENS = 400
MIN_CHUNK_TOKENS = 24

# Parent chunks retain Docling's structural context. Smaller overlapping child
# passages are embedded and searched, while the parent is shown to the user.
RETRIEVAL_CHILD_TOKENS = int(
    os.environ.get("RAG_RETRIEVAL_CHILD_TOKENS", "240")
)
RETRIEVAL_CHILD_OVERLAP_TOKENS = int(
    os.environ.get("RAG_RETRIEVAL_CHILD_OVERLAP_TOKENS", "40")
)

def _configured_positive_int(name: str, default: int = 0) -> int:
    """Read an optional positive override; zero and ``auto`` mean automatic."""

    raw_value = str(os.environ.get(name, default)).strip().lower()
    if raw_value in {"", "0", "auto"}:
        return 0
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer or 'auto'.")
    return value


# Zero means that the value is selected from live machine headroom in main().
# Explicit environment values remain available for reproducible benchmarking.
PDF_PAGE_WINDOW = _configured_positive_int("RAG_PDF_PAGE_WINDOW")
DOCLING_PAGE_BATCH_SIZE = _configured_positive_int("RAG_DOCLING_PAGE_BATCH_SIZE")
DOCLING_QUEUE_MAX_SIZE = _configured_positive_int("RAG_DOCLING_QUEUE_MAX_SIZE")
DOCLING_MODEL_BATCH_SIZE = _configured_positive_int("RAG_DOCLING_MODEL_BATCH_SIZE")
DOCLING_NUM_THREADS = _configured_positive_int("RAG_DOCLING_NUM_THREADS")


@dataclass(frozen=True)
class MachineResources:
    """A point-in-time resource snapshot used only for performance tuning."""

    cpu_count: int
    system_available_mb: int | None
    system_total_mb: int | None
    accelerator_available_mb: int | None
    accelerator_total_mb: int | None


@dataclass(frozen=True)
class IngestionTuning:
    """Quality-neutral Docling settings derived from current free resources."""

    page_window: int
    page_batch_size: int
    model_batch_size: int
    queue_max_size: int
    num_threads: int
    resources: MachineResources


def system_memory_info_mb() -> tuple[int, int] | None:
    """Return currently available and total physical RAM using the stdlib."""

    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            succeeded = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            )
        except (AttributeError, OSError):
            succeeded = 0
        if succeeded:
            divisor = 1024 * 1024
            return (
                int(status.ullAvailPhys // divisor),
                int(status.ullTotalPhys // divisor),
            )

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    divisor = 1024 * 1024
    return (
        int(available_pages * page_size // divisor),
        int(total_pages * page_size // divisor),
    )


def accelerator_memory_info_mb() -> tuple[int, int] | None:
    """Return live free and total accelerator memory, when it is measurable."""

    if not DOCLING_DEVICE.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, OSError):
        return None
    divisor = 1024 * 1024
    return int(free_bytes // divisor), int(total_bytes // divisor)


def machine_resources() -> MachineResources:
    system_memory = system_memory_info_mb()
    accelerator_memory = accelerator_memory_info_mb()
    return MachineResources(
        cpu_count=max(1, int(os.cpu_count() or 1)),
        system_available_mb=system_memory[0] if system_memory else None,
        system_total_mb=system_memory[1] if system_memory else None,
        accelerator_available_mb=(
            accelerator_memory[0] if accelerator_memory else None
        ),
        accelerator_total_mb=accelerator_memory[1] if accelerator_memory else None,
    )


def recommended_docling_threads(
    cpu_count: int | None = None,
    system_available_mb: int | None = None,
) -> int:
    """Use CPU parallelism only while live RAM can support native workers."""

    cpu_limit = max(1, int(cpu_count or os.cpu_count() or 1) // 2)
    memory_limit = (
        max(1, (int(system_available_mb) - 1024) // 3072)
        if system_available_mb is not None
        else 2
    )
    return max(1, min(4, cpu_limit, memory_limit))


def recommended_docling_batch_size(
    accelerator_available_mb: int | None,
    system_available_mb: int | None = None,
) -> int:
    """Choose model batches from live free memory rather than GPU capacity."""

    limits: list[int] = []
    if accelerator_available_mb is not None:
        limits.append(max(1, (int(accelerator_available_mb) - 1024) // 2048))
    if system_available_mb is not None:
        limits.append(max(1, (int(system_available_mb) - 2048) // 3072))
    return max(1, min(4, *(limits or [1])))


def recommended_page_window(
    accelerator_available_mb: int | None,
    system_available_mb: int | None,
) -> int:
    """Size conversion windows continuously from current memory headroom."""

    limits: list[int] = []
    if accelerator_available_mb is not None:
        limits.append(max(2, (int(accelerator_available_mb) - 512) // 768))
    if system_available_mb is not None:
        limits.append(max(2, (int(system_available_mb) - 1024) // 1280))
    return max(2, min(16, *(limits or [4])))


def resolve_ingestion_tuning(
    resources: MachineResources | None = None,
) -> IngestionTuning:
    """Resolve automatic tuning once, before Docling allocates its models."""

    resources = resources or machine_resources()
    automatic_window = recommended_page_window(
        resources.accelerator_available_mb,
        resources.system_available_mb,
    )
    automatic_batch = recommended_docling_batch_size(
        resources.accelerator_available_mb,
        resources.system_available_mb,
    )
    page_window = PDF_PAGE_WINDOW or automatic_window
    page_batch_size = DOCLING_PAGE_BATCH_SIZE or automatic_batch
    model_batch_size = DOCLING_MODEL_BATCH_SIZE or automatic_batch
    num_threads = DOCLING_NUM_THREADS or recommended_docling_threads(
        resources.cpu_count,
        resources.system_available_mb,
    )
    automatic_queue = max(
        4,
        min(
            16,
            page_window * 2,
            max(4, int(resources.system_available_mb or 3072) // 768),
        ),
    )
    return IngestionTuning(
        page_window=page_window,
        page_batch_size=page_batch_size,
        model_batch_size=model_batch_size,
        queue_max_size=DOCLING_QUEUE_MAX_SIZE or automatic_queue,
        num_threads=num_threads,
        resources=resources,
    )
# The CUDA-prefixed settings remain accepted for compatibility with existing
# installations. ROCm exposes memory through the same torch.cuda API.
CUDA_HEADROOM_WARNING_MB = int(
    os.environ.get(
        "RAG_GPU_HEADROOM_WARNING_MB",
        os.environ.get("RAG_CUDA_HEADROOM_WARNING_MB", "3500"),
    )
)
ALLOW_LOW_CUDA_HEADROOM = os.environ.get(
    "RAG_ALLOW_LOW_GPU_HEADROOM",
    os.environ.get("RAG_ALLOW_LOW_CUDA_HEADROOM", "0"),
) == "1"
GPU_HEADROOM_GUARD_MARKER = "RAG_GPU_HEADROOM_GUARD_FAILED"

ENABLE_OCR = False
AUTO_OCR = True
MIN_EXTRACTED_CHARS_PER_PAGE = 120
ENABLE_TABLE_STRUCTURE = True
TABLE_MODE = TableFormerMode.ACCURATE

BATCH_SIZE = 64
EMBEDDING_CACHE_ENABLED = os.environ.get("RAG_EMBEDDING_CACHE", "1") != "0"
EMBEDDING_CACHE_PATH = Path(
    os.environ.get("RAG_EMBEDDING_CACHE_PATH", str(DB_DIR / "embedding_cache.sqlite3"))
).resolve()
SKIP_UNCHANGED_FILES = True
EXPORT_DEBUG_FILES = True
INDEX_COMMIT_WAIT_SECONDS = 120

# Increment whenever chunking, prompts, or metadata change materially.
INGESTION_VERSION = "docling-hybrid-structural-v9-verified-pages-qwen"
MANIFEST_SCHEMA_VERSION = 2


# ============================================================
# General utilities
# ============================================================

def clean_text(text: str) -> str:
    """Light cleanup that preserves paragraphs, lists, and Markdown tables."""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def calculate_file_hash(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while block := file.read(block_size):
            sha256.update(block)

    return sha256.hexdigest()


def make_chunk_id(
    source_id: str,
    file_hash: str,
    chunk_index: int,
    text: str,
    ingestion_signature: str,
) -> str:
    """Create an ID that changes when source, logic, or content changes."""

    identity = "|".join(
        [
            source_id,
            file_hash,
            ingestion_signature,
            str(chunk_index),
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def scalar_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)

    return output


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def release_gpu_memory() -> None:
    """Release dead tensors and cached blocks before a smaller GPU retry."""

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # The original conversion error is more useful than a cleanup failure,
        # especially after an asynchronously reported GPU exception.
        pass


def release_cuda_memory() -> None:
    """Backward-compatible alias for extensions using the former name."""

    release_gpu_memory()


def is_gpu_out_of_memory(message: str | None) -> bool:
    normalized = (message or "").lower()
    return (
        "out of memory" in normalized
        or "hsa_status_error_out_of_resources" in normalized
    )


def is_memory_allocation_failure(message: str | None) -> bool:
    normalized = (message or "").lower()
    return is_gpu_out_of_memory(normalized) or any(
        marker in normalized
        for marker in (
            "std::bad_alloc",
            "cannot allocate memory",
            "memoryerror",
            "out of resources",
        )
    )


def is_cuda_out_of_memory(message: str | None) -> bool:
    """Backward-compatible alias that also detects ROCm/HIP exhaustion."""

    return is_gpu_out_of_memory(message)


def report_cuda_headroom(resources: MachineResources | None = None) -> None:
    """Warn when other resident models leave little accelerator memory."""

    if not DOCLING_DEVICE.startswith("cuda") or not torch.cuda.is_available():
        return
    memory = (
        (
            resources.accelerator_available_mb,
            resources.accelerator_total_mb,
        )
        if resources is not None
        else accelerator_memory_info_mb()
    )
    if memory is None or memory[0] is None or memory[1] is None:
        print("GPU memory status unavailable.")
        return
    free_mb, total_mb = memory
    accelerator = backend_label(COMPUTE_BACKEND)
    print(
        f"{accelerator} memory before Docling: "
        f"{free_mb} MiB free / {total_mb} MiB total"
    )
    if free_mb < CUDA_HEADROOM_WARNING_MB:
        # The supervising application uses this stable marker to distinguish a
        # startup headroom race from a document conversion failure. In auto
        # mode it can then unload search and retry the same queue exclusively.
        print(
            f"{GPU_HEADROOM_GUARD_MARKER} "
            f"free_mib={free_mb} required_mib={CUDA_HEADROOM_WARNING_MB}",
            flush=True,
        )
        message = (
            f"Only {free_mb} MiB of GPU memory is free; at least "
            f"{CUDA_HEADROOM_WARNING_MB} MiB is required by the ingestion "
            "guard. Stop the search application and run "
            f"`ollama stop {EMBEDDING_MODEL}` before retrying."
        )
        if not ALLOW_LOW_CUDA_HEADROOM:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


def batched(
    values: Sequence[Any],
    batch_size: int,
) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def ids_fingerprint(ids: Sequence[str]) -> str:
    return stable_fingerprint({"ids": sorted(ids)})


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"version": MANIFEST_SCHEMA_VERSION, "sources": {}, "migration_history": []}

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read ingestion manifest {MANIFEST_PATH}: {error}"
        ) from error

    if not isinstance(manifest.get("sources"), dict):
        raise RuntimeError(f"Invalid ingestion manifest: {MANIFEST_PATH}")
    previous_version = int(manifest.get("version") or 1)
    if previous_version < MANIFEST_SCHEMA_VERSION:
        backup = MANIFEST_PATH.with_name(
            f"{MANIFEST_PATH.name}.pre-v{MANIFEST_SCHEMA_VERSION}-migration.bak"
        )
        if not backup.exists():
            shutil.copy2(MANIFEST_PATH, backup)
        manifest["version"] = MANIFEST_SCHEMA_VERSION
        manifest["migration_history"] = [
            *list(manifest.get("migration_history") or []),
            {
                "from": previous_version,
                "to": MANIFEST_SCHEMA_VERSION,
                "applied_at": datetime.now(tz=UTC).isoformat(),
            },
        ][-20:]
        save_manifest(manifest)
    return manifest


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(MANIFEST_PATH)


def ingestion_fingerprint(tokenizer_name: str, ocr_policy: str) -> str:
    return stable_fingerprint(
        {
            "ingestion_version": INGESTION_VERSION,
            "embedding_fingerprint": embedding_fingerprint(),
            "chunk_tokenizer": tokenizer_name,
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
            "retrieval_child_tokens": RETRIEVAL_CHILD_TOKENS,
            "retrieval_child_overlap_tokens": RETRIEVAL_CHILD_OVERLAP_TOKENS,
            "ocr_policy": ocr_policy,
            "table_structure": ENABLE_TABLE_STRUCTURE,
            "table_mode": str(TABLE_MODE),
            "docling": package_version("docling"),
            "docling_core": package_version("docling-core"),
        }
    )


# ============================================================
# Docling configuration
# ============================================================

class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Serialize table chunks as compact Markdown with visible headers."""

    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


@dataclass(frozen=True)
class ChunkingRuntime:
    chunker: Any
    tokenizer: HuggingFaceTokenizer | None
    tokenizer_name: str

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return int(self.tokenizer.count_tokens(text=text))

        # Conservative approximation used only if no tokenizer can be loaded.
        words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return max(1, int(len(words) * 1.35))

    def contextualize(self, chunk: Any) -> str:
        contextualize = getattr(self.chunker, "contextualize", None)
        if callable(contextualize):
            return clean_text(contextualize(chunk=chunk))
        return clean_text(getattr(chunk, "text", "") or "")


def create_converter(
    enable_ocr: bool = ENABLE_OCR,
    pdf_backend: type[Any] | None = None,
    tuning: IngestionTuning | None = None,
) -> DocumentConverter:
    tuning = tuning or resolve_ingestion_tuning()
    settings.perf.page_batch_size = tuning.page_batch_size
    settings.perf.page_batch_concurrency = 1
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.do_table_structure = ENABLE_TABLE_STRUCTURE
    pipeline_options.queue_max_size = tuning.queue_max_size
    pipeline_options.ocr_batch_size = tuning.model_batch_size
    pipeline_options.layout_batch_size = tuning.model_batch_size
    pipeline_options.table_batch_size = tuning.model_batch_size
    pipeline_options.accelerator_options.num_threads = tuning.num_threads
    pipeline_options.accelerator_options.device = DOCLING_DEVICE

    if ENABLE_TABLE_STRUCTURE:
        pipeline_options.table_structure_options.mode = TABLE_MODE
        pipeline_options.table_structure_options.do_cell_matching = True

    format_option_kwargs: dict[str, Any] = {
        "pipeline_options": pipeline_options,
    }
    if pdf_backend is not None:
        format_option_kwargs["backend"] = pdf_backend

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(**format_option_kwargs)
        }
    )


def _ascii_pdf_name(pdf_path: Path) -> str:
    normalized = unicodedata.normalize("NFKD", pdf_path.stem)
    ascii_stem = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_stem)
    ascii_stem = re.sub(r"_+", "_", ascii_stem).strip(" ._")
    ascii_stem = ascii_stem[:80].rstrip(" ._") or "document"
    name_hash = hashlib.sha256(pdf_path.name.encode("utf-8")).hexdigest()[:10]
    return f"{ascii_stem}-{name_hash}.pdf"


@contextmanager
def docling_safe_pdf_path(pdf_path: Path) -> Iterable[Path]:
    """Yield a short ASCII path for parsers that mishandle Windows Unicode paths."""

    resolved_path = pdf_path.resolve()
    if str(resolved_path).isascii() and len(str(resolved_path)) < 240:
        yield resolved_path
        return

    preferred_root = Path(tempfile.gettempdir())
    if not str(preferred_root.resolve()).isascii():
        preferred_root = Path(__file__).resolve().parent
    if not str(preferred_root.resolve()).isascii():
        raise RuntimeError(
            "Docling needs an ASCII temporary directory to open this PDF path."
        )

    with tempfile.TemporaryDirectory(
        prefix="reference-desk-docling-",
        dir=preferred_root,
    ) as temporary_directory:
        safe_path = Path(temporary_directory) / _ascii_pdf_name(resolved_path)
        try:
            os.link(resolved_path, safe_path)
        except OSError:
            shutil.copy2(resolved_path, safe_path)
        print(f"  Using temporary Docling-safe filename: {safe_path.name}")
        yield safe_path


def _load_huggingface_tokenizer(model_name: str) -> HuggingFaceTokenizer:
    errors: list[str] = []
    token: bool = True if TOKENIZER_USE_AUTH else False
    tokenizer = None
    for local_only in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                local_files_only=local_only,
                token=token,
            )
            break
        except Exception as error:
            mode = "local cache" if local_only else "public download"
            errors.append(f"{mode}: {type(error).__name__}: {error}")
    if tokenizer is None:
        raise RuntimeError("; ".join(errors))
    return HuggingFaceTokenizer(
        tokenizer=tokenizer,
        max_tokens=MAX_CHUNK_TOKENS,
    )


def create_chunking_runtime() -> ChunkingRuntime:
    """Create HybridChunker, with graceful tokenizer fallbacks."""

    tokenizer: HuggingFaceTokenizer | None = None
    tokenizer_name = "approximate"

    for candidate in (PRIMARY_TOKENIZER_MODEL, FALLBACK_TOKENIZER_MODEL):
        try:
            tokenizer = _load_huggingface_tokenizer(candidate)
            tokenizer_name = candidate
            break
        except Exception as error:
            print(
                f"Tokenizer unavailable ({candidate}): "
                f"{type(error).__name__}: {error}"
            )

    if tokenizer is None:
        raise RuntimeError(
            "No chunk tokenizer could be loaded. Refusing to fall back to "
            "unbounded hierarchical chunks because they can exceed both the "
            "embedding and reranker limits. Cache one of the configured "
            "tokenizers and run ingestion again."
        )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=MarkdownTableSerializerProvider(),
    )

    return ChunkingRuntime(
        chunker=chunker,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
    )


# ============================================================
# Provenance extraction
# ============================================================

def bbox_to_dict(bbox: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "left": float(getattr(bbox, "l", getattr(bbox, "left", 0.0))),
        "top": float(getattr(bbox, "t", getattr(bbox, "top", 0.0))),
        "right": float(getattr(bbox, "r", getattr(bbox, "right", 0.0))),
        "bottom": float(getattr(bbox, "b", getattr(bbox, "bottom", 0.0))),
    }
    origin = getattr(bbox, "coord_origin", None)
    if origin is not None:
        result["coord_origin"] = str(getattr(origin, "value", origin))
    return result


def extract_chunk_locations(chunk: Any) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    metadata = getattr(chunk, "meta", None)

    if metadata is None:
        return locations

    for item in getattr(metadata, "doc_items", None) or []:
        label = getattr(item, "label", None)

        for provenance in getattr(item, "prov", None) or []:
            location: dict[str, Any] = {
                "page": getattr(provenance, "page_no", None),
                "label": str(label) if label else "",
            }

            bbox = getattr(provenance, "bbox", None)
            if bbox is not None:
                location["bbox"] = bbox_to_dict(bbox)

            locations.append(location)

    # Remove exact duplicate provenance records.
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for location in locations:
        key = json.dumps(location, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduplicated.append(location)

    return deduplicated


def extract_headings(chunk: Any) -> list[str]:
    metadata = getattr(chunk, "meta", None)
    if metadata is None:
        return []

    return unique_preserving_order(
        clean_text(str(heading))
        for heading in (getattr(metadata, "headings", None) or [])
        if clean_text(str(heading))
    )


def extract_labels(chunk: Any) -> list[str]:
    metadata = getattr(chunk, "meta", None)
    if metadata is None:
        return []

    labels = [
        str(label)
        for item in (getattr(metadata, "doc_items", None) or [])
        if (label := getattr(item, "label", None)) is not None
    ]
    return sorted(set(labels))


def get_page_range(
    locations: list[dict[str, Any]],
) -> tuple[int, int]:
    pages = sorted(
        {
            int(location["page"])
            for location in locations
            if location.get("page") is not None
        }
    )

    if not pages:
        return -1, -1

    return pages[0], pages[-1]


# ============================================================
# Chunk preparation
# ============================================================

PAGE_NUMBER_ONLY = re.compile(
    r"^(?:page\s*)?\d+(?:\s*(?:/|of|de)\s*\d+)?$",
    flags=re.IGNORECASE,
)
HEADING_LABELS = {"title", "section_header"}


@dataclass
class ChunkRecord:
    original_indices: list[int]
    raw_text: str
    content: str
    headings: list[str]
    labels: list[str]
    locations: list[dict[str, Any]]
    token_count: int
    ocr_applied: bool = False


def is_noise_chunk(text: str) -> bool:
    compact = clean_text(text)
    if not compact:
        return True
    if PAGE_NUMBER_ONLY.fullmatch(compact):
        return True
    return not any(character.isalnum() for character in compact)


def merge_text(left: str, right: str) -> str:
    left = clean_text(left)
    right = clean_text(right)

    if not left:
        return right
    if not right:
        return left

    # Contextualized chunks often repeat a heading already present in a short
    # title chunk. Keep only the richer copy in that case.
    if right.startswith(left + "\n") or right == left:
        return right
    if left.startswith(right + "\n"):
        return left

    return f"{left}\n\n{right}"


def merge_records(
    left: ChunkRecord,
    right: ChunkRecord,
    runtime: ChunkingRuntime,
) -> ChunkRecord:
    content = merge_text(left.content, right.content)

    return ChunkRecord(
        original_indices=left.original_indices + right.original_indices,
        raw_text=merge_text(left.raw_text, right.raw_text),
        content=content,
        headings=unique_preserving_order(left.headings + right.headings),
        labels=sorted(set(left.labels + right.labels)),
        locations=extract_unique_locations(left.locations + right.locations),
        token_count=runtime.count_tokens(content),
        ocr_applied=left.ocr_applied or right.ocr_applied,
    )


def extract_unique_locations(
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for location in locations:
        key = json.dumps(location, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            output.append(location)

    return output


def is_heading_like(record: ChunkRecord) -> bool:
    return bool(HEADING_LABELS.intersection(record.labels))


def build_chunk_records(
    chunks: Iterable[Any],
    runtime: ChunkingRuntime,
    ocr_applied: bool = False,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []

    for chunk_index, chunk in enumerate(chunks):
        raw_text = clean_text(getattr(chunk, "text", "") or "")
        content = runtime.contextualize(chunk) or raw_text

        if is_noise_chunk(content):
            continue

        records.append(
            ChunkRecord(
                original_indices=[chunk_index],
                raw_text=raw_text,
                content=content,
                headings=extract_headings(chunk),
                labels=extract_labels(chunk),
                locations=extract_chunk_locations(chunk),
                token_count=runtime.count_tokens(content),
                ocr_applied=ocr_applied,
            )
        )

    return records


def merge_short_chunks(
    records: list[ChunkRecord],
    runtime: ChunkingRuntime,
) -> list[ChunkRecord]:
    """Attach tiny headings/openings to meaningful neighboring content.

    This replaces the old `len(text) < 40: continue` behavior, which could
    silently remove the document title or first lines.
    """

    merged: list[ChunkRecord] = []
    pending_prefix: list[ChunkRecord] = []

    for record in records:
        is_short = record.token_count < MIN_CHUNK_TOKENS

        # A leading fragment or a short heading belongs to what follows.
        if is_short and (not merged or is_heading_like(record)):
            pending_prefix.append(record)
            continue

        if pending_prefix:
            # Prefixes are prepended to ``record``. Walking backwards preserves
            # their original title -> section -> body order.
            for prefix in reversed(pending_prefix):
                candidate = merge_records(prefix, record, runtime)
                if candidate.token_count <= MAX_CHUNK_TOKENS:
                    record = candidate
                else:
                    merged.append(prefix)
            pending_prefix.clear()

        # A small non-heading fragment usually belongs to the previous chunk.
        if is_short and merged:
            candidate = merge_records(merged[-1], record, runtime)
            if candidate.token_count <= MAX_CHUNK_TOKENS:
                merged[-1] = candidate
                continue

        merged.append(record)

    # A trailing section heading cannot be merged forward, so merge backward
    # when safe; otherwise preserve it rather than deleting it.
    for pending in pending_prefix:
        if merged:
            candidate = merge_records(merged[-1], pending, runtime)
            if candidate.token_count <= MAX_CHUNK_TOKENS:
                merged[-1] = candidate
                continue
        merged.append(pending)

    return merged


def stitch_window_headings(
    records: list[ChunkRecord],
    previous_headings: Sequence[str] | None,
    runtime: ChunkingRuntime,
) -> tuple[list[ChunkRecord], list[str]]:
    """Carry section context across independent Docling page windows."""

    active = list(previous_headings or [])
    for record in records:
        if record.headings:
            active = list(record.headings)
            continue

        # Some converters label a heading but omit it from meta.headings.
        if is_heading_like(record) and record.raw_text:
            heading = clean_text(record.raw_text).splitlines()[0][:300]
            if heading:
                active = [heading]
                record.headings = list(active)
            continue

        if not active:
            continue

        record.headings = list(active)
        heading_path = " > ".join(active)
        # Dense embeddings only see page_content and document title. Add the
        # inherited section explicitly when Docling could not contextualize it.
        if active[-1].casefold() not in record.content[:600].casefold():
            record.content = merge_text(
                f"Section context: {heading_path}",
                record.content,
            )
            record.token_count = runtime.count_tokens(record.content)
    return records, active


def split_text_for_retrieval(
    text: str,
    runtime: ChunkingRuntime,
    max_tokens: int = RETRIEVAL_CHILD_TOKENS,
    overlap_tokens: int = RETRIEVAL_CHILD_OVERLAP_TOKENS,
) -> list[str]:
    """Split a structural parent into overlapping token-bounded children."""

    text = clean_text(text)
    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError("Retrieval child token limit must be positive.")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Retrieval child overlap must be below the token limit.")
    if runtime.count_tokens(text) <= max_tokens:
        return [text]

    pieces = re.findall(r"\S+\s*", text, flags=re.UNICODE)
    children: list[str] = []
    start = 0
    while start < len(pieces):
        end = start
        chosen = ""
        while end < len(pieces):
            candidate = clean_text("".join(pieces[start : end + 1]))
            if runtime.count_tokens(candidate) > max_tokens and end > start:
                break
            chosen = candidate
            end += 1
            if runtime.count_tokens(chosen) >= max_tokens:
                break

        if not chosen:
            chosen = clean_text(pieces[start])
            end = start + 1
        children.append(chosen)
        if end >= len(pieces):
            break

        overlap_start = end
        while overlap_start > start:
            overlap = clean_text("".join(pieces[overlap_start - 1 : end]))
            if runtime.count_tokens(overlap) > overlap_tokens:
                break
            overlap_start -= 1
        next_start = overlap_start if overlap_start > start else end
        start = max(start + 1, next_start)

    return children


STRUCTURAL_REQUIREMENT_PATTERN = re.compile(
    r"\b(?:shall|must|required|requirement|deve|deverá|obrigatório)\b",
    flags=re.IGNORECASE,
)
STRUCTURAL_LIST_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-zA-Z][.)])\s+", re.MULTILINE)
STRUCTURAL_EQUATION_PATTERN = re.compile(
    r"(?:\$[^$]+\$|\\(?:frac|sum|int|sqrt)\b|[=≈≤≥∑∫√])"
)


def infer_retrieval_unit_type(record: ChunkRecord) -> str:
    """Classify a parent into a retrieval unit without an LLM."""

    labels = " ".join(record.labels).casefold()
    text = record.content
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_rows = sum(line.count("|") >= 2 for line in lines)
    if "table" in labels or pipe_rows >= 2:
        return "table"
    if "list" in labels or len(STRUCTURAL_LIST_PATTERN.findall(text)) >= 2:
        return "list"
    if "formula" in labels or "equation" in labels or STRUCTURAL_EQUATION_PATTERN.search(text):
        return "equation"
    if STRUCTURAL_REQUIREMENT_PATTERN.search(text):
        return "requirement"
    if re.search(r"(?m)^\s*[^:\n]{2,80}:\s+\S", text):
        return "definition"
    return "paragraph"


def split_structural_record(
    record: ChunkRecord,
    runtime: ChunkingRuntime,
) -> list[tuple[str, str, str]]:
    """Return meaning-preserving child text, type, and optional table header."""

    unit_type = infer_retrieval_unit_type(record)
    if unit_type != "table":
        return [
            (child, unit_type, "")
            for child in split_text_for_retrieval(record.content, runtime)
        ]

    lines = [line.rstrip() for line in record.content.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.count("|") >= 2]
    if len(table_lines) < 2:
        return [
            (child, unit_type, "")
            for child in split_text_for_retrieval(record.content, runtime)
        ]

    header_lines = [table_lines[0]]
    if len(table_lines) > 1 and re.fullmatch(r"[\s|:+-]+", table_lines[1]):
        header_lines.append(table_lines[1])
    data_rows = table_lines[len(header_lines) :]
    table_header = clean_text(table_lines[0])[:1000]
    prefix = [line for line in lines if line not in table_lines]
    groups: list[str] = []
    current = [*prefix, *header_lines]
    for row in data_rows:
        candidate = "\n".join([*current, row])
        if (
            len(current) > len(prefix) + len(header_lines)
            and runtime.count_tokens(candidate) > RETRIEVAL_CHILD_TOKENS
        ):
            groups.append(clean_text("\n".join(current)))
            current = [*header_lines, row]
        else:
            current.append(row)
    if current:
        groups.append(clean_text("\n".join(current)))
    return [(group, unit_type, table_header) for group in groups if group]


def infer_document_title(
    records: list[ChunkRecord],
    pdf_path: Path,
) -> str:
    """Resolve a bibliographic label without promoting later chapter headings.

    Docling's ``headings`` describe hierarchy and are inherited by following
    chunks. They are not document metadata. Restricting candidates to physical
    page one prevents a heading from page two or three (for example, a praise,
    contents, or chapter heading) from becoming the title of the whole book.
    """

    def location_page(location: dict[str, Any]) -> int | None:
        try:
            return int(location.get("page"))
        except (TypeError, ValueError):
            return None

    def record_pages(record: ChunkRecord) -> set[int]:
        return {
            page
            for location in record.locations
            if (page := location_page(location)) is not None
        }

    def is_on_first_page(record: ChunkRecord) -> bool:
        for location in record.locations:
            if location_page(location) == 1:
                return True
        return False

    def has_page_one_title_label(record: ChunkRecord) -> bool:
        return any(
            location_page(location) == 1
            and str(location.get("label") or "").casefold().rsplit(".", 1)[-1]
            == "title"
            for location in record.locations
        )

    def candidate_line(value: str) -> str:
        for raw_line in clean_text(value).splitlines():
            line = re.sub(r"^[#>*_`\s]+|[#>*_`\s]+$", "", raw_line).strip()
            normalized = line.casefold()
            if not line or PAGE_NUMBER_ONLY.fullmatch(line):
                continue
            if normalized in {"contents", "table of contents"}:
                continue
            if normalized.startswith(
                ("isbn", "copyright", "©", "edited by ", "http://", "https://", "www.")
            ):
                continue
            if len(line) > 300 or sum(character.isalpha() for character in line) < 2:
                continue
            return line[:300]
        return ""

    page_one_records = [record for record in records if is_on_first_page(record)]

    # Prefer an explicit title item detected on the first physical page.
    for record in page_one_records:
        labels = {label.casefold().rsplit(".", 1)[-1] for label in record.labels}
        if has_page_one_title_label(record) or (
            "title" in labels and record_pages(record) == {1}
        ):
            title = candidate_line(record.raw_text)
            if title:
                return title

    # A page-one heading is still useful, but a later-page heading never is.
    for record in page_one_records:
        if record_pages(record) != {1}:
            continue
        for heading in record.headings:
            title = candidate_line(heading)
            if title:
                return title

    # Some cover pages are classified as plain text or code. Use their first
    # meaningful visible line rather than continuing into later pages.
    for record in page_one_records:
        title = candidate_line(record.raw_text)
        if title:
            return title

    # Image-only covers may not expose text when OCR is unavailable. The file
    # name is safer than guessing from a chapter heading on another page.
    return clean_text(pdf_path.stem.replace("_", " "))[:300] or "Untitled document"


def records_to_langchain(
    records: list[ChunkRecord],
    runtime: ChunkingRuntime,
    pdf_path: Path,
    source_id: str,
    file_hash: str,
    page_count: int,
    tokenizer_name: str,
    current_ingestion_fingerprint: str,
) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    document_title = infer_document_title(records, pdf_path)
    model_revision = resolved_embedding_revision()
    current_embedding_fingerprint = embedding_fingerprint()

    child_entries: list[
        tuple[int, ChunkRecord, str, int, int, str, str, str]
    ] = []
    for parent_index, record in enumerate(records):
        parent_chunk_id = make_chunk_id(
            source_id=source_id,
            file_hash=file_hash,
            chunk_index=parent_index,
            text=record.content,
            ingestion_signature=current_ingestion_fingerprint + "|parent",
        )
        children = split_structural_record(record, runtime)
        for child_index, (child_text, unit_type, table_header) in enumerate(
            children,
            start=1,
        ):
            child_entries.append(
                (
                    parent_index,
                    record,
                    parent_chunk_id,
                    child_index,
                    len(children),
                    child_text,
                    unit_type,
                    table_header,
                )
            )

    for chunk_index, entry in enumerate(child_entries):
        (
            parent_index,
            record,
            parent_chunk_id,
            child_index,
            child_count,
            child_text,
            unit_type,
            table_header,
        ) = entry
        page_start, page_end = get_page_range(record.locations)
        section_path = " > ".join(record.headings)
        chunk_id = make_chunk_id(
            source_id=source_id,
            file_hash=file_hash,
            chunk_index=chunk_index,
            text=child_text,
            ingestion_signature=current_ingestion_fingerprint,
        )

        metadata = {
            "source": pdf_path.name,
            "source_id": source_id,
            "file_path": str(pdf_path.resolve()),
            "file_hash": file_hash,
            "document_title": document_title,
            "page_count": page_count,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "parent_chunk_id": parent_chunk_id,
            "parent_chunk_index": parent_index,
            "parent_content": record.content,
            "parent_character_count": len(record.content),
            "parent_token_count": record.token_count,
            "child_index": child_index,
            "child_count": child_count,
            "docling_chunk_indices_json": record.original_indices,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "section_path": section_path,
            "section_title": record.headings[-1] if record.headings else "",
            "section_ancestors_json": record.headings,
            "hierarchy_depth": len(record.headings),
            "content_labels": ", ".join(record.labels),
            "retrieval_unit_type": unit_type,
            "table_header": table_header,
            "character_count": len(child_text),
            "word_count": len(child_text.split()),
            "token_count": runtime.count_tokens(child_text),
            "content_hash": hashlib.sha256(
                child_text.encode("utf-8")
            ).hexdigest(),
            "parent_content_hash": hashlib.sha256(
                record.content.encode("utf-8")
            ).hexdigest(),
            "parser": "docling",
            "ocr_applied": record.ocr_applied,
            "chunker": "hybrid-structural-parent-child",
            "tokenizer": tokenizer_name,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_revision": model_revision,
            "embedding_fingerprint": current_embedding_fingerprint,
            "ingestion_version": INGESTION_VERSION,
            "ingestion_fingerprint": current_ingestion_fingerprint,
            "source_parent_count": len(records),
            "source_chunk_count": len(child_entries),
            "docling_version": package_version("docling"),
            "docling_core_version": package_version("docling-core"),
            "locations_json": record.locations,
        }

        documents.append(
            Document(
                page_content=child_text,
                metadata={
                    key: scalar_metadata_value(value)
                    for key, value in metadata.items()
                },
            )
        )
        ids.append(chunk_id)

    return documents, ids


# ============================================================
# Existing-record management
# ============================================================

def get_existing_source_records(
    collection: Any,
    source_id: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    result = collection.get(
        where={"source_id": source_id},
        include=["metadatas"],
    )

    ids = list(result.get("ids") or [])
    metadatas = list(result.get("metadatas") or [])
    return ids, metadatas


def source_is_current(
    ids: list[str],
    metadatas: list[dict[str, Any]],
    file_hash: str,
    current_ingestion_fingerprint: str,
    manifest_entry: dict[str, Any] | None,
) -> bool:
    if not manifest_entry or not ids or len(ids) != len(metadatas):
        return False

    manifest_matches = (
        manifest_entry.get("complete") is True
        and manifest_entry.get("file_hash") == file_hash
        and manifest_entry.get("ingestion_fingerprint")
        == current_ingestion_fingerprint
        and manifest_entry.get("chunk_count") == len(ids)
        and manifest_entry.get("ids_fingerprint") == ids_fingerprint(ids)
    )
    return manifest_matches and all(
        metadata.get("file_hash") == file_hash
        and metadata.get("ingestion_fingerprint")
        == current_ingestion_fingerprint
        for metadata in metadatas
    )


@dataclass
class EmbeddingCacheStats:
    hits: int = 0
    misses: int = 0


def _embed_document_positions(
    documents: list[Document],
    embeddings: Any,
    positions: Sequence[int],
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for position_batch in batched(positions, BATCH_SIZE):
        batch = [documents[position] for position in position_batch]
        vectors.extend(
            embeddings.embed_documents_with_titles(
                [document.page_content for document in batch],
                [document.metadata.get("document_title") for document in batch],
            )
        )
    return vectors


def embed_documents_in_batches(
    documents: list[Document],
    embeddings: Any,
    *,
    cache_path: Path | None = EMBEDDING_CACHE_PATH,
    stats: EmbeddingCacheStats | None = None,
) -> list[list[float]]:
    """Embed documents while reusing vectors for the exact model prompt."""

    cache_stats = stats if stats is not None else EmbeddingCacheStats()
    if not documents:
        return []
    if not EMBEDDING_CACHE_ENABLED or cache_path is None:
        cache_stats.misses += len(documents)
        return _embed_document_positions(
            documents,
            embeddings,
            list(range(len(documents))),
        )

    fingerprint = embedding_fingerprint()
    prompted = [
        document_embedding_input(
            document.page_content,
            document.metadata.get("document_title"),
        )
        for document in documents
    ]
    keys = [embedding_cache_key(fingerprint, value) for value in prompted]
    vectors: list[list[float] | None] = [None] * len(documents)
    try:
        with EmbeddingCache(cache_path, fingerprint) as cache:
            cache.prune_other_fingerprints()
            cached: dict[str, list[float]] = {}
            for key_batch in batched(keys, BATCH_SIZE):
                cached.update(cache.get_many(list(key_batch)))
            missing_positions: list[int] = []
            for position, cache_key in enumerate(keys):
                vector = cached.get(cache_key)
                if vector is None:
                    missing_positions.append(position)
                else:
                    vectors[position] = vector

            cache_stats.hits += len(documents) - len(missing_positions)
            cache_stats.misses += len(missing_positions)
            for position_batch in batched(missing_positions, BATCH_SIZE):
                positions = list(position_batch)
                fresh = _embed_document_positions(documents, embeddings, positions)
                if len(fresh) != len(positions):
                    raise ValueError("Embedding service returned an unexpected vector count.")
                cache.put_many(
                    (keys[position], vector)
                    for position, vector in zip(positions, fresh, strict=True)
                )
                for position, vector in zip(positions, fresh, strict=True):
                    vectors[position] = vector
    except (OSError, sqlite3.DatabaseError) as error:
        print(
            "  WARNING: embedding cache unavailable; embedding this document "
            f"normally ({type(error).__name__}: {error})."
        )
        cache_stats.hits = 0
        cache_stats.misses = len(documents)
        return _embed_document_positions(
            documents,
            embeddings,
            list(range(len(documents))),
        )

    if any(vector is None for vector in vectors):
        raise RuntimeError("Embedding cache left one or more passages unresolved.")
    return [vector for vector in vectors if vector is not None]


def upsert_documents_in_batches(
    collection: Any,
    documents: list[Document],
    ids: list[str],
    vectors: list[list[float]],
    existing_ids: Sequence[str],
) -> None:
    if not (len(documents) == len(ids) == len(vectors)):
        raise ValueError("Documents, IDs, and embeddings must have equal lengths.")

    preexisting = set(existing_ids)
    newly_written: list[str] = []
    try:
        for document_batch, id_batch, vector_batch in zip(
            batched(documents, BATCH_SIZE),
            batched(ids, BATCH_SIZE),
            batched(vectors, BATCH_SIZE),
            strict=True,
        ):
            batch_ids = list(id_batch)
            collection.upsert(
                ids=batch_ids,
                embeddings=list(vector_batch),
                documents=[document.page_content for document in document_batch],
                metadatas=[document.metadata for document in document_batch],
            )
            newly_written.extend(
                chunk_id for chunk_id in batch_ids if chunk_id not in preexisting
            )
    except Exception:
        if newly_written:
            collection.delete(ids=newly_written)
        raise


# ============================================================
# Diagnostics
# ============================================================


@contextmanager
def measure_stage(timings: dict[str, float], stage: str) -> Iterable[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - started)


def show_ingestion_timings(
    timings: dict[str, float],
    *,
    page_count: int,
    cache_stats: EmbeddingCacheStats,
) -> None:
    total = timings.get("total", 0.0)
    ordered = ("conversion", "chunking", "postprocess", "embedding", "index")
    details = " | ".join(
        f"{name} {timings.get(name, 0.0):.1f}s" for name in ordered
    )
    rate = page_count / total if total > 0 else 0.0
    print(f"  Timing: {details} | total {total:.1f}s ({rate:.2f} pages/s)")
    print(
        "  Embedding cache: "
        f"{cache_stats.hits} reused, {cache_stats.misses} computed"
    )

def export_debug_files(
    source_id: str,
    markdown: str,
    documents: list[Document],
) -> None:
    if not EXPORT_DEBUG_FILES:
        return

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path, chunks_path = debug_artifact_paths(DEBUG_DIR, source_id)
    markdown_path.write_text(markdown, encoding="utf-8")

    with chunks_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            row = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def opening_coverage(markdown: str, documents: list[Document]) -> float:
    opening_words = re.findall(r"\w+", markdown[:1500].lower())[:60]
    indexed_words = set(
        re.findall(
            r"\w+",
            " ".join(doc.page_content for doc in documents[:3]).lower(),
        )
    )

    meaningful = {word for word in opening_words if len(word) > 2}
    if not meaningful:
        return 1.0

    return len(meaningful.intersection(indexed_words)) / len(meaningful)


def show_document_summary(
    documents: list[Document],
    markdown: str,
) -> None:
    pages: set[int] = set()
    for document in documents:
        try:
            page_start = int(document.metadata.get("page_start", -1))
            page_end = int(document.metadata.get("page_end", page_start))
        except (TypeError, ValueError):
            continue
        if page_start > 0 and page_end >= page_start:
            pages.update(range(page_start, page_end + 1))
    sections = {
        document.metadata.get("section_path")
        for document in documents
        if document.metadata.get("section_path")
    }

    parent_ids = {
        str(document.metadata.get("parent_chunk_id"))
        for document in documents
        if document.metadata.get("parent_chunk_id")
    }
    print(f"  Stored retrieval children: {len(documents)}")
    print(f"  Structural parents: {len(parent_ids)}")
    print(f"  Referenced pages: {len(pages)}")
    if documents:
        try:
            page_count = int(documents[0].metadata.get("page_count", 0))
        except (TypeError, ValueError):
            page_count = 0
        missing_pages = sorted(set(range(1, page_count + 1)) - pages)
        if missing_pages:
            preview = ", ".join(str(page) for page in missing_pages[:20])
            suffix = " ..." if len(missing_pages) > 20 else ""
            print(f"  WARNING: pages without indexed provenance: {preview}{suffix}")
    print(f"  Detected section paths: {len(sections)}")
    print(f"  Opening coverage in first 3 chunks: {opening_coverage(markdown, documents):.0%}")

    print("  Parsed document opening:")
    print("   ", clean_text(markdown)[:240].replace("\n", " "))

    for position, document in enumerate(documents[:3], start=1):
        print(f"  Indexed chunk {position}:")
        print(f"    Page: {document.metadata.get('page')}")
        print(
            "    Section:",
            document.metadata.get("section_path") or "(none)",
        )
        print(f"    Tokens: {document.metadata.get('token_count')}")
        print(
            "    Text:",
            document.page_content[:240].replace("\n", " "),
        )


# ============================================================
# PDF processing
# ============================================================

@dataclass(frozen=True)
class ProcessResult:
    status: str
    chunks: int = 0


def get_pdf_page_count(pdf_path: Path) -> int:
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(document)
    finally:
        document.close()


def conversion_problem(
    result: Any,
    page_start: int,
    page_end: int,
) -> str | None:
    expected_pages = set(range(page_start, page_end + 1))
    actual_pages = {int(page) for page in result.document.pages}
    missing_pages = sorted(expected_pages - actual_pages)
    messages = [
        str(getattr(error, "error_message", error))
        for error in (getattr(result, "errors", None) or [])
    ]

    details: list[str] = []
    if missing_pages:
        details.append(f"missing pages {missing_pages}")
    if messages:
        details.append("; ".join(messages[:5]))
    return " | ".join(details) or None


@dataclass
class AdaptivePageWindow:
    """Lower future window sizes after this document shows memory pressure."""

    current: int
    minimum: int = 1

    def observe_memory_pressure(self, attempted_pages: int) -> None:
        if attempted_pages < self.current:
            return
        self.current = max(self.minimum, min(self.current, attempted_pages // 2))


def convert_page_range_resilient(
    converter: DocumentConverter,
    pdf_path: Path,
    page_start: int,
    page_end: int,
    *,
    fallback_converter: DocumentConverter | None = None,
    source_name: str | None = None,
    on_memory_pressure: Callable[[int], None] | None = None,
) -> list[Any]:
    """Convert a range, recursively splitting it after native/page failures."""

    try:
        result = converter.convert(
            pdf_path,
            page_range=(page_start, page_end),
            raises_on_error=False,
        )
        problem = conversion_problem(result, page_start, page_end)
        if problem is None:
            return [result]
    except Exception as error:
        result = None
        problem = f"{type(error).__name__}: {error}"

    gpu_oom = is_gpu_out_of_memory(problem)
    allocation_failure = is_memory_allocation_failure(problem)

    # Allocation failures usually describe the size of the current native
    # preprocessing run, not a malformed PDF. Retry the same high-quality
    # parser on smaller ranges before changing parser backends.
    if allocation_failure and page_start < page_end:
        attempted_pages = page_end - page_start + 1
        if on_memory_pressure is not None:
            on_memory_pressure(attempted_pages)
        midpoint = (page_start + page_end) // 2
        print(
            f"  Memory pressure on pages {page_start}-{page_end}; "
            "retrying the primary parser on smaller ranges."
        )
        del result
        release_cuda_memory()
        return convert_page_range_resilient(
            converter,
            pdf_path,
            page_start,
            midpoint,
            fallback_converter=fallback_converter,
            source_name=source_name,
            on_memory_pressure=on_memory_pressure,
        ) + convert_page_range_resilient(
            converter,
            pdf_path,
            midpoint + 1,
            page_end,
            fallback_converter=fallback_converter,
            source_name=source_name,
            on_memory_pressure=on_memory_pressure,
        )

    if not gpu_oom and fallback_converter is not None:
        print("  Default PDF parser failed; retrying with PDFium backend.")
        try:
            fallback_result = fallback_converter.convert(
                pdf_path,
                page_range=(page_start, page_end),
                raises_on_error=False,
            )
            fallback_problem = conversion_problem(
                fallback_result,
                page_start,
                page_end,
            )
            if fallback_problem is None:
                return [fallback_result]
        except Exception as error:
            fallback_result = None
            fallback_problem = f"{type(error).__name__}: {error}"
        converter = fallback_converter
        fallback_converter = None
        result = fallback_result
        problem = f"PDFium fallback: {fallback_problem}"
        gpu_oom = is_gpu_out_of_memory(problem)

    if gpu_oom:
        print("  GPU memory exhaustion detected; clearing cache before retry.")
        release_cuda_memory()

    if page_start >= page_end:
        recovery = (
            " Stop the search application, run "
            f"`ollama stop {EMBEDDING_MODEL}`, and retry ingestion."
            if gpu_oom
            else ""
        )
        raise RuntimeError(
            f"Docling could not convert page {page_start} of "
            f"{source_name or pdf_path.name}: "
            f"{problem}.{recovery}"
        )

    midpoint = (page_start + page_end) // 2
    print(
        f"  Retrying pages {page_start}-{page_end} as smaller ranges "
        f"after: {problem}"
    )
    del result
    release_cuda_memory()
    return convert_page_range_resilient(
        converter,
        pdf_path,
        page_start,
        midpoint,
        fallback_converter=fallback_converter,
        source_name=source_name,
        on_memory_pressure=on_memory_pressure,
    ) + convert_page_range_resilient(
        converter,
        pdf_path,
        midpoint + 1,
        page_end,
        fallback_converter=fallback_converter,
        source_name=source_name,
        on_memory_pressure=on_memory_pressure,
    )


def should_retry_with_ocr(markdown: str, page_count: int) -> bool:
    if page_count <= 0:
        return not clean_text(markdown)
    extracted_characters = len(re.sub(r"\s+", "", markdown))
    return extracted_characters / page_count < MIN_EXTRACTED_CHARS_PER_PAGE


def record_page_numbers(records: Sequence[ChunkRecord]) -> set[int]:
    """Collect every physical PDF page represented by chunk provenance."""

    pages: set[int] = set()
    for record in records:
        for location in record.locations:
            try:
                page = int(location.get("page") or 0)
            except (TypeError, ValueError):
                continue
            if page > 0:
                pages.add(page)
    return pages


def extractable_text_pages(
    pdf_path: Path,
    page_numbers: Iterable[int],
    *,
    minimum_characters: int = MIN_CHUNK_TOKENS,
) -> set[int]:
    """Return pages whose source PDF contains enough text to require a chunk."""

    required_pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not required_pages:
        return set()
    document = pdfium.PdfDocument(str(pdf_path))
    extractable: set[int] = set()
    try:
        for page_number in required_pages:
            if page_number > len(document):
                continue
            page = document[page_number - 1]
            text_page = None
            try:
                text_page = page.get_textpage()
                text = text_page.get_text_range() or ""
            finally:
                if text_page is not None:
                    text_page.close()
                page.close()
            if len(re.sub(r"\s+", "", text)) >= minimum_characters:
                extractable.add(page_number)
    finally:
        document.close()
    return extractable


def missing_extractable_pages(
    pdf_path: Path,
    records: Sequence[ChunkRecord],
    page_count: int,
) -> list[int]:
    """Find source pages with text but no indexed passage provenance."""

    represented_pages = record_page_numbers(records)
    unrepresented_pages = set(range(1, page_count + 1)) - represented_pages
    return sorted(extractable_text_pages(pdf_path, unrepresented_pages))


def page_coverage_warnings(
    missing_pages: Sequence[int],
    *,
    allow_incomplete_index: bool,
) -> list[str]:
    """Enforce page coverage unless this source has an explicit user override."""

    if not missing_pages:
        return []
    preview = ", ".join(str(page) for page in missing_pages[:20])
    suffix = " ..." if len(missing_pages) > 20 else ""
    if not allow_incomplete_index:
        raise RuntimeError(
            "Refusing to mark the document indexed because extractable text "
            f"is still missing from pages {preview}{suffix}."
        )
    return [
        "Indexed by explicit user override even though extractable text "
        f"is missing from pages {preview}{suffix}."
    ]


def emit_ingestion_progress(
    source_id: str,
    message: str,
    *,
    phase: str,
    page_start: int | None = None,
    page_end: int | None = None,
    page_count: int | None = None,
) -> None:
    """Send structured UI progress and a flushed human-readable log line."""

    emit_corpus_event(
        "progress",
        source_id=source_id,
        phase=phase,
        message=message,
        page_start=page_start,
        page_end=page_end,
        page_count=page_count,
    )
    print(f"  {message}", flush=True)


def process_pdf(
    pdf_path: Path,
    converter: DocumentConverter,
    runtime: ChunkingRuntime,
    collection: Any,
    embeddings: Any,
    manifest: dict[str, Any],
    force: bool = False,
    ocr_enabled: bool = False,
    auto_ocr: bool = AUTO_OCR,
    ocr_converter: DocumentConverter | None = None,
    fallback_converter: DocumentConverter | None = None,
    ocr_fallback_converter: DocumentConverter | None = None,
    docling_pdf_path: Path | None = None,
    commit_gate: Path | None = None,
    tuning: IngestionTuning | None = None,
    allow_incomplete_index: bool = False,
) -> ProcessResult:
    process_started = time.perf_counter()
    timings: dict[str, float] = {}
    cache_stats = EmbeddingCacheStats()
    source_id = pdf_path.resolve().relative_to(PDF_DIR.resolve()).as_posix()
    file_hash = calculate_file_hash(pdf_path)
    ocr_policy = "force" if ocr_enabled else "auto" if auto_ocr else "off"
    current_fingerprint = ingestion_fingerprint(
        runtime.tokenizer_name,
        ocr_policy,
    )
    if allow_incomplete_index:
        current_fingerprint = stable_fingerprint(
            {
                "base_ingestion_fingerprint": current_fingerprint,
                "allow_incomplete_index": True,
            }
        )

    print(f"\nProcessing: {source_id}")

    existing_ids, existing_metadatas = get_existing_source_records(
        collection=collection,
        source_id=source_id,
    )

    if (
        not force
        and SKIP_UNCHANGED_FILES
        and source_is_current(
            ids=existing_ids,
            metadatas=existing_metadatas,
            file_hash=file_hash,
            current_ingestion_fingerprint=current_fingerprint,
            manifest_entry=manifest["sources"].get(source_id),
        )
    ):
        print(f"  Unchanged; keeping {len(existing_ids)} existing chunks.")
        return ProcessResult(status="unchanged")

    tuning = tuning or resolve_ingestion_tuning()
    conversion_path = docling_pdf_path or pdf_path
    page_count = get_pdf_page_count(conversion_path)
    if page_count <= 0:
        print("  PDF contains no pages.")
        return ProcessResult(status="no_output")

    records: list[ChunkRecord] = []
    markdown_parts: list[str] = []
    carried_headings: list[str] = []
    page_window = AdaptivePageWindow(tuning.page_window)

    def lower_future_windows(attempted_pages: int) -> None:
        previous_window = page_window.current
        page_window.observe_memory_pressure(attempted_pages)
        if page_window.current < previous_window:
            emit_ingestion_progress(
                source_id,
                (
                    "Memory pressure detected; future conversion windows were "
                    f"reduced from {previous_window} to {page_window.current} pages."
                ),
                phase="tuning",
                page_count=page_count,
            )

    def consume_results(
        base_results: list[Any],
        *,
        inherit_previous_headings: bool,
    ) -> None:
        nonlocal carried_headings
        while base_results:
            base_result = base_results.pop(0)
            base_document = base_result.document
            base_pages = sorted(int(page) for page in base_document.pages)
            with measure_stage(timings, "chunking"):
                base_markdown = base_document.export_to_markdown()
            final_results: list[tuple[Any, bool]] = [(base_result, ocr_enabled)]

            if (
                auto_ocr
                and not ocr_enabled
                and should_retry_with_ocr(base_markdown, len(base_pages))
            ):
                if ocr_converter is None:
                    raise RuntimeError(
                        "Automatic OCR was requested without an OCR converter."
                    )
                retry_start, retry_end = base_pages[0], base_pages[-1]
                emit_ingestion_progress(
                    source_id,
                    (
                        f"Sparse text on pages {retry_start}-{retry_end}; "
                        "retrying that range with OCR."
                    ),
                    phase="ocr",
                    page_start=retry_start,
                    page_end=retry_end,
                    page_count=page_count,
                )
                with measure_stage(timings, "conversion"):
                    ocr_results = convert_page_range_resilient(
                        ocr_converter,
                        conversion_path,
                        retry_start,
                        retry_end,
                        fallback_converter=ocr_fallback_converter,
                        source_name=pdf_path.name,
                        on_memory_pressure=lower_future_windows,
                    )
                final_results = [(result, True) for result in ocr_results]
                base_result = None
                base_document = None
                gc.collect()

            for final_result, ocr_applied in final_results:
                with measure_stage(timings, "chunking"):
                    docling_document = final_result.document
                    batch_markdown = docling_document.export_to_markdown()
                    markdown_parts.append(batch_markdown)
                    raw_chunks = runtime.chunker.chunk(dl_doc=docling_document)
                    window_records = build_chunk_records(
                        raw_chunks,
                        runtime,
                        ocr_applied=ocr_applied,
                    )
                    inherited = carried_headings if inherit_previous_headings else []
                    window_records, active_headings = stitch_window_headings(
                        window_records,
                        inherited,
                        runtime,
                    )
                    if inherit_previous_headings:
                        carried_headings = active_headings
                    records.extend(window_records)
                del raw_chunks, docling_document, final_result
                gc.collect()
            final_results.clear()
            base_result = None
            base_document = None
            gc.collect()

    page_start = 1
    while page_start <= page_count:
        page_end = min(page_start + page_window.current - 1, page_count)
        emit_ingestion_progress(
            source_id,
            f"Converting pages {page_start}-{page_end} of {page_count}",
            phase="conversion",
            page_start=page_start,
            page_end=page_end,
            page_count=page_count,
        )
        with measure_stage(timings, "conversion"):
            base_results = convert_page_range_resilient(
                converter,
                conversion_path,
                page_start,
                page_end,
                fallback_converter=fallback_converter,
                source_name=pdf_path.name,
                on_memory_pressure=lower_future_windows,
            )
        consume_results(base_results, inherit_previous_headings=True)
        release_cuda_memory()
        page_start = page_end + 1

    pages_to_recover = missing_extractable_pages(
        conversion_path,
        records,
        page_count,
    )
    for page_number in pages_to_recover:
        emit_ingestion_progress(
            source_id,
            f"Verifying and retrying omitted text page {page_number} of {page_count}",
            phase="page_recovery",
            page_start=page_number,
            page_end=page_number,
            page_count=page_count,
        )
        with measure_stage(timings, "conversion"):
            recovery_results = convert_page_range_resilient(
                converter,
                conversion_path,
                page_number,
                page_number,
                fallback_converter=fallback_converter,
                source_name=pdf_path.name,
                on_memory_pressure=lower_future_windows,
            )
        consume_results(recovery_results, inherit_previous_headings=False)
        release_cuda_memory()

    still_missing = missing_extractable_pages(
        conversion_path,
        records,
        page_count,
    )
    index_warnings = page_coverage_warnings(
        still_missing,
        allow_incomplete_index=allow_incomplete_index,
    )
    if index_warnings:
        emit_ingestion_progress(
            source_id,
            index_warnings[0],
            phase="quality_override",
            page_count=page_count,
        )

    records.sort(
        key=lambda record: (
            min(record_page_numbers([record]) or {page_count + 1}),
            record.original_indices[0] if record.original_indices else 0,
        )
    )

    with measure_stage(timings, "postprocess"):
        markdown = "\n\n".join(markdown_parts)
        records = merge_short_chunks(records, runtime)

        documents, ids = records_to_langchain(
            records=records,
            runtime=runtime,
            pdf_path=pdf_path,
            source_id=source_id,
            file_hash=file_hash,
            page_count=page_count,
            tokenizer_name=runtime.tokenizer_name,
            current_ingestion_fingerprint=current_fingerprint,
        )
        if index_warnings:
            for document in documents:
                document.metadata["incomplete_page_coverage"] = True
                document.metadata["index_warning"] = index_warnings[0]

    if not documents:
        print("  No usable chunks were produced.")
        return ProcessResult(status="no_output")

    # Complete every model call before touching the active index. In concurrent
    # mode, the app pauses search only for this short commit window so dense and
    # lexical retrieval never observe different source revisions.
    with measure_stage(timings, "embedding"):
        vectors = embed_documents_in_batches(
            documents,
            embeddings,
            stats=cache_stats,
        )
    with measure_stage(timings, "index"):
        with index_commit_window(commit_gate, source_id):
            upsert_documents_in_batches(
                collection=collection,
                documents=documents,
                ids=ids,
                vectors=vectors,
                existing_ids=existing_ids,
            )

            new_id_set = set(ids)
            old_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in new_id_set]
            if old_ids:
                collection.delete(ids=old_ids)

            replace_lexical_source(
                path=LEXICAL_DB_PATH,
                source_id=source_id,
                documents=documents,
                ids=ids,
            )

            manifest["sources"][source_id] = {
                "complete": True,
                "file_hash": file_hash,
                "ingestion_fingerprint": current_fingerprint,
                "embedding_fingerprint": embedding_fingerprint(),
                "chunk_count": len(ids),
                "ids_fingerprint": ids_fingerprint(ids),
                "allow_incomplete_index": bool(index_warnings),
                "index_warnings": index_warnings,
            }
            save_manifest(manifest)

    try:
        export_debug_files(source_id, markdown, documents)
    except OSError as error:
        print(f"  WARNING: debug export failed: {error}")
    show_document_summary(documents, markdown)
    timings["total"] = time.perf_counter() - process_started
    show_ingestion_timings(
        timings,
        page_count=page_count,
        cache_stats=cache_stats,
    )

    return ProcessResult(status="stored", chunks=len(documents))


# ============================================================
# Chroma setup and main
# ============================================================

def create_collection() -> Any:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    expected_metadata = {
        "hnsw:space": "cosine",
        "embedding_fingerprint": embedding_fingerprint(),
        "schema_version": 4,
    }
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=expected_metadata,
    )

    actual_fingerprint = (collection.metadata or {}).get(
        "embedding_fingerprint"
    )
    if collection.count() and actual_fingerprint != embedding_fingerprint():
        raise RuntimeError(
            "The existing collection was built with a different or unknown "
            "embedding configuration. Use a new RAG_COLLECTION name or rebuild "
            f"the index. Expected fingerprint: {embedding_fingerprint()}"
        )
    # Chroma treats hnsw:space as immutable and rejects Collection.modify even
    # when the repeated value is unchanged. Existing collections are upgraded
    # through per-source ingestion fingerprints instead of metadata mutation.
    if not collection.count() and collection.metadata != expected_metadata:
        collection.modify(metadata=expected_metadata)
    return collection


def prune_removed_sources(
    collection: Any,
    manifest: dict[str, Any],
    active_source_ids: set[str],
) -> int:
    result = collection.get(include=["metadatas"])
    stored_source_ids = {
        str(metadata.get("source_id"))
        for metadata in (result.get("metadatas") or [])
        if metadata and metadata.get("source_id")
    }
    removed = sorted(stored_source_ids - active_source_ids)
    for source_id in removed:
        collection.delete(where={"source_id": source_id})
        delete_lexical_source(LEXICAL_DB_PATH, source_id)
        manifest["sources"].pop(source_id, None)
        remove_debug_artifacts(DEBUG_DIR, source_id)
        print(f"Pruned removed source: {source_id}")
    if removed:
        save_manifest(manifest)
    return len(removed)


def synchronize_lexical_index(collection: Any) -> None:
    collection_ids = list(collection.get(include=[]).get("ids") or [])
    current_state = lexical_state(LEXICAL_DB_PATH)
    current_ids_fingerprint = fingerprint_lexical_ids(collection_ids)
    if (
        count_lexical_chunks(LEXICAL_DB_PATH) != collection.count()
        or current_state.get("ids_fingerprint") != current_ids_fingerprint
    ):
        rebuild_lexical_index(
            path=LEXICAL_DB_PATH,
            collection=collection,
            fingerprint=embedding_fingerprint(),
        )
        return
    set_lexical_state(
        LEXICAL_DB_PATH,
        {
            "embedding_fingerprint": embedding_fingerprint(),
            "chunk_count": str(collection.count()),
            "ids_fingerprint": current_ids_fingerprint,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest local PDFs into the technical-document index."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess every PDF even if its completed manifest is current.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete indexed sources whose PDFs are no longer present.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Force OCR for every PDF.",
    )
    parser.add_argument(
        "--no-auto-ocr",
        action="store_true",
        help="Do not retry low-text PDFs with OCR automatically.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Process only this relative source ID. May be repeated.",
    )
    parser.add_argument(
        "--allow-incomplete-source",
        action="append",
        default=[],
        help=(
            "Permit one selected source to be indexed with an explicit warning "
            "when page-coverage validation still fails. May be repeated."
        ),
    )
    parser.add_argument(
        "--queue-managed",
        action="store_true",
        help="Treat --source values as the complete managed queue selection.",
    )
    parser.add_argument(
        "--queue-control",
        type=Path,
        help="Persistent queue state checked between documents for pause requests.",
    )
    parser.add_argument(
        "--commit-gate",
        type=Path,
        help=(
            "Optional app-controlled gate that pauses search only while an "
            "index revision is committed."
        ),
    )
    return parser.parse_args()


def emit_corpus_event(event: str, **values: Any) -> None:
    payload = {"event": event, **values}
    # Keep the stdout control protocol ASCII-only.  On Windows the child can
    # otherwise encode a curly quote or accented letter in a console code page
    # while the parent correctly reads UTF-8, turning a real source name into
    # U+FFFD and preventing quarantine from locating the PDF.
    print("CORPUS_EVENT " + json.dumps(payload, ensure_ascii=True), flush=True)


def _commit_gate_is_open(path: Path, token: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("token") == token


@contextmanager
def index_commit_window(gate_path: Path | None, source_id: str) -> Iterable[None]:
    """Wait for the app to pause searches around one atomic source update."""

    token = ""
    if gate_path is not None:
        token = token_urlsafe(16)
        emit_corpus_event("commit_requested", source_id=source_id, token=token)
        deadline = time.monotonic() + INDEX_COMMIT_WAIT_SECONDS
        while not _commit_gate_is_open(gate_path, token):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for the application to pause searches "
                    "before committing the index."
                )
            time.sleep(0.05)
    try:
        yield
    except Exception:
        if token:
            emit_corpus_event(
                "commit_finished",
                source_id=source_id,
                token=token,
                status="failed",
            )
        raise
    else:
        if token:
            emit_corpus_event(
                "commit_finished",
                source_id=source_id,
                token=token,
                status="complete",
            )


def queue_pause_requested(control_path: Path | None) -> bool:
    if control_path is None or not control_path.is_file():
        return False
    try:
        value = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(value.get("paused")) if isinstance(value, dict) else False


def main() -> None:
    args = parse_args()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    all_pdf_paths = sorted(PDF_DIR.rglob("*.pdf"))
    selected_sources = {
        str(value or "").strip().replace("\\", "/") for value in args.source
    }
    allowed_incomplete_sources = {
        str(value or "").strip().replace("\\", "/")
        for value in getattr(args, "allow_incomplete_source", [])
    }
    pdf_paths = (
        [
            path
            for path in all_pdf_paths
            if path.resolve().relative_to(PDF_DIR.resolve()).as_posix()
            in selected_sources
        ]
        if args.queue_managed
        else all_pdf_paths
    )

    if not pdf_paths and not args.prune:
        print(f"No PDF files found in: {PDF_DIR.resolve()}")
        return

    print("Docling technical-document ingestion")
    print(f"PDF directory: {PDF_DIR.resolve()}")
    print(f"PDF files: {len(pdf_paths)}")
    if args.queue_managed:
        print(f"Managed queue selection: {len(selected_sources)} source(s)")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding revision: {resolved_embedding_revision()}")
    print(f"Accelerator backend: {backend_label(COMPUTE_BACKEND)}")
    print(f"Docling device: {DOCLING_DEVICE}")
    print("Ollama embeddings: automatic GPU offload requested")
    print(f"Chunk token target: {MAX_CHUNK_TOKENS}")
    print(
        "Retrieval child target / overlap: "
        f"{RETRIEVAL_CHILD_TOKENS} / {RETRIEVAL_CHILD_OVERLAP_TOKENS} tokens"
    )
    ocr_enabled = bool(args.ocr or ENABLE_OCR)
    auto_ocr = bool(AUTO_OCR and not args.no_auto_ocr and not ocr_enabled)
    print(f"OCR enabled: {ocr_enabled}")
    print(f"Automatic OCR fallback: {auto_ocr}")
    print(f"Table structure enabled: {ENABLE_TABLE_STRUCTURE}")
    tuning = resolve_ingestion_tuning()
    resources = tuning.resources
    system_memory_label = (
        f"{resources.system_available_mb} MiB free / "
        f"{resources.system_total_mb} MiB total"
        if resources.system_available_mb is not None
        and resources.system_total_mb is not None
        else "unavailable"
    )
    accelerator_memory_label = (
        f"{resources.accelerator_available_mb} MiB free / "
        f"{resources.accelerator_total_mb} MiB total"
        if resources.accelerator_available_mb is not None
        and resources.accelerator_total_mb is not None
        else "not available"
    )
    print(
        "Live resources: "
        f"{resources.cpu_count} logical CPUs, RAM {system_memory_label}, "
        f"accelerator {accelerator_memory_label}"
    )
    print(
        "Adaptive conversion: "
        f"{tuning.page_window}-page windows, "
        f"page/model batches {tuning.page_batch_size}/{tuning.model_batch_size}, "
        f"queue {tuning.queue_max_size}, "
        f"{tuning.num_threads} preprocessing threads"
    )
    print(
        "Embedding cache: "
        + (str(EMBEDDING_CACHE_PATH) if EMBEDDING_CACHE_ENABLED else "disabled")
    )
    report_cuda_headroom(resources)

    collection = create_collection()
    manifest = load_manifest()

    if pdf_paths and queue_pause_requested(args.queue_control):
        emit_corpus_event("paused")
        raise SystemExit(75)

    if not pdf_paths:
        active_source_ids = {
            path.resolve().relative_to(PDF_DIR.resolve()).as_posix()
            for path in sorted(PDF_DIR.rglob("*.pdf"))
        }
        with index_commit_window(args.commit_gate, "__corpus_prune__"):
            pruned_documents = prune_removed_sources(
                collection=collection,
                manifest=manifest,
                active_source_ids=active_source_ids,
            )
            synchronize_lexical_index(collection)
        print(f"Pruned documents: {pruned_documents}")
        return

    converter = create_converter(enable_ocr=ocr_enabled, tuning=tuning)
    fallback_converter = create_converter(
        enable_ocr=ocr_enabled,
        pdf_backend=PyPdfiumDocumentBackend,
        tuning=tuning,
    )
    ocr_converter = (
        create_converter(enable_ocr=True, tuning=tuning) if auto_ocr else None
    )
    ocr_fallback_converter = (
        create_converter(
            enable_ocr=True,
            pdf_backend=PyPdfiumDocumentBackend,
            tuning=tuning,
        )
        if auto_ocr
        else None
    )
    runtime = create_chunking_runtime()
    embeddings = create_embeddings()

    print(f"Chunk tokenizer: {runtime.tokenizer_name}")

    successful_documents = 0
    failed_documents = 0
    skipped_documents = 0
    no_output_documents = 0
    total_chunks = 0
    paused = False

    for pdf_path in pdf_paths:
        if queue_pause_requested(args.queue_control):
            paused = True
            emit_corpus_event("paused")
            break
        source_id = pdf_path.resolve().relative_to(PDF_DIR.resolve()).as_posix()
        emit_corpus_event("started", source_id=source_id)
        try:
            with docling_safe_pdf_path(pdf_path) as conversion_path:
                result = process_pdf(
                    pdf_path=pdf_path,
                    converter=converter,
                    runtime=runtime,
                    collection=collection,
                    embeddings=embeddings,
                    manifest=manifest,
                    force=args.force,
                    ocr_enabled=ocr_enabled,
                    auto_ocr=auto_ocr,
                    ocr_converter=ocr_converter,
                    fallback_converter=fallback_converter,
                    ocr_fallback_converter=ocr_fallback_converter,
                    docling_pdf_path=conversion_path,
                    commit_gate=args.commit_gate,
                    tuning=tuning,
                    allow_incomplete_index=(
                        source_id in allowed_incomplete_sources
                    ),
                )

            if result.status == "unchanged":
                skipped_documents += 1
            elif result.status == "no_output":
                no_output_documents += 1
                if args.queue_managed:
                    failed_documents += 1
                    emit_corpus_event(
                        "failed",
                        source_id=source_id,
                        error_type="NoUsableOutput",
                        error="Docling produced no indexable passages.",
                    )
                    continue
            else:
                successful_documents += 1
                total_chunks += result.chunks
            emit_corpus_event(
                "completed",
                source_id=source_id,
                status=result.status,
                chunks=result.chunks,
            )

        except Exception as error:
            failed_documents += 1
            print(f"\nFAILED: {pdf_path.name}")
            print(f"  {type(error).__name__}: {error}")
            emit_corpus_event(
                "failed",
                source_id=source_id,
                error_type=type(error).__name__,
                error=str(error),
            )
            # A malformed or unsupported PDF belongs to this queue item, not
            # the whole ingestion run. The parent quarantines it while this
            # worker proceeds to the next selected source.
            continue

    pruned_documents = 0
    with index_commit_window(args.commit_gate, "__corpus_prune__"):
        if args.prune and not paused:
            active_source_ids = {
                path.resolve().relative_to(PDF_DIR.resolve()).as_posix()
                for path in sorted(PDF_DIR.rglob("*.pdf"))
            }
            pruned_documents = prune_removed_sources(
                collection=collection,
                manifest=manifest,
                active_source_ids=active_source_ids,
            )

        synchronize_lexical_index(collection)

    print("\n" + "=" * 70)
    print("Ingestion complete")
    print(f"Ingested/updated documents: {successful_documents}")
    print(f"Unchanged documents: {skipped_documents}")
    print(f"Documents with no usable output: {no_output_documents}")
    print(f"Failed documents: {failed_documents}")
    print(f"Pruned documents: {pruned_documents}")
    print(f"New chunks stored: {total_chunks}")
    print(f"Database directory: {DB_DIR.resolve()}")
    if EXPORT_DEBUG_FILES:
        print(f"Debug directory: {DEBUG_DIR.resolve()}")
    if paused:
        raise SystemExit(75)
    if failed_documents:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
