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
import gc
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from secrets import token_urlsafe
from typing import Any, Iterable, Sequence

import chromadb
import pypdfium2 as pdfium
import torch
from docling.chunking import HybridChunker
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

from hardware import backend_label
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

# Large PDFs are converted in independent page windows so Docling never needs
# to retain hundreds of rendered pages in the native preprocessing pipeline.
PDF_PAGE_WINDOW = int(os.environ.get("RAG_PDF_PAGE_WINDOW", "6"))
DOCLING_PAGE_BATCH_SIZE = int(
    os.environ.get("RAG_DOCLING_PAGE_BATCH_SIZE", "2")
)
DOCLING_QUEUE_MAX_SIZE = int(
    os.environ.get("RAG_DOCLING_QUEUE_MAX_SIZE", "16")
)
DOCLING_MODEL_BATCH_SIZE = int(
    os.environ.get("RAG_DOCLING_MODEL_BATCH_SIZE", "2")
)
DOCLING_NUM_THREADS = int(os.environ.get("RAG_DOCLING_NUM_THREADS", "2"))
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

ENABLE_OCR = False
AUTO_OCR = True
MIN_EXTRACTED_CHARS_PER_PAGE = 120
ENABLE_TABLE_STRUCTURE = True
TABLE_MODE = TableFormerMode.ACCURATE

BATCH_SIZE = 64
SKIP_UNCHANGED_FILES = True
EXPORT_DEBUG_FILES = True
INDEX_COMMIT_WAIT_SECONDS = 120

# Increment whenever chunking, prompts, or metadata change materially.
INGESTION_VERSION = "docling-hybrid-parent-child-v6-qwen"


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


def is_cuda_out_of_memory(message: str | None) -> bool:
    """Backward-compatible alias that also detects ROCm/HIP exhaustion."""

    return is_gpu_out_of_memory(message)


def report_cuda_headroom() -> None:
    """Warn when other resident models leave little accelerator memory."""

    if not DOCLING_DEVICE.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as error:
        print(f"GPU memory status unavailable: {error}")
        return
    free_mb = free_bytes // (1024 * 1024)
    total_mb = total_bytes // (1024 * 1024)
    accelerator = backend_label(COMPUTE_BACKEND)
    print(
        f"{accelerator} memory before Docling: "
        f"{free_mb} MiB free / {total_mb} MiB total"
    )
    if free_mb < CUDA_HEADROOM_WARNING_MB:
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
        return {"version": 1, "sources": {}}

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read ingestion manifest {MANIFEST_PATH}: {error}"
        ) from error

    if not isinstance(manifest.get("sources"), dict):
        raise RuntimeError(f"Invalid ingestion manifest: {MANIFEST_PATH}")
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
            "pdf_page_window": PDF_PAGE_WINDOW,
            "docling_page_batch_size": DOCLING_PAGE_BATCH_SIZE,
            "docling_queue_max_size": DOCLING_QUEUE_MAX_SIZE,
            "docling_model_batch_size": DOCLING_MODEL_BATCH_SIZE,
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


def create_converter(enable_ocr: bool = ENABLE_OCR) -> DocumentConverter:
    settings.perf.page_batch_size = DOCLING_PAGE_BATCH_SIZE
    settings.perf.page_batch_concurrency = 1
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.do_table_structure = ENABLE_TABLE_STRUCTURE
    pipeline_options.queue_max_size = DOCLING_QUEUE_MAX_SIZE
    pipeline_options.ocr_batch_size = DOCLING_MODEL_BATCH_SIZE
    pipeline_options.layout_batch_size = DOCLING_MODEL_BATCH_SIZE
    pipeline_options.table_batch_size = DOCLING_MODEL_BATCH_SIZE
    pipeline_options.accelerator_options.num_threads = DOCLING_NUM_THREADS
    pipeline_options.accelerator_options.device = DOCLING_DEVICE

    if ENABLE_TABLE_STRUCTURE:
        pipeline_options.table_structure_options.mode = TABLE_MODE
        pipeline_options.table_structure_options.do_cell_matching = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


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


def infer_document_title(
    records: list[ChunkRecord],
    pdf_path: Path,
) -> str:
    for record in records[:5]:
        if record.headings:
            return record.headings[0]
        if "title" in record.labels and record.raw_text:
            return record.raw_text.splitlines()[0][:300]

    return pdf_path.stem


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

    child_entries: list[tuple[int, ChunkRecord, str, int, int, str]] = []
    for parent_index, record in enumerate(records):
        parent_chunk_id = make_chunk_id(
            source_id=source_id,
            file_hash=file_hash,
            chunk_index=parent_index,
            text=record.content,
            ingestion_signature=current_ingestion_fingerprint + "|parent",
        )
        children = split_text_for_retrieval(record.content, runtime)
        for child_index, child_text in enumerate(children, start=1):
            child_entries.append(
                (
                    parent_index,
                    record,
                    parent_chunk_id,
                    child_index,
                    len(children),
                    child_text,
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
            "content_labels": ", ".join(record.labels),
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
            "chunker": "hybrid-parent-child",
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


def embed_documents_in_batches(
    documents: list[Document],
    embeddings: Any,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for document_batch in batched(documents, BATCH_SIZE):
        batch = list(document_batch)
        vectors.extend(
            embeddings.embed_documents_with_titles(
                [document.page_content for document in batch],
                [document.metadata.get("document_title") for document in batch],
            )
        )
    return vectors


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

def debug_base_path(source_id: str) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_id).stem)
    suffix = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:10]
    return DEBUG_DIR / f"{safe_stem}-{suffix}"


def export_debug_files(
    source_id: str,
    markdown: str,
    documents: list[Document],
) -> None:
    if not EXPORT_DEBUG_FILES:
        return

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    base_path = debug_base_path(source_id)
    base_path.with_suffix(".md").write_text(markdown, encoding="utf-8")

    with base_path.with_suffix(".chunks.jsonl").open(
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


def convert_page_range_resilient(
    converter: DocumentConverter,
    pdf_path: Path,
    page_start: int,
    page_end: int,
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
            f"Docling could not convert page {page_start} of {pdf_path.name}: "
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
    ) + convert_page_range_resilient(
        converter,
        pdf_path,
        midpoint + 1,
        page_end,
    )


def should_retry_with_ocr(markdown: str, page_count: int) -> bool:
    if page_count <= 0:
        return not clean_text(markdown)
    extracted_characters = len(re.sub(r"\s+", "", markdown))
    return extracted_characters / page_count < MIN_EXTRACTED_CHARS_PER_PAGE


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
    commit_gate: Path | None = None,
) -> ProcessResult:
    source_id = pdf_path.resolve().relative_to(PDF_DIR.resolve()).as_posix()
    file_hash = calculate_file_hash(pdf_path)
    ocr_policy = "force" if ocr_enabled else "auto" if auto_ocr else "off"
    current_fingerprint = ingestion_fingerprint(
        runtime.tokenizer_name,
        ocr_policy,
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

    page_count = get_pdf_page_count(pdf_path)
    if page_count <= 0:
        print("  PDF contains no pages.")
        return ProcessResult(status="no_output")

    records: list[ChunkRecord] = []
    markdown_parts: list[str] = []
    carried_headings: list[str] = []
    for page_start in range(1, page_count + 1, PDF_PAGE_WINDOW):
        page_end = min(page_start + PDF_PAGE_WINDOW - 1, page_count)
        print(f"  Converting pages {page_start}-{page_end} of {page_count}")
        base_results = convert_page_range_resilient(
            converter,
            pdf_path,
            page_start,
            page_end,
        )

        while base_results:
            base_result = base_results.pop(0)
            base_document = base_result.document
            base_pages = sorted(int(page) for page in base_document.pages)
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
                print(
                    f"  Sparse text on pages {retry_start}-{retry_end}; "
                    "retrying that range with OCR."
                )
                final_results = [
                    (result, True)
                    for result in convert_page_range_resilient(
                        ocr_converter,
                        pdf_path,
                        retry_start,
                        retry_end,
                    )
                ]
                base_result = None
                base_document = None
                gc.collect()

            for final_result, ocr_applied in final_results:
                docling_document = final_result.document
                batch_markdown = docling_document.export_to_markdown()
                markdown_parts.append(batch_markdown)
                raw_chunks = runtime.chunker.chunk(dl_doc=docling_document)
                window_records = build_chunk_records(
                    raw_chunks,
                    runtime,
                    ocr_applied=ocr_applied,
                )
                window_records, carried_headings = stitch_window_headings(
                    window_records,
                    carried_headings,
                    runtime,
                )
                records.extend(window_records)
                del raw_chunks, docling_document, final_result
                gc.collect()
            final_results.clear()
            base_result = None
            base_document = None
            gc.collect()

        release_cuda_memory()

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

    if not documents:
        print("  No usable chunks were produced.")
        return ProcessResult(status="no_output")

    # Complete every model call before touching the active index. In concurrent
    # mode, the app pauses search only for this short commit window so dense and
    # lexical retrieval never observe different source revisions.
    vectors = embed_documents_in_batches(documents, embeddings)
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
        }
        save_manifest(manifest)

    try:
        export_debug_files(source_id, markdown, documents)
    except OSError as error:
        print(f"  WARNING: debug export failed: {error}")
    show_document_summary(documents, markdown)

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
    print("CORPUS_EVENT " + json.dumps(payload, ensure_ascii=False), flush=True)


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
    report_cuda_headroom()

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

    converter = create_converter(enable_ocr=ocr_enabled)
    ocr_converter = create_converter(enable_ocr=True) if auto_ocr else None
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
                commit_gate=args.commit_gate,
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
