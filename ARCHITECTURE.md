# Project architecture

This document is the maintenance map. The README describes how to operate the
app; this file describes where behavior belongs and what must remain stable.

## System boundary

The project is a local evidence-retrieval application:

```text
PDFs
  -> Docling conversion and provenance
  -> parent/child chunk preparation
  -> Qwen3-Embedding vectors + SQLite FTS5
  -> reciprocal-rank fusion
  -> Qwen3-Reranker yes/no scoring
  -> results, feedback, citations, PDF viewer, and research workspace
```

There is no answer-generation model. Search results are source passages and
every product feature should preserve a route back to the original document.

## Module ownership

| Module | Owns | Should not own |
| --- | --- | --- |
| `main.py` | Stable CLI and command dispatch | Retrieval or ingestion logic |
| `app_settings.py` | Search, reranking, and server environment settings | Storage and embedding clients |
| `hardware.py` | Backend detection, saved profile, and CUDA/ROCm device mapping | Model loading or UI policy |
| `rag_common.py` | Shared paths, embedding fingerprint, Ollama embeddings | UI and ranking policy |
| `ingest.py` | Docling conversion, chunking, provenance, embedding, Chroma writes | Search presentation |
| `lexical_index.py` | SQLite FTS schema, query lanes, and lexical persistence | Dense retrieval |
| `search_app.py` | Runtime lifecycle, hybrid retrieval, reranking, citations, JSON APIs, PDF viewer | Document file mutations |
| `frontend/src/` | React pages and local shadcn-style presentation components | Retrieval, persistence, or GPU lifecycle |
| `frontend/dist/` | Checked-in production UI bundle served by FastAPI | Hand-edited source |
| `document_manager.py` | Safe PDF repository operations and legacy HTML fallback | Chroma ingestion internals |
| `corpus_scale.py` | Persistent queue state, health/storage, quarantine coordination, backups | Docling conversion or search ranking |
| `workspace_store.py` | History, bookmarks, notes, collections, and export data | Search ranking |
| `workspace_ui.py` | Legacy research-workspace fallback | SQLite queries |
| `quality_ui.py` | Legacy quality-dashboard fallback | Threshold fitting or retrieval |
| `evaluate.py` | Offline labeled evaluation and metrics | Production state changes |

## Runtime data

These paths are local state, not source code:

| Path | Contents |
| --- | --- |
| `docs/` | Private source PDFs |
| `chroma_db/` | Chroma collection, manifest, and lexical SQLite index |
| `ingestion_debug/` | Optional conversion and chunk diagnostics |
| `document_trash/` | Recoverable deleted PDFs |
| `document_quarantine/` | Failed PDFs and their error records |
| `document_revisions/` | Archived PDFs replaced by a newer revision |
| `corpus_backups/` | Validated portable corpus ZIP snapshots |
| `reference_workspace.sqlite3*` | Notes, collections, history, feedback, and calibration |
| `evaluation_results*.json` | Generated benchmark output |

They are ignored by Git. Back them up separately when the corpus or research
workspace matters.

## Critical invariants

1. `source_id` is a normalized relative path, not only a filename.
2. Source paths must resolve inside `RAG_PDF_DIR`; traversal and non-PDF files
   must remain rejected.
3. Chunk IDs and index fingerprints must change when stored chunk semantics or
   embedding configuration changes.
4. A source becomes current only after its complete document update succeeds.
5. Manual pruning stays explicit. The document manager may request pruning
   because it owns the staged file changes.
6. Search and ingestion may share a GPU only when live free VRAM covers the
   Docling headroom plus a query reserve. Otherwise ingestion releases search
   models and runs exclusively; total VRAM alone is not the deciding factor.
7. Citation metadata must retain page and bounding-box provenance when Docling
   provides it.
8. Workspace data is independent from ingestion and must survive index rebuilds.
9. Wrong-result labels are hard negatives; only explicit no-result judgments may create unanswerable benchmark cases.
10. Learned rejection remains inactive until both configured label minimums are met.
11. Queue pause is cooperative between documents; an active source must finish or fail before the process exits.
12. One failed source must not roll back successfully completed sources or block the remaining queue.
13. Snapshot restore must reject unsafe archive paths and keep a rollback copy until all corpus paths are replaced.
14. AMD ROCm uses PyTorch's `torch.cuda` API; vendor detection must use
    `torch.version.hip`, never a different device string.
15. Hardware-specific PyTorch wheels must be installed before the portable
    dependency layer, and local hardware profiles must never be committed.
16. Embedding dimensions, prompts, model revisions, and collection names are
    part of the index identity. Qwen's 1024-dimensional vectors must never be
    inserted into a legacy 768-dimensional collection.
17. Relevance calibration is reranker-specific. Scores from BGE and Qwen must
    never be fitted into the same threshold.

## Application lifecycle

### Search

1. Load the Chroma collection, embedding client, lexical state, and reranker.
2. Retrieve dense and lexical candidates independently.
3. Fuse the candidate ranks.
4. Rerank the bounded fused shortlist.
5. Apply diversity and optional calibrated relevance filtering.
6. Register source navigation and render citation links.
7. Return structured result, score, citation, and feedback payloads to React.
8. Persist explicit user judgments separately from the retrieval index.

### Managed ingestion

1. Stage document changes in `DocumentRepository` and create persistent queue items.
2. Block new searches and wait for the current search lock.
3. Measure free VRAM. Keep the loaded search runtime only when the combined
   Docling and query reserve fits; otherwise release the reranker and retained
   Ollama embedding model.
4. Run the selected sources through `ingest.py --queue-managed --prune`.
5. Commit each successful source immediately; quarantine a failed source and continue.
6. Honor pause requests before beginning the next document.
7. Prune deleted and quarantined sources, then reload search resources lazily.
   Concurrent searches may see the previous or newly committed source state
   while a document is being processed.

## Where to make common changes

| Change | Start here | Required verification |
| --- | --- | --- |
| Chunk size, overlap, or parent context | `ingest.py` | Rebuild index and run ingestion tests |
| Embedding model or revision | `rag_common.py` | New fingerprint and full rebuild |
| Candidate counts or reranker policy | `app_settings.py`, `search_app.py` | Labeled evaluation |
| Exact-term matching | `lexical_index.py` | Lexical and retrieval tests |
| Citation geometry or PDF viewer | `search_app.py` citation/viewer section | Viewer HTML tests and browser check |
| Document upload, move, or trash | `document_manager.py` | Path and route tests |
| Queue, quarantine, revisions, health, or backups | `corpus_scale.py`, `document_manager.py` | Corpus Scale tests and restore smoke test |
| Search, library, workspace, or quality layout | `frontend/src/` | Vite build, route tests, and browser check |
| Notes, collections, or exports | `workspace_store.py`, `frontend/src/pages/workspace-page.tsx` | Workspace tests |
| Feedback, benchmark export, or evidence calibration | `workspace_store.py`, `frontend/src/pages/quality-page.tsx` | Reference-quality tests and labeled evaluation |
| HTTP routes or lifecycle | `search_app.py:create_web_app` | Route tests and live smoke test |

## Test layout

- `test_ingestion_logic.py`: chunk preparation and provenance behavior.
- `test_hardware.py`: CUDA/ROCm detection, profile precedence, and VRAM reporting.
- `test_lexical_index.py`: FTS query construction and persistence.
- `test_retrieval_logic.py`: fusion, reranking, citations, and viewer output.
- `test_search_api.py`: structured React search payload and citation link.
- `test_document_manager.py`: repository safety, routes, and ingestion lifecycle.
- `test_corpus_scale.py`: queue recovery, quarantine, revisions, storage health, and backup/restore.
- `test_workspace_store.py`: research data, HTML, routes, and exports.
- `test_reference_quality.py`: judgments, benchmark generation, calibration, gating, and dashboard routes.
- `test_evaluation_logic.py`: metric correctness.
- `test_main.py`: stable CLI dispatch.

Run the full suite with `python main.py test`.

The UI is a separate Vite project under `frontend/`. FastAPI uses the built
bundle when `frontend/dist/index.html` exists and retains the old HTML pages as
a safe fallback during development. The frontend talks only to versioned local
route families (`/api`, `/documents/api`, `/workspace/api`, `/quality/api`) and
does not import Python-side behavior.

## Next safe extractions

The project is now operationally organized, but two modules remain deliberately
large. Future refactors should be behavioral no-ops and happen one boundary at
a time:

1. Move the pure citation-viewer HTML/CSS/JavaScript builder from
   `search_app.py` into `viewer_ui.py`.
2. Move FastAPI route registration into route modules that receive explicit
   repository and service dependencies.
3. Split `ingest.py` into conversion, chunk preparation, and index persistence
   after adding end-to-end fixture coverage for one small PDF.

Avoid reorganizing all three boundaries in one change. The current tests are
strong at logic and route level, but a small deterministic end-to-end PDF
fixture should precede deeper ingestion extraction.
