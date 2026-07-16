# Local technical-document retrieval

A local reference workspace for technical PDFs. It ingests documents with
Docling, creates EmbeddingGemma vectors through Ollama, combines dense search
with SQLite FTS5, and reranks passages with `BAAI/bge-reranker-v2-m3`.

The application retrieves evidence; it does not generate answers or rewrite
queries with an LLM.

## Capabilities

- Search every indexed document or restrict by source and metadata filters.
- Parent-child retrieval with exact terms, semantic search, and BGE reranking.
- Offline citations linked to the correct PDF page and Docling source region.
- Selectable PDF text, passage navigation, thumbnails, and copy controls.
- Saved passages, notes, collections, comparison, and Markdown/Word export.
- A document library with a persistent ingestion queue, pause/resume, and quarantine.
- Duplicate blocking, revision rollback, corpus health/storage, and portable backups.
- Result-level relevance feedback, benchmark export, and a quality dashboard.
- A learned "no strong evidence" gate that activates only after enough labels.
- Offline retrieval evaluation with recall, MRR, nDCG, rejection, and latency.
- A responsive React interface built from local shadcn-style components.

## Requirements

- Windows or Linux with Python 3.12.
- Ollama with `embeddinggemma`.
- An NVIDIA CUDA or AMD ROCm-compatible GPU with at least 8 GB VRAM.
- A current NVIDIA/AMD driver. AMD Windows systems must also be supported by
  AMD's current PyTorch-on-Windows package.
- Node.js 20 or newer only when changing the React interface.

The portable installer detects the GPU, installs the matching PyTorch build,
records the local hardware profile, installs the application, and checks VRAM:

```powershell
.\scripts\setup.ps1 -Backend auto
.\start.ps1
```

Force a particular Windows backend when needed:

```powershell
.\scripts\setup.ps1 -Backend rocm
.\scripts\setup.ps1 -Backend cuda
```

On Linux, run `bash scripts/setup.sh auto`, then `bash start.sh`. The setup
scripts install PyTorch before Docling so a transitive dependency cannot select
the wrong GPU package. `requirements.txt` therefore contains only the portable
application layer; do not install it alone on a new GPU machine.

ROCm deliberately uses PyTorch device names such as `cuda:0`. The application
tracks the actual vendor separately and shows `AMD ROCm` or `NVIDIA CUDA` in
logs and diagnostics. Ollama selects its own AMD ROCm/Vulkan or NVIDIA backend,
so its GPU choice is not inferred from the PyTorch package.

## Commands

`main.py` is the single project entry point:

```powershell
python main.py serve
python main.py ingest
python main.py ingest --force --prune
python main.py evaluate benchmark.starter.jsonl
python main.py doctor
python main.py export
python main.py test
```

Running `python main.py` without a command also starts the search app. The
original script entry points remain available for debugging. The `serve`
command runs the server in the launcher process, so closing its terminal or
pressing `Ctrl+C` stops the web server without leaving an orphan process.

The production interface is already bundled in `frontend/dist`, so normal
users do not need Node.js. When changing files under `frontend/src`, rebuild it:

```powershell
cd frontend
npm install
npm run build
cd ..
python main.py test
```

FastAPI serves the bundled React application at `/`, `/documents`, `/workspace`,
and `/quality`. The specialized citation viewer remains server-rendered at
`/viewer/...` because it owns PDF page rendering, selectable text, and exact
Docling overlays.

## Normal workflow

1. Start the app with `python main.py serve`.
2. Open `http://127.0.0.1:7860/documents`.
3. Add, organize, or remove PDFs.
4. Select **Apply pending changes**.
5. Return to `http://127.0.0.1:7860/` and search the complete collection.
6. Label useful and incorrect results; review progress at `/quality`.

The document manager serializes GPU-heavy work: it pauses new searches,
releases retained search models, runs ingestion with pruning, and reloads
search models lazily. **Pause** finishes the active document and stops before
the next one. A failed PDF moves to quarantine without blocking the rest of
the queue; retrying it returns the file to pending changes.

Exact duplicate uploads are rejected. Replacing a PDF stores its prior version
in revision history, where it can be restored later. The same page reports
corpus health and disk use for PDFs, indexes, workspace data, quarantine,
revisions, debug data, and backups.

**Create backup** writes one portable ZIP snapshot containing the source PDFs,
Chroma/lexical indexes, workspace database, trash, quarantine, and revisions.
Restore replaces those local data stores only after the archive is validated.
The post-ingestion **Reload search models** control resets model state only;
after editing Python source, stop and start the terminal server instead.

For manual ingestion, place PDFs under `docs/` and run:

```powershell
python main.py ingest --prune
```

Useful ingestion options are `--force`, `--ocr`, and `--no-auto-ocr`.

## Move to another computer

Application source and private research data use separate transfer paths:

- Publish the source as a private GitHub repository with
  `.\scripts\publish_github.ps1`, then clone it on the other computer.
- Or run `.\.venv\Scripts\python.exe scripts\export_release.py` and copy the
  generated `release/reference-desk-source.zip`.
- Move PDFs, indexes, notes, and collections with **Create backup** on the
  Documents page. They are intentionally never included in the source ZIP.

Tagged GitHub releases automatically receive the safe source ZIP. Full steps
are in [GITHUB_TRANSFER.md](GITHUB_TRANSFER.md).

## Evaluation

The starter benchmark contains page-verified English, Portuguese, table,
acronym, and unanswerable cases:

```powershell
python main.py evaluate benchmark.starter.jsonl --output evaluation_results.json
```

Tune retrieval only after inspecting the per-stage recall and latency output.
The most relevant controls are:

- `RAG_DENSE_CANDIDATES`
- `RAG_LEXICAL_CANDIDATES`
- `RAG_RERANK_CANDIDATES`
- `RAG_RERANK_WEIGHT`
- `RAG_RERANK_MAX_LENGTH`
- `RAG_MAX_RESULTS_PER_SECTION`
- `RAG_MAX_RESULTS_PER_PAGE`

The **Reference quality** page exports real search judgments as evaluation
JSONL. A relevant label creates an answerable case; an explicit **No relevant
result** label creates an unanswerable case. Wrong-passage and wrong-document
labels are retained as hard negatives but never claim that the corpus has no
answer.

The automatic evidence gate defaults to collecting mode. It becomes eligible
only after 20 scored relevant and 20 scored incorrect judgments, with a learned
cutoff constrained to retain at least 90% of labeled relevant passages. It can
be paused from the quality dashboard.

## Configuration

Paths are resolved relative to the project directory. Common overrides:

| Setting | Purpose | Default |
| --- | --- | --- |
| `RAG_PDF_DIR` | Source PDF directory | `docs` |
| `RAG_DB_DIR` | Chroma and manifest directory | `chroma_db` |
| `RAG_WORKSPACE_DB` | Notes and collections database | `reference_workspace.sqlite3` |
| `RAG_COLLECTION` | Chroma collection | `technical_docs_v3` |
| `RAG_ACCELERATOR` | Hardware backend: `auto`, `cuda`, `rocm`, or `cpu` | saved profile or `auto` |
| `RAG_DEVICE` | Legacy alias for `RAG_ACCELERATOR` | unset |
| `RAG_DOCLING_DEVICE` | Docling device override | detected GPU |
| `RAG_RERANKER_DEVICE` | Reranker device override | detected GPU |
| `RAG_OLLAMA_ACCELERATOR` | Ollama offload: `auto` or `cpu` | `auto` |
| `RAG_GPU_HEADROOM_WARNING_MB` | Free VRAM required before ingestion | `3500` |
| `RAG_PDF_PAGE_WINDOW` | Docling conversion window | `6` |
| `RAG_OPEN_BROWSER` | Open a browser at startup | `1` |
| `RAG_SERVER_PORT` | Local web port | `7860` |
| `RAG_QUALITY_MIN_POSITIVE_LABELS` | Relevant labels before calibration | `20` |
| `RAG_QUALITY_MIN_NEGATIVE_LABELS` | Incorrect labels before calibration | `20` |
| `RAG_QUALITY_MIN_RECALL` | Minimum labeled-relevant recall for cutoff | `0.90` |

Embedding revisions are fingerprinted. If the model revision or ingestion
format changes, rebuild the affected index instead of mixing incompatible
vectors.

## GPU memory

The supported target is 8 GB VRAM or more on NVIDIA CUDA or AMD ROCm. Do not
run an independent ingestion process while the search model is resident. Use
the document manager because it releases the reranker and retained Ollama
model before ingestion. The compatibility check can be repeated at any time:

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py
```

For lower-memory conversion, reduce the page window before rebuilding:

```powershell
$env:RAG_PDF_PAGE_WINDOW = "3"
python main.py ingest --force
```

## Project maintenance

- Run `python main.py test` before and after structural changes.
- Run `npm run build` in `frontend/` after any interface change.
- Keep PDFs, indexes, logs, debug exports, and workspace databases local.
- Keep hardware-specific PyTorch requirements separate from portable packages.
- Run `scripts/export_release.py --check` before publishing source.
- Read [ARCHITECTURE.md](ARCHITECTURE.md) before moving retrieval, ingestion,
  citation, or workspace responsibilities between modules.

The source tree intentionally excludes local documents and generated data, so
it can be shared without packaging the private corpus or Chroma database.

