"""Build a source-only archive that cannot include the private corpus."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release" / "reference-desk-source.zip"
EXCLUDED_PARTS = {
    ".git", ".venv", ".pytest_cache", ".uv-cache", "__pycache__",
    "node_modules", ".npm-cache", ".vite", "release", "chroma_db",
    "ingestion_debug", "document_trash", "document_quarantine",
    "document_revisions", "corpus_backups",
}
EXCLUDED_NAMES = {".rag-profile", ".env", "reference_workspace.sqlite3"}
EXCLUDED_SUFFIXES = {".pdf", ".log", ".pyc", ".pyo", ".sqlite3", ".sqlite3-shm", ".sqlite3-wal"}


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.name.startswith("evaluation_results"):
            continue
        if any(path.name.lower().endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="List the safe export without writing it.")
    args = parser.parse_args()
    files = included_files()
    if args.check:
        print(f"Safe source export: {len(files)} files")
        for path in files:
            print(path.relative_to(ROOT).as_posix())
        return 0

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "product": "Reference Desk",
        "format": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "contains_local_corpus": False,
        "setup": {
            "windows": ".\\scripts\\setup.ps1 -Backend auto",
            "linux": "./scripts/setup.sh auto",
        },
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("release-manifest.json", json.dumps(manifest, indent=2))
    print(f"Created {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print("Private PDFs, indexes, notes, logs, and backups were excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
