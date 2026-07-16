"""Build a friend-friendly archive that cannot include the private corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release" / "reference-desk-windows.zip"
ARCHIVE_ROOT = "reference-desk"
EXCLUDED_PARTS = {
    ".git", ".venv", ".pytest_cache", ".uv-cache", "__pycache__",
    "node_modules", ".npm-cache", ".vite", "release", "chroma_db",
    "ingestion_debug", "document_trash", "document_quarantine",
    "document_revisions", "corpus_backups",
}
EXCLUDED_NAMES = {".rag-profile", ".env", "reference_workspace.sqlite3"}
EXCLUDED_SUFFIXES = {".pdf", ".log", ".pyc", ".pyo", ".sqlite3", ".sqlite3-shm", ".sqlite3-wal"}


def candidate_files() -> list[Path]:
    """Prefer Git's allow-list, with a safe fallback for downloaded copies."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [path for path in ROOT.rglob("*") if path.is_file()]
    return [ROOT / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in candidate_files():
        if not path.is_file() or not path.is_relative_to(ROOT):
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


def verify_safe(files: list[Path]) -> None:
    """Fail closed if a private-data path ever reaches the archive list."""

    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            raise RuntimeError(f"Refusing to package private directory: {relative}")
        if path.name in EXCLUDED_NAMES or path.name.startswith("evaluation_results"):
            raise RuntimeError(f"Refusing to package local data: {relative}")
        if any(path.name.lower().endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
            raise RuntimeError(f"Refusing to package private file type: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="List the safe export without writing it.")
    args = parser.parse_args()
    files = included_files()
    verify_safe(files)
    if args.check:
        print(f"Safe source export: {len(files)} files")
        for path in files:
            print(path.relative_to(ROOT).as_posix())
        return 0

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "product": "Reference Desk",
        "format": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "contains_local_corpus": False,
        "setup": {
            "windows": "Double-click SETUP.bat, then START.bat",
            "linux": "./scripts/setup.sh auto",
        },
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{ARCHIVE_ROOT}/{relative}")
        archive.writestr(
            f"{ARCHIVE_ROOT}/release-manifest.json",
            json.dumps(manifest, indent=2),
        )
    print(f"Created {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print("Private PDFs, indexes, notes, logs, and backups were excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
