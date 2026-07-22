"""Safe, fast-forward-only updates for source-based installations."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit


PROJECT_DIR = Path(__file__).resolve().parent
UPDATE_MARKER_PATH = PROJECT_DIR / ".reference-desk-update.json"
DEPENDENCY_FILES = {
    "pyproject.toml",
    "requirements.txt",
    "requirements-base.txt",
    "requirements-cpu.txt",
    "requirements-cuda.txt",
    "requirements-rocm-linux.txt",
    "requirements-rocm-windows.txt",
    "scripts/setup.ps1",
}


class SourceUpdateError(RuntimeError):
    """Raised when an update cannot be checked or applied safely."""


@dataclass(frozen=True)
class GitResult:
    stdout: str
    stderr: str
    returncode: int


_UPDATE_LOCK = threading.Lock()


def _run_git(
    arguments: Sequence[str],
    *,
    timeout: float = 45,
    check: bool = True,
) -> GitResult:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=(
                int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if os.name == "nt"
                else 0
            ),
        )
    except FileNotFoundError as error:
        raise SourceUpdateError(
            "Git is not installed or is not available on PATH."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise SourceUpdateError("GitHub did not respond before the update check timed out.") from error
    result = GitResult(
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        returncode=completed.returncode,
    )
    if check and result.returncode:
        message = result.stderr or result.stdout or "Git command failed."
        raise SourceUpdateError(message.splitlines()[-1])
    return result


def _git_output(arguments: Sequence[str], *, timeout: float = 45) -> str:
    return _run_git(arguments, timeout=timeout).stdout


def _safe_remote_url(value: str) -> str:
    """Remove embedded credentials before returning a remote URL to the UI."""

    if "://" not in value:
        return value
    parts = urlsplit(value)
    hostname = parts.hostname or ""
    if parts.port:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, parts.query, parts.fragment))


def _tracking_branch(branch: str) -> str:
    result = _run_git(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        check=False,
    )
    if not result.returncode and result.stdout:
        return result.stdout
    fallback = f"origin/{branch}"
    exists = _run_git(["show-ref", "--verify", "--quiet", f"refs/remotes/{fallback}"], check=False)
    if exists.returncode:
        raise SourceUpdateError(
            "This branch has no GitHub tracking branch. Configure an upstream branch first."
        )
    return fallback


def _changed_tracked_paths() -> list[str]:
    unstaged = _git_output(["diff", "--name-only"])
    staged = _git_output(["diff", "--cached", "--name-only"])
    return sorted({line for text in (unstaged, staged) for line in text.splitlines() if line})


def _version() -> str:
    try:
        import tomllib

        with (PROJECT_DIR / "pyproject.toml").open("rb") as handle:
            return str(tomllib.load(handle).get("project", {}).get("version") or "unknown")
    except (OSError, ValueError):
        return "unknown"


def source_update_status(*, fetch: bool = False) -> dict[str, Any]:
    """Describe whether the checkout can safely fast-forward from GitHub."""

    inside = _run_git(["rev-parse", "--is-inside-work-tree"], check=False)
    if inside.returncode or inside.stdout.lower() != "true":
        raise SourceUpdateError("This installation is not a Git checkout.")

    branch = _git_output(["branch", "--show-current"])
    if not branch:
        raise SourceUpdateError("Updates are disabled while Git is in detached HEAD state.")
    upstream = _tracking_branch(branch)
    remote = upstream.split("/", 1)[0]
    if fetch:
        _run_git(["fetch", "--prune", remote], timeout=120)

    local_commit = _git_output(["rev-parse", "HEAD"])
    remote_commit = _git_output(["rev-parse", upstream])
    counts = _git_output(
        ["rev-list", "--left-right", "--count", f"HEAD...{upstream}"]
    ).split()
    if len(counts) != 2:
        raise SourceUpdateError("Git could not compare the local and GitHub revisions.")
    ahead, behind = (int(counts[0]), int(counts[1]))
    changed_paths = _changed_tracked_paths()
    divergent = ahead > 0 and behind > 0
    available = behind > 0 and ahead == 0
    blocked_reason = ""
    if changed_paths:
        blocked_reason = "Tracked source files have local changes. Commit or discard them before updating."
    elif divergent:
        blocked_reason = "The local and GitHub histories have diverged. Resolve them with Git before updating."
    elif ahead > 0:
        blocked_reason = "This checkout contains local commits that are not on GitHub."

    subject = _git_output(["log", "-1", "--format=%s", upstream])
    commit_date = _git_output(["log", "-1", "--format=%cI", upstream])
    remote_url = _safe_remote_url(_git_output(["remote", "get-url", remote]))
    supervised = os.environ.get("RAG_UPDATE_SUPERVISED", "").strip() == "1"
    return {
        "available": available,
        "can_update": available and not blocked_reason,
        "blocked_reason": blocked_reason or None,
        "branch": branch,
        "upstream": upstream,
        "remote_url": remote_url,
        "version": _version(),
        "local_commit": local_commit,
        "remote_commit": remote_commit,
        "local_short": local_commit[:8],
        "remote_short": remote_commit[:8],
        "latest_subject": subject,
        "latest_date": commit_date,
        "ahead": ahead,
        "behind": behind,
        "changed_paths": changed_paths,
        "supervised_restart": supervised,
        "checked_remote": fetch,
        "restart_required": False,
    }


def apply_source_update() -> dict[str, Any]:
    """Fetch and apply one clean fast-forward update without overwriting files."""

    if not _UPDATE_LOCK.acquire(blocking=False):
        raise SourceUpdateError("Another source update is already running.")
    try:
        before = source_update_status(fetch=True)
        if before["blocked_reason"]:
            raise SourceUpdateError(str(before["blocked_reason"]))
        if not before["available"]:
            return before

        old_commit = str(before["local_commit"])
        upstream = str(before["upstream"])
        _run_git(["merge", "--ff-only", upstream], timeout=120)
        after = source_update_status(fetch=False)
        changed = _git_output(["diff", "--name-only", f"{old_commit}..{after['local_commit']}"])
        changed_paths = [line for line in changed.splitlines() if line]
        dependencies_changed = any(path in DEPENDENCY_FILES for path in changed_paths)
        marker = {
            "old_commit": old_commit,
            "new_commit": after["local_commit"],
            "dependencies_changed": dependencies_changed,
            "changed_paths": changed_paths,
        }
        UPDATE_MARKER_PATH.write_text(
            json.dumps(marker, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            **after,
            "updated": True,
            "restart_required": True,
            "dependencies_changed": dependencies_changed,
            "updated_paths": changed_paths,
        }
    finally:
        _UPDATE_LOCK.release()
