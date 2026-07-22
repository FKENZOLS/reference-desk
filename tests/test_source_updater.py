from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import source_updater


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def commit_file(path: Path, name: str, content: str, message: str) -> None:
    (path / name).write_text(content, encoding="utf-8")
    git(path, "add", name)
    git(path, "commit", "-m", message)


@pytest.fixture()
def update_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    remote.mkdir()
    git(remote, "init", "--bare", "--initial-branch=main")
    seed.mkdir()
    git(seed, "init", "--initial-branch=main")
    git(seed, "config", "user.email", "tests@example.com")
    git(seed, "config", "user.name", "Reference Desk tests")
    (seed / "pyproject.toml").write_text(
        '[project]\nname = "reference-desk"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    git(seed, "add", "pyproject.toml")
    git(seed, "commit", "-m", "initial")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-u", "origin", "main")
    git(tmp_path, "clone", str(remote), str(checkout))
    git(checkout, "config", "user.email", "tests@example.com")
    git(checkout, "config", "user.name", "Reference Desk tests")
    monkeypatch.setattr(source_updater, "PROJECT_DIR", checkout)
    monkeypatch.setattr(
        source_updater,
        "UPDATE_MARKER_PATH",
        checkout / ".reference-desk-update.json",
    )
    return seed, checkout


def test_status_detects_available_fast_forward(update_checkout: tuple[Path, Path]) -> None:
    seed, checkout = update_checkout
    commit_file(seed, "feature.txt", "new", "new feature")
    git(seed, "push")

    status = source_updater.source_update_status(fetch=True)

    assert status["available"] is True
    assert status["can_update"] is True
    assert status["behind"] == 1
    assert status["version"] == "1.2.3"
    assert status["latest_subject"] == "new feature"
    assert status["local_commit"] == git(checkout, "rev-parse", "HEAD")


def test_local_tracked_change_blocks_update(update_checkout: tuple[Path, Path]) -> None:
    _, checkout = update_checkout
    (checkout / "pyproject.toml").write_text("changed locally", encoding="utf-8")

    status = source_updater.source_update_status()

    assert status["can_update"] is False
    assert status["changed_paths"] == ["pyproject.toml"]
    assert "local changes" in status["blocked_reason"].lower()


def test_apply_uses_fast_forward_and_writes_restart_marker(
    update_checkout: tuple[Path, Path],
) -> None:
    seed, checkout = update_checkout
    commit_file(seed, "requirements-base.txt", "fastapi\n", "dependency update")
    git(seed, "push")

    status = source_updater.apply_source_update()

    assert status["updated"] is True
    assert status["restart_required"] is True
    assert status["dependencies_changed"] is True
    assert git(checkout, "rev-parse", "HEAD") == git(seed, "rev-parse", "HEAD")
    marker = json.loads((checkout / ".reference-desk-update.json").read_text(encoding="utf-8"))
    assert marker["dependencies_changed"] is True
    assert "requirements-base.txt" in marker["changed_paths"]
