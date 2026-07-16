import sys

import main as main_module
from main import PROJECT_DIR, command_for, main


def test_command_for_uses_active_python_and_project_scripts() -> None:
    assert command_for("ingest", ["--prune"]) == [
        sys.executable,
        str(PROJECT_DIR / "ingest.py"),
        "--prune",
    ]
    assert command_for("test", ["-q"]) == [sys.executable, "-m", "pytest", "-q"]


def test_help_does_not_launch_a_child_process(capsys) -> None:
    assert main(["--help"]) == 0
    assert "python main.py serve" in capsys.readouterr().out


def test_unknown_command_is_reported(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "Unknown command" in capsys.readouterr().err


def test_serve_runs_in_launcher_process(monkeypatch) -> None:
    launched = []
    monkeypatch.setattr(main_module, "launch_server", lambda: launched.append(True))

    assert main(["serve"]) == 0
    assert launched == [True]
