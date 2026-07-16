"""Single command-line entry point for the project."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT_COMMANDS = {
    "serve": "search_app.py",
    "ingest": "ingest.py",
    "evaluate": "evaluate.py",
    "doctor": "scripts/doctor.py",
    "export": "scripts/export_release.py",
}
HELP = """Local document retrieval

Usage:
  python main.py serve
  python main.py ingest [ingest options]
  python main.py evaluate <benchmark.jsonl> [evaluation options]
  python main.py doctor [--allow-low-vram]
  python main.py export [--output archive.zip]
  python main.py test [pytest options]

Running without a command starts the search app.
"""


def command_for(command: str, arguments: Sequence[str]) -> list[str]:
    """Build a child command using the active Python environment."""

    if command == "test":
        return [sys.executable, "-m", "pytest", *arguments]
    try:
        script = SCRIPT_COMMANDS[command]
    except KeyError as error:
        raise ValueError(f"Unknown command: {command}") from error
    return [sys.executable, str(PROJECT_DIR / script), *arguments]


def launch_server() -> None:
    """Run the web server here so closing this terminal always stops it."""

    from search_app import launch_gradio

    launch_gradio()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"-h", "--help", "help"}:
        print(HELP)
        return 0

    command = arguments.pop(0) if arguments else "serve"
    if command == "serve":
        if arguments:
            print("The serve command does not accept arguments.", file=sys.stderr)
            return 2
        launch_server()
        return 0
    try:
        child_command = command_for(command, arguments)
    except ValueError as error:
        print(f"{error}\n\n{HELP}", file=sys.stderr)
        return 2
    return subprocess.run(
        child_command,
        cwd=PROJECT_DIR,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
