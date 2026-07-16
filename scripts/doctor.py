"""Check whether this installation can use its selected accelerator."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware import accelerator_info, backend_label, normalize_accelerator  # noqa: E402


MINIMUM_GPU_BYTES = int(7.5 * 1024**3)


def ollama_status() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        return False, f"Ollama is not reachable ({type(error).__name__})."
    names = {
        str(model.get("name", "")).split(":", 1)[0]
        for model in payload.get("models", [])
        if isinstance(model, dict)
    }
    if "embeddinggemma" not in names:
        return False, "Ollama is running, but embeddinggemma is not installed."
    return True, "Ollama and embeddinggemma are ready."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", default=None)
    parser.add_argument("--allow-low-vram", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    expected = normalize_accelerator(args.expect) if args.expect else None
    try:
        info = accelerator_info(expected)
        hardware_error = ""
    except (RuntimeError, ValueError) as error:
        info = None
        hardware_error = str(error)

    blocking: list[str] = []
    warnings: list[str] = []
    if hardware_error:
        blocking.append(hardware_error)
    elif info is not None:
        if expected and info.backend != expected:
            blocking.append(f"Expected {expected}, but PyTorch detected {info.backend}.")
        if info.backend != "cpu" and info.total_memory_bytes is not None:
            if info.total_memory_bytes < MINIMUM_GPU_BYTES:
                message = (
                    f"{info.device_name} exposes {info.total_memory_bytes / 1024**3:.1f} GiB; "
                    "this project targets at least 8 GB VRAM."
                )
                (warnings if args.allow_low_vram else blocking).append(message)

    ollama_ready, ollama_message = ollama_status()
    if not ollama_ready:
        warnings.append(ollama_message)

    payload = {
        "ok": not blocking,
        "hardware": info.as_dict() if info else None,
        "blocking": blocking,
        "warnings": warnings,
        "ollama": ollama_message,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if info:
            memory = (
                f" | {info.total_memory_bytes / 1024**3:.1f} GiB"
                if info.total_memory_bytes is not None
                else ""
            )
            print(f"Accelerator: {backend_label(info.backend)} | {info.device_name}{memory}")
            print(f"PyTorch: {info.torch_version} | runtime {info.runtime_version or 'CPU'}")
        for message in blocking:
            print(f"ERROR: {message}")
        for message in warnings:
            print(f"WARNING: {message}")
        print(ollama_message)
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
