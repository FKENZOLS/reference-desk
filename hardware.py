"""Portable PyTorch accelerator detection for NVIDIA CUDA and AMD ROCm."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = PROJECT_DIR / ".rag-profile"
_ALIASES = {
    "amd": "rocm",
    "hip": "rocm",
    "nvidia": "cuda",
    "gpu": "auto",
    "": "auto",
}
VALID_ACCELERATORS = {"auto", "cuda", "rocm", "cpu"}


def _torch_module() -> Any:
    import torch

    return torch


@dataclass(frozen=True)
class AcceleratorInfo:
    requested: str
    backend: str
    torch_device: str
    available: bool
    device_name: str
    torch_version: str
    runtime_version: str
    total_memory_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_accelerator(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        normalized = "cuda"
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in VALID_ACCELERATORS:
        raise ValueError(
            "Accelerator must be auto, cuda/nvidia, rocm/amd, or cpu; "
            f"received {value!r}."
        )
    return normalized


def configured_accelerator(
    environment: Mapping[str, str] | None = None,
    profile_path: Path = PROFILE_PATH,
) -> str:
    values = os.environ if environment is None else environment
    explicit = values.get("RAG_ACCELERATOR") or values.get("RAG_DEVICE")
    if explicit:
        return normalize_accelerator(explicit)
    try:
        saved = profile_path.read_text(encoding="utf-8").strip()
    except OSError:
        saved = "auto"
    return normalize_accelerator(saved)


def detected_accelerator(torch_module: Any = None) -> str:
    torch_module = torch_module or _torch_module()
    if not bool(torch_module.cuda.is_available()):
        return "cpu"
    version = getattr(torch_module, "version", None)
    if getattr(version, "hip", None):
        return "rocm"
    return "cuda"


def runtime_version(backend: str, torch_module: Any = None) -> str:
    torch_module = torch_module or _torch_module()
    version = getattr(torch_module, "version", None)
    if backend == "rocm":
        return str(getattr(version, "hip", None) or "unknown")
    if backend == "cuda":
        return str(getattr(version, "cuda", None) or "unknown")
    return ""


def resolve_accelerator(
    requested: str | None,
    torch_module: Any = None,
) -> str:
    torch_module = torch_module or _torch_module()
    normalized = normalize_accelerator(requested)
    detected = detected_accelerator(torch_module)
    if normalized == "auto":
        return detected
    if normalized == "cpu":
        return "cpu"
    if detected == normalized:
        return normalized

    installed = runtime_version(detected, torch_module) or "CPU-only"
    if normalized == "rocm":
        raise RuntimeError(
            "AMD ROCm was requested, but this PyTorch installation cannot use "
            f"ROCm (detected backend: {detected}, runtime: {installed}). Run the "
            "portable setup script with -Backend rocm."
        )
    raise RuntimeError(
        "NVIDIA CUDA was requested, but this PyTorch installation cannot use "
        f"CUDA (detected backend: {detected}, runtime: {installed}). Run the "
        "portable setup script with -Backend cuda."
    )


def torch_device_for(backend: str, index: int = 0) -> str:
    """Return a PyTorch device; ROCm intentionally uses the cuda device API."""

    normalized = normalize_accelerator(backend)
    if normalized in {"cuda", "rocm"}:
        return f"cuda:{index}"
    return "cpu"


def accelerator_info(
    requested: str | None = None,
    torch_module: Any = None,
) -> AcceleratorInfo:
    torch_module = torch_module or _torch_module()
    configured = normalize_accelerator(requested or configured_accelerator())
    backend = resolve_accelerator(configured, torch_module)
    available = backend != "cpu"
    device_name = "CPU"
    total_memory: int | None = None
    if available:
        device_name = str(torch_module.cuda.get_device_name(0))
        try:
            properties = torch_module.cuda.get_device_properties(0)
            total_memory = int(properties.total_memory)
        except Exception:
            try:
                _, total_memory = torch_module.cuda.mem_get_info(0)
                total_memory = int(total_memory)
            except Exception:
                total_memory = None
    return AcceleratorInfo(
        requested=configured,
        backend=backend,
        torch_device=torch_device_for(backend),
        available=available,
        device_name=device_name,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        runtime_version=runtime_version(backend, torch_module),
        total_memory_bytes=total_memory,
    )


def backend_label(backend: str) -> str:
    return {
        "cuda": "NVIDIA CUDA",
        "rocm": "AMD ROCm",
        "cpu": "CPU",
    }.get(backend, backend.upper())
