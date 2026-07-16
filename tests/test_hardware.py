from types import SimpleNamespace

import pytest

from hardware import (
    accelerator_info,
    configured_accelerator,
    detected_accelerator,
    resolve_accelerator,
    torch_device_for,
)


class FakeCuda:
    def __init__(self, available: bool, name: str = "Test GPU", memory: int = 8 * 1024**3):
        self.available = available
        self.name = name
        self.memory = memory

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, index: int) -> str:
        return self.name

    def get_device_properties(self, index: int):
        return SimpleNamespace(total_memory=self.memory)


def fake_torch(*, available: bool, hip: str | None = None, cuda: str | None = None):
    return SimpleNamespace(
        cuda=FakeCuda(available),
        version=SimpleNamespace(hip=hip, cuda=cuda),
        __version__="2.9.1",
    )


def test_nvidia_and_amd_are_discriminated_by_torch_runtime() -> None:
    nvidia = fake_torch(available=True, cuda="12.6")
    amd = fake_torch(available=True, hip="7.2.1")
    assert detected_accelerator(nvidia) == "cuda"
    assert detected_accelerator(amd) == "rocm"
    assert torch_device_for("rocm") == "cuda:0"


def test_requested_vendor_never_silently_uses_another_backend() -> None:
    nvidia = fake_torch(available=True, cuda="12.6")
    with pytest.raises(RuntimeError, match="AMD ROCm was requested"):
        resolve_accelerator("rocm", nvidia)


def test_saved_profile_is_used_but_environment_has_precedence(tmp_path) -> None:
    profile = tmp_path / ".rag-profile"
    profile.write_text("rocm\n", encoding="utf-8")
    assert configured_accelerator({}, profile) == "rocm"
    assert configured_accelerator({"RAG_ACCELERATOR": "nvidia"}, profile) == "cuda"


def test_accelerator_info_reports_amd_memory() -> None:
    amd = fake_torch(available=True, hip="7.2.1")
    info = accelerator_info("rocm", amd)
    assert info.backend == "rocm"
    assert info.device_name == "Test GPU"
    assert info.total_memory_bytes == 8 * 1024**3
