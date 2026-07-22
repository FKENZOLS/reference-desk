"""Download the public Hugging Face assets needed after installation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_settings import RERANKER_PRESETS, reranker_configuration  # noqa: E402


CHUNK_TOKENIZER = os.environ.get(
    "RAG_CHUNK_TOKENIZER_MODEL",
    "Qwen/Qwen3-Embedding-0.6B",
)
USE_AUTH = os.environ.get("RAG_RERANKER_USE_AUTH", "0") == "1"


def main() -> int:
    token: bool = True if USE_AUTH else False
    print(f"Caching chunk tokenizer: {CHUNK_TOKENIZER}")
    AutoTokenizer.from_pretrained(
        CHUNK_TOKENIZER,
        use_fast=True,
        token=token,
    )

    for choice in RERANKER_PRESETS:
        configuration = reranker_configuration(choice)
        model = str(configuration["model"])
        revision = str(configuration["revision"])
        print(f"Caching reranker: {model}@{revision}")
        snapshot_download(
            repo_id=model,
            revision=revision,
            token=token,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.model",
                "*.py",
                "*.safetensors",
                "*.tiktoken",
                "*.txt",
            ],
        )

    # GTE's official config delegates its architecture to this repository.
    # Cache the code alongside the weights for later offline use.
    trusted_code = {
        (
            str(configuration["code_repository"]),
            str(configuration["code_revision"]),
        )
        for choice in RERANKER_PRESETS
        if (configuration := reranker_configuration(choice))["code_repository"]
    }
    for repository, revision in sorted(trusted_code):
        print(f"Caching trusted reranker code: {repository}@{revision}")
        snapshot_download(
            repo_id=repository,
            revision=revision,
            token=token,
            allow_patterns=["*.json", "*.py"],
        )
    print("Hugging Face model cache is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
