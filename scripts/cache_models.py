"""Download the public Hugging Face assets needed after installation."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


CHUNK_TOKENIZER = os.environ.get(
    "RAG_CHUNK_TOKENIZER_MODEL",
    "Qwen/Qwen3-Embedding-0.6B",
)
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL",
    "Qwen/Qwen3-Reranker-0.6B",
)
RERANKER_REVISION = os.environ.get("RAG_RERANKER_REVISION", "main")
USE_AUTH = os.environ.get("RAG_RERANKER_USE_AUTH", "0") == "1"


def main() -> int:
    token: bool = True if USE_AUTH else False
    print(f"Caching chunk tokenizer: {CHUNK_TOKENIZER}")
    AutoTokenizer.from_pretrained(
        CHUNK_TOKENIZER,
        use_fast=True,
        token=token,
    )

    print(f"Caching reranker: {RERANKER_MODEL}@{RERANKER_REVISION}")
    snapshot_download(
        repo_id=RERANKER_MODEL,
        revision=RERANKER_REVISION,
        token=token,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.model",
            "*.safetensors",
            "*.tiktoken",
            "*.txt",
        ],
    )
    print("Hugging Face model cache is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
