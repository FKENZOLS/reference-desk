"""Crash-isolated reranker inference worker.

The web application owns this lightweight proxy.  PyTorch, the tokenizer, and
the GPU model live in a spawned child process so a poisoned CUDA context can be
discarded without restarting FastAPI or losing browser state.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app_settings import (
    RERANKER_CHOICE,
    RERANKER_USE_AUTH,
    reranker_configuration,
    reranker_fingerprint,
    resolve_reranker_backend,
)
from rag_common import COMPUTE_BACKEND, RERANKER_DEVICE


FATAL_GPU_MARKERS = (
    "device-side assert",
    "index out of bounds",
    "cuda error: an illegal memory access",
    "unspecified launch failure",
)


def is_fatal_gpu_error(error: BaseException | str) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in FATAL_GPU_MARKERS)


class RerankerWorkerError(RuntimeError):
    """A worker request failed, while the FastAPI process remained healthy."""

    def __init__(self, message: str, *, recovered: bool = False) -> None:
        self.recovered = recovered
        super().__init__(message)


def _resolve_worker_device(configured: str) -> torch.device:
    requested = configured.strip().lower()
    value = requested
    if value == "auto":
        value = "cuda:0" if COMPUTE_BACKEND in {"cuda", "rocm"} else "cpu"
    elif value in {"cuda", "nvidia", "rocm", "amd", "hip"}:
        value = "cuda:0"
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU reranking was requested but this PyTorch worker cannot use {value}."
            )
        device = torch.device(value)
        torch.cuda.set_device(device)
        return device
    if value != "cpu":
        raise ValueError(f"Unsupported reranker device: {configured!r}")
    return torch.device("cpu")


class _WorkerModel:
    """The model implementation kept private to the isolated child."""

    def __init__(self, choice: str) -> None:
        configuration = reranker_configuration(choice)
        self.choice = str(configuration["choice"])
        self.model_name = str(configuration["model"])
        self.max_length = int(configuration["max_length"])
        self.batch_size = int(configuration["batch_size"])
        self.device = _resolve_worker_device(RERANKER_DEVICE)
        self.backend = resolve_reranker_backend(
            self.model_name,
            str(configuration["backend"]),
        )
        self.instruction = str(configuration["instruction"])
        self.fingerprint = reranker_fingerprint(self.choice)
        revision = str(configuration["revision"])
        code_revision = str(configuration["code_revision"])
        trust_remote_code = bool(configuration["trust_remote_code"])
        model_token: bool = True if RERANKER_USE_AUTH else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=revision,
            code_revision=code_revision,
            padding_side="right",
            token=model_token,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_dtype = (
            torch.float16
            if self.device.type == "cuda" and bool(configuration["use_fp16"])
            else torch.float32
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            revision=revision,
            code_revision=code_revision,
            dtype=model_dtype,
            token=model_token,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self._repair_gte_position_ids()
        self.model.eval()

    def _repair_gte_position_ids(self) -> None:
        if self.choice != "gte":
            return
        base_model = getattr(self.model, "new", None)
        embeddings = getattr(base_model, "embeddings", None)
        position_ids = getattr(embeddings, "position_ids", None)
        if embeddings is None or not isinstance(position_ids, torch.Tensor):
            return
        max_positions = int(
            getattr(self.model.config, "max_position_embeddings", position_ids.numel())
        )
        embeddings.register_buffer(
            "position_ids",
            torch.arange(max_positions, device=self.device, dtype=torch.long),
            persistent=False,
        )

    def validate(self) -> dict[str, Any]:
        passage = (
            "Technical document validation passage with requirements, tables, "
            "identifiers, dates, and explanatory context. "
        ) * 20
        result = self.predict([("startup reranker validation", passage)])
        if len(result) != 1 or not all(
            torch.isfinite(torch.tensor(value)) for value in result[0][:2]
        ):
            raise RuntimeError("Reranker startup validation returned invalid scores.")
        return {"token_count": int(result[0][2]), "truncated": bool(result[0][3])}

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[float, float, int, bool]]:
        output: list[tuple[float, float, int, bool]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            queries = [query for query, _ in batch]
            passages = [passage for _, passage in batch]
            untruncated = self.tokenizer(
                queries,
                passages,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            token_counts = [len(ids) for ids in untruncated["input_ids"]]
            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits.view(-1).float()
                probabilities = torch.sigmoid(logits)
            for logit, probability, token_count in zip(
                logits.cpu().tolist(),
                probabilities.cpu().tolist(),
                token_counts,
                strict=True,
            ):
                output.append(
                    (
                        float(logit),
                        float(probability),
                        int(token_count),
                        int(token_count) > self.max_length,
                    )
                )
        return output


def _worker_main(connection: Connection, choice: str) -> None:
    """Load one model and serve inference requests until explicitly stopped."""

    try:
        started = perf_counter()
        reranker = _WorkerModel(choice=choice)
        validation = reranker.validate()
        connection.send(
            {
                "ok": True,
                "event": "ready",
                "metadata": {
                    "choice": reranker.choice,
                    "model_name": reranker.model_name,
                    "fingerprint": reranker.fingerprint,
                    "device": str(reranker.device),
                    "load_seconds": perf_counter() - started,
                    "preflight_tokens": int(validation.get("token_count", 0)),
                    "pid": os.getpid(),
                },
            }
        )
    except BaseException as error:
        try:
            connection.send(
                {
                    "ok": False,
                    "event": "load_error",
                    "error": f"{type(error).__name__}: {error}",
                    "fatal": is_fatal_gpu_error(error),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
        finally:
            connection.close()
        return

    while True:
        try:
            request = connection.recv()
        except (EOFError, OSError):
            break
        operation = str(request.get("op") or "")
        if operation == "shutdown":
            try:
                connection.send({"ok": True, "event": "stopped"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            break
        if operation == "ping":
            connection.send({"ok": True, "event": "pong", "pid": os.getpid()})
            continue
        if operation != "predict":
            connection.send(
                {"ok": False, "event": "request_error", "error": "Unknown worker operation."}
            )
            continue
        try:
            pairs = [
                (str(query), str(passage))
                for query, passage in request.get("pairs", [])
            ]
            connection.send(
                {
                    "ok": True,
                    "event": "prediction",
                    "scores": reranker.predict(pairs),
                }
            )
        except BaseException as error:
            fatal = is_fatal_gpu_error(error)
            try:
                connection.send(
                    {
                        "ok": False,
                        "event": "prediction_error",
                        "error": f"{type(error).__name__}: {error}",
                        "fatal": fatal,
                        "traceback": traceback.format_exc(limit=12),
                    }
                )
            except (BrokenPipeError, EOFError, OSError):
                pass
            if fatal:
                # A CUDA device-side assertion poisons the process.  Returning
                # lets the parent replace the entire worker and GPU context.
                break
    connection.close()


class RerankerWorkerClient:
    """Synchronous proxy with automatic child-process recovery."""

    def __init__(
        self,
        choice: str = RERANKER_CHOICE,
        *,
        startup_timeout: float = 600.0,
        inference_timeout: float = 300.0,
    ) -> None:
        configuration = reranker_configuration(choice)
        self.choice = str(configuration["choice"])
        self.model_name = str(configuration["model"])
        self.fingerprint = reranker_fingerprint(self.choice)
        self.device = ""
        self.load_seconds = 0.0
        self.preflight_tokens = 0
        self.pid: int | None = None
        self.restart_count = 0
        self.startup_timeout = float(startup_timeout)
        self.inference_timeout = float(inference_timeout)
        self._process: mp.Process | None = None
        self._connection: Connection | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._start()

    def _start(self) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker_main,
            args=(child, self.choice),
            name=f"reranker-{self.choice}",
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        if not parent.poll(self.startup_timeout):
            self._discard_worker()
            raise RerankerWorkerError(
                f"{self.choice.upper()} worker did not become ready within "
                f"{self.startup_timeout:.0f} seconds."
            )
        try:
            response = parent.recv()
        except (EOFError, OSError) as error:
            self._discard_worker()
            raise RerankerWorkerError(
                f"{self.choice.upper()} worker exited during model loading."
            ) from error
        if not response.get("ok"):
            self._discard_worker()
            raise RerankerWorkerError(str(response.get("error") or "Worker load failed."))
        metadata = dict(response.get("metadata") or {})
        self.choice = str(metadata.get("choice") or self.choice)
        self.model_name = str(metadata.get("model_name") or self.model_name)
        self.fingerprint = str(metadata.get("fingerprint") or self.fingerprint)
        self.device = str(metadata.get("device") or "")
        self.load_seconds = float(metadata.get("load_seconds") or 0.0)
        self.preflight_tokens = int(metadata.get("preflight_tokens") or 0)
        self.pid = int(metadata.get("pid") or process.pid or 0) or None

    def _discard_worker(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        self.pid = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            process.join(timeout=1.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)

    def restart(self) -> None:
        with self._lock:
            if self._closed:
                raise RerankerWorkerError("The reranker worker has been closed.")
            self._discard_worker()
            self.restart_count += 1
            self._start()

    def _recover(self, reason: str) -> RerankerWorkerError:
        try:
            self.restart()
        except BaseException as restart_error:
            return RerankerWorkerError(
                f"{reason} The worker could not be restarted: {restart_error}"
            )
        return RerankerWorkerError(
            f"{reason} The GPU worker restarted successfully; retry the search.",
            recovered=True,
        )

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[float, float, int, bool]]:
        if not pairs:
            return []
        with self._lock:
            if self._closed:
                raise RerankerWorkerError("The reranker worker has been closed.")
            connection = self._connection
            process = self._process
            if connection is None or process is None or not process.is_alive():
                raise self._recover("The reranker worker stopped unexpectedly.")
            try:
                connection.send({"op": "predict", "pairs": list(pairs)})
                if not connection.poll(self.inference_timeout):
                    raise self._recover(
                        f"Reranking exceeded {self.inference_timeout:.0f} seconds."
                    )
                response = connection.recv()
            except RerankerWorkerError:
                raise
            except (BrokenPipeError, EOFError, OSError) as error:
                raise self._recover("The reranker worker connection was lost.") from error
            if not response.get("ok"):
                message = str(response.get("error") or "Reranker inference failed.")
                if response.get("fatal") or not process.is_alive():
                    raise self._recover(message)
                raise RerankerWorkerError(message)
            return [
                (float(logit), float(probability), int(tokens), bool(truncated))
                for logit, probability, tokens, truncated in response.get("scores", [])
            ]

    def status(self) -> dict[str, Any]:
        process = self._process
        return {
            "pid": self.pid,
            "alive": bool(process is not None and process.is_alive()),
            "restart_count": self.restart_count,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            connection = self._connection
            if connection is not None:
                try:
                    connection.send({"op": "shutdown"})
                    if connection.poll(2):
                        connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._discard_worker()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
