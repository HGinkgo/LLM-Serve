"""Parent-process coordinator for dual-GPU PD workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from queue import Empty
from typing import Any

from llmserve.pd.process import worker_main


class PDWorkerError(RuntimeError):
    """Raised when a Prefill or Decode worker reports an error."""


@dataclass(slots=True)
class PDConfig:
    model: str
    prefill_gpu: int
    decode_gpu: int
    prefill_enforce_eager: bool = True
    decode_enforce_eager: bool = True
    prefill_init_method: str = "tcp://127.0.0.1:24431"
    decode_init_method: str = "tcp://127.0.0.1:24432"
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    request_timeout_seconds: float = 120.0

    def __post_init__(self):
        if self.prefill_gpu < 0 or self.decode_gpu < 0:
            raise ValueError("worker GPU ids must be non-negative")
        if self.prefill_gpu == self.decode_gpu:
            raise ValueError("Prefill and Decode workers require different GPUs")
        if self.prefill_init_method == self.decode_init_method:
            raise ValueError("worker distributed endpoints must be different")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")

    def engine_kwargs_for(self, role: str) -> dict[str, Any]:
        if role not in {"prefill", "decode"}:
            raise ValueError(f"unsupported PD worker role: {role}")
        kwargs = dict(self.engine_kwargs)
        kwargs.update(
            tensor_parallel_size=1,
            enforce_eager=(
                self.prefill_enforce_eager
                if role == "prefill"
                else self.decode_enforce_eager
            ),
            speculative_model=None,
            speculative_tree_nodes=0,
            enable_speculative_cuda_graph=False,
            distributed_init_method=(
                self.prefill_init_method
                if role == "prefill"
                else self.decode_init_method
            ),
        )
        return kwargs


class PDCoordinator:
    """Own the Prefill/Decode worker processes and their RPC channels."""

    def __init__(self, config: PDConfig, context=None):
        self.config = config
        self.context = context or mp.get_context("spawn")
        self._workers: dict[str, Any] = {}
        self._started = False

    def start(self):
        if self._started:
            return
        try:
            for role, gpu_id in (
                ("prefill", self.config.prefill_gpu),
                ("decode", self.config.decode_gpu),
            ):
                command_queue = self.context.Queue()
                response_queue = self.context.Queue()
                process = self.context.Process(
                    target=worker_main,
                    args=(
                        role,
                        self.config.model,
                        gpu_id,
                        self.config.engine_kwargs_for(role),
                        command_queue,
                        response_queue,
                    ),
                    name=f"llmserve-{role}-worker",
                )
                process.start()
                self._workers[role] = {
                    "process": process,
                    "commands": command_queue,
                    "responses": response_queue,
                    "ready": False,
                }
            for role in ("prefill", "decode"):
                self._wait_worker_ready(role)
            self._started = True
        except Exception:
            self._abort_startup()
            raise

    def _wait_worker_ready(self, role: str):
        worker = self._workers[role]
        try:
            response = worker["responses"].get(
                timeout=self.config.request_timeout_seconds
            )
        except Empty as error:
            raise PDWorkerError(f"{role} worker did not become ready") from error
        if not response.get("ok"):
            error = response.get("error") or {}
            detail = (
                error.get("traceback")
                or error.get("message")
                or "unknown worker error"
            )
            raise PDWorkerError(f"{role} worker failed during startup: {detail}")
        result = response.get("result") or {}
        if not result.get("ready") or result.get("role") != role:
            raise PDWorkerError(
                f"{role} worker returned an invalid readiness response"
            )
        worker["ready"] = True

    def _abort_startup(self):
        for worker in self._workers.values():
            process = worker.get("process")
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
            for queue_name in ("commands", "responses"):
                queue = worker.get(queue_name)
                if queue is not None:
                    queue.close()
        self._workers.clear()
        self._started = False

    def _call(self, role: str, command: dict):
        if not self._started:
            self.start()
        worker = self._workers[role]
        worker["commands"].put(command)
        try:
            response = worker["responses"].get(
                timeout=self.config.request_timeout_seconds
            )
        except Empty as error:
            raise PDWorkerError(f"{role} worker timed out") from error
        if not response.get("ok"):
            error = response.get("error") or {}
            detail = (
                error.get("traceback")
                or error.get("message")
                or "unknown worker error"
            )
            raise PDWorkerError(f"{role} worker failed: {detail}")
        return response.get("result")

    def worker_health(self) -> dict[str, dict[str, Any]]:
        """Return process-level health without sending a worker command."""
        health = {}
        for role in ("prefill", "decode"):
            worker = self._workers.get(role)
            process = worker.get("process") if worker else None
            health[role] = {
                "pid": getattr(process, "pid", None),
                "alive": bool(process is not None and process.is_alive()),
                "exitcode": getattr(process, "exitcode", None),
            }
        return health

    def prefill_batch(self, envelopes):
        return self._call(
            "prefill",
            {
                "type": "prefill_batch",
                "envelopes": [envelope.to_payload() for envelope in envelopes],
            },
        )

    def admit_batch(self, handoffs):
        return self._call("decode", {"type": "admit_batch", "handoffs": handoffs})

    def decode_step(self):
        return self._call("decode", {"type": "step"})

    def decode_metrics(self):
        return self._call("decode", {"type": "metrics"})

    def reset_decode_metrics(self):
        return self._call("decode", {"type": "reset_metrics"})

    def close(self):
        if not self._started:
            return
        for role in ("prefill", "decode"):
            worker = self._workers.get(role)
            if worker is None:
                continue
            try:
                self._call(role, {"type": "shutdown"})
            except (PDWorkerError, EOFError, BrokenPipeError):
                pass
        for worker in self._workers.values():
            process = worker["process"]
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            worker["commands"].close()
            worker["responses"].close()
        self._workers.clear()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
