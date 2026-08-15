"""Parent-process coordinator for dual-GPU PD workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from queue import Empty
from time import perf_counter
from typing import Any

from llmserve.pd.process import worker_main


class PDWorkerError(RuntimeError):
    """Raised when a Prefill or Decode worker reports an error."""


@dataclass(slots=True)
class PDConfig:
    model: str
    prefill_gpu: int
    decode_gpu: int
    decode_gpus: tuple[int, ...] = ()
    prefill_enforce_eager: bool = True
    decode_enforce_eager: bool = True
    prefill_init_method: str = "tcp://127.0.0.1:24431"
    decode_init_method: str = "tcp://127.0.0.1:24432"
    decode_init_methods: tuple[str, ...] = ()
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    request_timeout_seconds: float = 120.0
    kv_slot_count: int = 2
    kv_slot_capacity_tokens: int = 1024

    def __post_init__(self):
        decode_gpus = tuple(self.decode_gpus) or (self.decode_gpu,)
        if (
            not all(isinstance(gpu_id, int) for gpu_id in decode_gpus)
            or self.prefill_gpu < 0
            or any(gpu_id < 0 for gpu_id in decode_gpus)
        ):
            raise ValueError("worker GPU ids must be non-negative")
        if self.decode_gpu != decode_gpus[0]:
            raise ValueError("decode_gpu must match the first decode_gpus entry")
        if len(set(decode_gpus)) != len(decode_gpus):
            raise ValueError("decode worker GPUs must be unique")
        if self.prefill_gpu in decode_gpus:
            raise ValueError("Prefill and Decode workers require different GPUs")

        decode_init_methods = tuple(self.decode_init_methods)
        if not decode_init_methods:
            decode_init_methods = self._derive_decode_init_methods(len(decode_gpus))
        if len(decode_init_methods) != len(decode_gpus):
            raise ValueError(
                "decode_init_methods must contain one endpoint per decode worker"
            )
        if len(set((self.prefill_init_method, *decode_init_methods))) != (
            1 + len(decode_init_methods)
        ):
            raise ValueError("worker distributed endpoints must be different")
        self.decode_gpus = decode_gpus
        self.decode_init_methods = decode_init_methods
        self.decode_init_method = decode_init_methods[0]
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if (
            not isinstance(self.kv_slot_count, int)
            or self.kv_slot_count < 0
            or self.kv_slot_count == 1
        ):
            raise ValueError("KV slot count must be zero or at least two")
        if (
            not isinstance(self.kv_slot_capacity_tokens, int)
            or self.kv_slot_capacity_tokens <= 0
        ):
            raise ValueError("KV slot capacity must be a positive integer")

    def _derive_decode_init_methods(self, count: int) -> tuple[str, ...]:
        if count == 1:
            return (self.decode_init_method,)
        prefix, separator, raw_port = self.decode_init_method.rpartition(":")
        if not separator:
            raise ValueError(
                "multiple Decode workers require explicit decode_init_methods"
            )
        try:
            first_port = int(raw_port)
        except ValueError as error:
            raise ValueError(
                "multiple Decode workers require explicit decode_init_methods"
            ) from error
        return tuple(f"{prefix}:{first_port + index}" for index in range(count))

    @property
    def decode_worker_ids(self) -> tuple[str, ...]:
        if len(self.decode_gpus) == 1:
            return ("decode",)
        return tuple(f"decode-{index}" for index in range(len(self.decode_gpus)))

    def worker_specs(self) -> tuple[tuple[str, int, str], ...]:
        return (
            ("prefill", self.prefill_gpu, self.prefill_init_method),
            *tuple(
                zip(
                    self.decode_worker_ids,
                    self.decode_gpus,
                    self.decode_init_methods,
                )
            ),
        )

    def transport_config(self) -> dict[str, int]:
        return {
            "slot_count": self.kv_slot_count,
            "capacity_tokens": self.kv_slot_capacity_tokens,
        }

    def prefill_transport_config(self) -> dict[str, Any]:
        return {
            **self.transport_config(),
            "target_workers": self.decode_worker_ids,
        }

    def engine_kwargs_for(self, role: str) -> dict[str, Any]:
        if role != "prefill" and role not in self.decode_worker_ids:
            raise ValueError(f"unsupported PD worker role: {role}")
        decode_index = (
            self.decode_worker_ids.index(role) if role != "prefill" else None
        )
        kwargs = dict(self.engine_kwargs)
        kwargs.update(
            tensor_parallel_size=1,
            enforce_eager=(
                self.prefill_enforce_eager
                if role == "prefill"
                else self.decode_enforce_eager
            ),
            speculative_model=None,
            enable_speculative_cuda_graph=False,
            distributed_init_method=(
                self.prefill_init_method
                if role == "prefill"
                else self.decode_init_methods[decode_index]
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
        self._transport_handle = None
        self._transport_handles: dict[str, Any] = {}
        self._last_rpc_timing: dict[str, dict[str, float | None]] = {}
        self._failed_roles: set[str] = set()

    def start(self):
        if self._started:
            return
        self._failed_roles.clear()
        try:
            for role, gpu_id, _ in self.config.worker_specs():
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
                        (
                            self.config.prefill_transport_config()
                            if role == "prefill"
                            else self.config.transport_config()
                        ),
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
            for role, _, _ in self.config.worker_specs():
                self._wait_worker_ready(role)
            self._started = True
            prefill_ready = self._workers["prefill"].get("ready_result") or {}
            self._transport_handles = dict(
                prefill_ready.get("kv_slot_handles") or {}
            )
            if not self._transport_handles:
                handle = prefill_ready.get("kv_slot_handle")
                if handle is not None:
                    self._transport_handles = {
                        self.config.decode_worker_ids[0]: handle,
                    }
            self._transport_handle = self._transport_handles.get(
                self.config.decode_worker_ids[0]
            )
            if self.config.kv_slot_count and not self._transport_handles:
                raise PDWorkerError("Prefill Worker did not publish shared KV slots")
            if self._transport_handles:
                self._attach_decode_slot_pools(self._transport_handles)
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
        worker["ready_result"] = result

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
        self._failed_roles.clear()
        self._transport_handles = {}

    def _mark_worker_failed(self, role: str):
        failed_roles = getattr(self, "_failed_roles", None)
        if failed_roles is None:
            self._failed_roles = set()
        self._failed_roles.add(role)

    def _send(self, role: str, command: dict) -> dict[str, Any]:
        if not self._started:
            self.start()
        worker = self._workers[role]
        process = worker.get("process")
        if process is not None and not process.is_alive():
            self._mark_worker_failed(role)
            raise PDWorkerError(
                f"{role} worker is not alive (exitcode={process.exitcode})"
            )
        command = dict(command)
        parent_sent_at = perf_counter()
        command["_rpc_parent_sent_at"] = parent_sent_at
        try:
            worker["commands"].put(command)
        except (EOFError, BrokenPipeError, OSError, ValueError) as error:
            self._mark_worker_failed(role)
            raise PDWorkerError(f"{role} worker command channel failed") from error
        parent_put_at = perf_counter()
        return {
            "parent_sent_at": parent_sent_at,
            "parent_put_at": parent_put_at,
        }

    def _receive(self, role: str, sent: dict[str, float]):
        worker = self._workers[role]
        try:
            response = worker["responses"].get(
                timeout=self.config.request_timeout_seconds
            )
        except Empty as error:
            self._mark_worker_failed(role)
            raise PDWorkerError(f"{role} worker timed out") from error
        except (EOFError, BrokenPipeError, OSError, ValueError) as error:
            self._mark_worker_failed(role)
            raise PDWorkerError(f"{role} worker response channel failed") from error
        parent_received_at = perf_counter()
        parent_sent_at = sent["parent_sent_at"]
        parent_put_at = sent["parent_put_at"]
        remote_timing = response.get("timing") or {}
        worker_received_at = remote_timing.get("worker_received_at")
        worker_reply_enqueued_at = remote_timing.get("worker_reply_enqueued_at")
        timings = getattr(self, "_last_rpc_timing", None)
        if timings is None:
            self._last_rpc_timing = {}
        self._last_rpc_timing[role] = {
            "roundtrip_ms": (parent_received_at - parent_sent_at) * 1000,
            "parent_queue_put_ms": (parent_put_at - parent_sent_at) * 1000,
            "command_queue_ms": (
                (worker_received_at - parent_put_at) * 1000
                if worker_received_at is not None
                else None
            ),
            "worker_service_ms": (
                (worker_reply_enqueued_at - worker_received_at) * 1000
                if worker_received_at is not None
                and worker_reply_enqueued_at is not None
                else None
            ),
            "response_queue_ms": (
                (parent_received_at - worker_reply_enqueued_at) * 1000
                if worker_reply_enqueued_at is not None
                else None
            ),
        }
        if not response.get("ok"):
            self._mark_worker_failed(role)
            error = response.get("error") or {}
            detail = (
                error.get("traceback")
                or error.get("message")
                or "unknown worker error"
            )
            raise PDWorkerError(f"{role} worker failed: {detail}")
        return response.get("result")

    def _call(self, role: str, command: dict):
        return self._receive(role, self._send(role, command))

    def _call_many(self, commands: dict[str, dict]) -> dict[str, Any]:
        """Send every Worker command before awaiting any response."""
        sent = {
            role: self._send(role, command)
            for role, command in commands.items()
        }
        return {
            role: self._receive(role, sent[role])
            for role in commands
        }

    def last_rpc_timing(self, role: str) -> dict[str, float | None]:
        return dict(getattr(self, "_last_rpc_timing", {}).get(role, {}))

    def _decode_role(self, decode_worker_id: str | None = None) -> str:
        role = decode_worker_id or self.config.decode_worker_ids[0]
        if role not in self.config.decode_worker_ids:
            raise ValueError(f"unknown Decode Worker: {role}")
        return role

    @property
    def decode_worker_ids(self) -> tuple[str, ...]:
        return self.config.decode_worker_ids

    def _attach_decode_slot_pools(self, handles: dict[str, Any]):
        expected = set(self.config.decode_worker_ids)
        if set(handles) != expected:
            raise PDWorkerError(
                "Prefill Worker shared KV slot handles do not match Decode Workers"
            )
        for role in self.config.decode_worker_ids:
            self._call(
                role,
                {"type": "attach_shared_slots", "handle": handles[role]},
            )

    def worker_health(self) -> dict[str, dict[str, Any]]:
        """Return process-level health without sending a worker command."""
        health = {}
        decode_roles = tuple(
            role for role in self._workers if role.startswith("decode")
        )
        for role in ("prefill", *decode_roles):
            worker = self._workers.get(role)
            process = worker.get("process") if worker else None
            item = {
                "pid": getattr(process, "pid", None),
                "alive": bool(process is not None and process.is_alive()),
                "exitcode": getattr(process, "exitcode", None),
            }
            environment = worker.get("ready_result", {}).get("environment")
            if environment is not None:
                item["environment"] = environment
            health[role] = item
        return health

    def prefill_batch(self, envelopes, release_transfer_ids=()):
        return self._call(
            "prefill",
            {
                "type": "prefill_batch",
                "envelopes": [envelope.to_payload() for envelope in envelopes],
                "release_transfer_ids": list(release_transfer_ids),
            },
        )

    def admit_batch(self, handoffs, *, decode_worker_id: str | None = None):
        return self._call(
            self._decode_role(decode_worker_id),
            {"type": "admit_batch", "handoffs": handoffs},
        )

    def release_prefill_transfers(self, transfer_ids):
        return self._call(
            "prefill",
            {
                "type": "release_transfers",
                "transfer_ids": list(transfer_ids),
            },
        )

    def decode_step(self, *, decode_worker_id: str | None = None):
        return self._call(self._decode_role(decode_worker_id), {"type": "step"})

    def decode_step_all(
        self,
        decode_worker_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        worker_ids = decode_worker_ids or self.config.decode_worker_ids
        for role in worker_ids:
            self._decode_role(role)
        return self._call_many({
            role: {"type": "step"}
            for role in worker_ids
        })

    def decode_step_with_handoffs(self, handoffs, *, decode_worker_id: str | None = None):
        return self._call(
            self._decode_role(decode_worker_id),
            {"type": "step_with_handoffs", "handoffs": list(handoffs)},
        )

    def decode_step_with_handoffs_all(
        self,
        handoffs_by_worker: dict[str, list],
        *,
        active_worker_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        worker_ids = tuple(dict.fromkeys(
            (*active_worker_ids, *handoffs_by_worker.keys())
        ))
        for role in worker_ids:
            self._decode_role(role)
        return self._call_many({
            role: (
                {
                    "type": "step_with_handoffs",
                    "handoffs": list(handoffs_by_worker[role]),
                }
                if handoffs_by_worker.get(role)
                else {"type": "step"}
            )
            for role in worker_ids
        })

    def collect_completed_transfers(
        self,
        *,
        wait: bool = False,
        decode_worker_id: str | None = None,
    ):
        return self._call(
            self._decode_role(decode_worker_id),
            {"type": "collect_completed_transfers", "wait": bool(wait)},
        )

    def abort_decode_request(
        self,
        seq_id: int,
        *,
        decode_worker_id: str | None = None,
    ) -> bool:
        return bool(
            self._call(
                self._decode_role(decode_worker_id),
                {"type": "abort_request", "seq_id": seq_id},
            )
        )

    def decode_metrics(self, *, decode_worker_id: str | None = None):
        return self._call(
            self._decode_role(decode_worker_id),
            {"type": "metrics"},
        )

    def reset_decode_metrics(self, *, decode_worker_id: str | None = None):
        return self._call(
            self._decode_role(decode_worker_id),
            {"type": "reset_metrics"},
        )

    def decode_metrics_all(self) -> dict[str, Any]:
        return self._call_many({
            role: {"type": "metrics"}
            for role in self.config.decode_worker_ids
        })

    def reset_decode_metrics_all(self):
        return self._call_many({
            role: {"type": "reset_metrics"}
            for role in self.config.decode_worker_ids
        })

    def close(self):
        if not self._started:
            return
        failed_roles = getattr(self, "_failed_roles", set())
        for role in (*self.config.decode_worker_ids, "prefill"):
            worker = self._workers.get(role)
            if worker is None:
                continue
            process = worker.get("process")
            if role in failed_roles or (
                process is not None and not process.is_alive()
            ):
                continue
            try:
                self._call(role, {"type": "shutdown"})
            except (PDWorkerError, EOFError, BrokenPipeError):
                pass
        for worker in self._workers.values():
            process = worker.get("process")
            if process is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            worker["commands"].close()
            worker["responses"].close()
        self._workers.clear()
        self._started = False
        self._transport_handle = None
        self._transport_handles = {}
        self._failed_roles.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
