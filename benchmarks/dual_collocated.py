"""Benchmark-only frontend for two independent Collocated LLM replicas."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import multiprocessing as mp
import os
from queue import Empty
import signal
from time import perf_counter
import traceback
from typing import Any


class DualCollocatedWorkerError(RuntimeError):
    """Raised when a complete collocated replica cannot serve an RPC."""


def _handle_termination(signum, _frame):
    raise SystemExit(128 + signum)


def _destroy_process_group():
    import torch.distributed as distributed

    if distributed.is_initialized():
        distributed.destroy_process_group()


def _reply(response_queue, *, result=None, error=None):
    response_queue.put({"ok": error is None, "result": result, "error": error})


def collocated_worker_main(
    model: str,
    gpu_id: int,
    engine_kwargs: dict,
    command_queue,
    response_queue,
):
    """Run one self-contained model replica without PD or KV transport."""
    signal.signal(signal.SIGTERM, _handle_termination)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    engine = None
    try:
        from llmserve import LLM, SamplingParams

        engine = LLM(model, **engine_kwargs)
        _reply(response_queue, result={"ready": True, "gpu_id": gpu_id})
        while True:
            command = command_queue.get()
            command_type = command.get("type")
            if command_type == "shutdown":
                _reply(response_queue, result={"stopped": True})
                return
            if command_type == "add_request":
                seq_id = engine.add_request(
                    command["prompt_token_ids"],
                    SamplingParams(**command["sampling_params"]),
                )
                submitted_at = command.get("submitted_at")
                if submitted_at is not None:
                    engine.record_benchmark_submit(seq_id, submitted_at)
                _reply(response_queue, result={"seq_id": seq_id})
                continue
            if command_type == "record_benchmark_submit":
                engine.record_benchmark_submit(
                    command["seq_id"], command["submitted_at"]
                )
                _reply(response_queue, result={"recorded": True})
                continue
            if command_type == "step":
                outputs, num_tokens = engine.step()
                _reply(
                    response_queue,
                    result={
                        "outputs": outputs,
                        "num_tokens": num_tokens,
                        "last_step_events": engine.last_step_events,
                    },
                )
                continue
            if command_type == "metrics":
                _reply(response_queue, result=engine.get_metrics())
                continue
            if command_type == "reset_metrics":
                engine.reset_metrics()
                _reply(response_queue, result={"reset": True})
                continue
            raise ValueError(f"unsupported collocated worker command: {command_type}")
    except Exception as error:
        _reply(
            response_queue,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            _destroy_process_group()


@dataclass(slots=True)
class DualCollocatedConfig:
    model: str
    gpu_ids: tuple[int, int]
    init_methods: tuple[str, str]
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    request_timeout_seconds: float = 120.0
    startup_timeout_seconds: float = 300.0

    def __post_init__(self):
        self.gpu_ids = tuple(self.gpu_ids)
        self.init_methods = tuple(self.init_methods)
        if len(self.gpu_ids) != 2 or len(set(self.gpu_ids)) != 2:
            raise ValueError("dual collocated runtime requires two unique GPUs")
        if any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("replica GPU ids must be non-negative integers")
        if len(self.init_methods) != 2 or len(set(self.init_methods)) != 2:
            raise ValueError("dual collocated runtime requires two unique endpoints")
        if self.request_timeout_seconds <= 0 or self.startup_timeout_seconds <= 0:
            raise ValueError("worker timeouts must be positive")

    @property
    def worker_ids(self) -> tuple[str, str]:
        return ("replica-0", "replica-1")

    def engine_kwargs_for(self, worker_id: str) -> dict[str, Any]:
        try:
            index = self.worker_ids.index(worker_id)
        except ValueError as error:
            raise ValueError(f"unknown replica: {worker_id}") from error
        kwargs = dict(self.engine_kwargs)
        kwargs.update(
            tensor_parallel_size=1,
            speculative_model=None,
            enable_speculative_cuda_graph=False,
            distributed_init_method=self.init_methods[index],
        )
        return kwargs


class DualCollocatedCoordinator:
    """Own two complete, independent collocated Runtime worker processes."""

    def __init__(self, config: DualCollocatedConfig, context=None):
        self.config = config
        self.context = context or mp.get_context("spawn")
        self._workers: dict[str, dict[str, Any]] = {}
        self._started = False
        self._failed_workers: set[str] = set()

    @property
    def worker_ids(self):
        return self.config.worker_ids

    def start(self):
        if self._started:
            return
        try:
            for worker_id, gpu_id in zip(self.worker_ids, self.config.gpu_ids):
                commands = self.context.Queue()
                responses = self.context.Queue()
                process = self.context.Process(
                    target=collocated_worker_main,
                    args=(
                        self.config.model,
                        gpu_id,
                        self.config.engine_kwargs_for(worker_id),
                        commands,
                        responses,
                    ),
                    name=f"llmserve-{worker_id}-worker",
                )
                process.start()
                self._workers[worker_id] = {
                    "process": process,
                    "commands": commands,
                    "responses": responses,
                }
            for worker_id in self.worker_ids:
                self._wait_ready(worker_id)
            self._started = True
        except Exception:
            self._abort_startup()
            raise

    def _wait_ready(self, worker_id: str):
        worker = self._workers[worker_id]
        try:
            response = worker["responses"].get(
                timeout=self.config.startup_timeout_seconds
            )
        except Empty as error:
            process = worker["process"]
            raise DualCollocatedWorkerError(
                f"{worker_id} did not become ready within "
                f"{self.config.startup_timeout_seconds:.1f}s "
                f"(pid={process.pid}, alive={process.is_alive()}, "
                f"exitcode={process.exitcode})"
            ) from error
        self._unwrap_response(worker_id, response, startup=True)
        result = response.get("result") or {}
        if not result.get("ready"):
            raise DualCollocatedWorkerError(f"{worker_id} returned invalid readiness")
        worker["ready_result"] = result

    def _unwrap_response(self, worker_id: str, response: dict, *, startup=False):
        if response.get("ok"):
            return response.get("result")
        self._failed_workers.add(worker_id)
        error = response.get("error") or {}
        detail = error.get("traceback") or error.get("message") or "unknown error"
        phase = "during startup" if startup else "while serving"
        raise DualCollocatedWorkerError(f"{worker_id} failed {phase}: {detail}")

    def _send(self, worker_id: str, command: dict):
        if not self._started:
            self.start()
        worker = self._workers[worker_id]
        process = worker["process"]
        if not process.is_alive():
            self._failed_workers.add(worker_id)
            raise DualCollocatedWorkerError(
                f"{worker_id} is not alive (exitcode={process.exitcode})"
            )
        try:
            worker["commands"].put(command)
        except (EOFError, BrokenPipeError, OSError, ValueError) as error:
            self._failed_workers.add(worker_id)
            raise DualCollocatedWorkerError(
                f"{worker_id} command channel failed"
            ) from error

    def _receive(self, worker_id: str):
        worker = self._workers[worker_id]
        try:
            response = worker["responses"].get(
                timeout=self.config.request_timeout_seconds
            )
        except Empty as error:
            self._failed_workers.add(worker_id)
            raise DualCollocatedWorkerError(f"{worker_id} timed out") from error
        except (EOFError, BrokenPipeError, OSError, ValueError) as error:
            self._failed_workers.add(worker_id)
            raise DualCollocatedWorkerError(
                f"{worker_id} response channel failed"
            ) from error
        return self._unwrap_response(worker_id, response)

    def _call(self, worker_id: str, command: dict):
        self._send(worker_id, command)
        return self._receive(worker_id)

    def _call_many(self, commands: dict[str, dict]):
        for worker_id, command in commands.items():
            self._send(worker_id, command)
        return {
            worker_id: self._receive(worker_id)
            for worker_id in commands
        }

    @staticmethod
    def _sampling_payload(sampling_params):
        return {
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "ignore_eos": sampling_params.ignore_eos,
        }

    def add_request(
        self,
        worker_id: str,
        prompt_token_ids,
        sampling_params,
        submitted_at,
    ) -> int:
        result = self._call(
            worker_id,
            {
                "type": "add_request",
                "prompt_token_ids": list(prompt_token_ids),
                "sampling_params": self._sampling_payload(sampling_params),
                "submitted_at": submitted_at,
            },
        )
        return int(result["seq_id"])

    def record_benchmark_submit(self, worker_id: str, seq_id: int, submitted_at: float):
        self._call(
            worker_id,
            {
                "type": "record_benchmark_submit",
                "seq_id": seq_id,
                "submitted_at": submitted_at,
            },
        )

    def step_all(self, worker_ids) -> dict[str, dict]:
        return self._call_many({worker_id: {"type": "step"} for worker_id in worker_ids})

    def metrics_all(self) -> dict[str, dict]:
        return self._call_many({worker_id: {"type": "metrics"} for worker_id in self.worker_ids})

    def reset_metrics_all(self):
        return self._call_many({worker_id: {"type": "reset_metrics"} for worker_id in self.worker_ids})

    def worker_health(self) -> dict[str, dict[str, Any]]:
        health = {}
        for worker_id in self.worker_ids:
            worker = self._workers.get(worker_id)
            process = worker.get("process") if worker else None
            health[worker_id] = {
                "gpu_id": self.config.gpu_ids[self.worker_ids.index(worker_id)],
                "pid": getattr(process, "pid", None),
                "alive": bool(process is not None and process.is_alive()),
                "exitcode": getattr(process, "exitcode", None),
            }
        return health

    def _abort_startup(self):
        for worker in self._workers.values():
            process = worker["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            worker["commands"].close()
            worker["responses"].close()
        self._workers.clear()
        self._started = False

    def close(self):
        for worker_id, worker in tuple(self._workers.items()):
            process = worker["process"]
            if process.is_alive() and worker_id not in self._failed_workers:
                try:
                    self._call(worker_id, {"type": "shutdown"})
                except DualCollocatedWorkerError:
                    pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            worker["commands"].close()
            worker["responses"].close()
        self._workers.clear()
        self._started = False
        self._failed_workers.clear()


class DualCollocatedServingEngine:
    """Route whole requests across complete replicas without P/D handoff."""

    is_dual_collocated = True

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.worker_ids = tuple(coordinator.worker_ids)
        if len(self.worker_ids) != 2 or len(set(self.worker_ids)) != 2:
            raise ValueError("dual collocated serving requires exactly two workers")
        self._stripe_start_index = 0
        self._stripe_offset = 0
        self._next_request_id = 0
        self._requests = {}
        self._local_to_global = {}
        self._active_request_ids = set()
        self.last_step_events = {}

    def _add_request(self, prompt_token_ids, sampling_params, submitted_at):
        request_id = self._next_request_id
        self._next_request_id += 1
        # Rotate the leading replica for each two-request stripe. This stays
        # class-agnostic while avoiding a fixed short/long trace alternating
        # onto different GPUs under ordinary one-request round robin.
        worker_id = self.worker_ids[
            (self._stripe_start_index + self._stripe_offset) % len(self.worker_ids)
        ]
        self._stripe_offset += 1
        if self._stripe_offset == len(self.worker_ids):
            self._stripe_offset = 0
            self._stripe_start_index = (
                self._stripe_start_index + 1
            ) % len(self.worker_ids)
        local_seq_id = self.coordinator.add_request(
            worker_id,
            prompt_token_ids,
            sampling_params,
            submitted_at,
        )
        self._requests[request_id] = {
            "worker_id": worker_id,
            "local_seq_id": local_seq_id,
        }
        self._local_to_global[(worker_id, local_seq_id)] = request_id
        self._active_request_ids.add(request_id)
        return request_id

    def add_request(self, prompt_token_ids, sampling_params):
        return self._add_request(prompt_token_ids, sampling_params, None)

    def add_benchmark_request(self, prompt_token_ids, sampling_params, submitted_at):
        return self._add_request(prompt_token_ids, sampling_params, submitted_at)

    def record_benchmark_submit(self, request_id, submitted_at):
        request = self._requests[request_id]
        self.coordinator.record_benchmark_submit(
            request["worker_id"], request["local_seq_id"], submitted_at
        )

    def request_assignments(self):
        return {
            request_id: request["worker_id"]
            for request_id, request in sorted(self._requests.items())
        }

    def _global_request_id(self, worker_id, local_seq_id):
        try:
            return self._local_to_global[(worker_id, local_seq_id)]
        except KeyError as error:
            raise RuntimeError(
                f"unknown {worker_id} sequence {local_seq_id}"
            ) from error

    def _active_workers(self):
        return tuple(
            worker_id
            for worker_id in self.worker_ids
            if any(
                request["worker_id"] == worker_id
                for request_id, request in self._requests.items()
                if request_id in self._active_request_ids
            )
        )

    def _merge_step_events(self, responses):
        sequence_fields = (
            "scheduled_seq_ids",
            "prefill_seq_ids",
            "decode_seq_ids",
            "first_token_seq_ids",
            "finished_seq_ids",
            "partial_prefill_seq_ids",
        )
        summed_fields = (
            "waiting_queue_size",
            "running_queue_size",
            "prefill_token_count",
            "decode_token_count",
            "prefill_request_count",
            "decode_request_count",
            "num_tokens",
            "partial_prefill_chunk_count",
        )
        merged = {"replica_events": {}, "replica_queue_state": {}}
        starts = []
        ends = []
        for worker_id, response in responses.items():
            events = deepcopy(response.get("last_step_events") or {})
            merged["replica_events"][worker_id] = events
            merged["replica_queue_state"][worker_id] = {
                "running_queue_size": events.get("running_queue_size", 0),
                "decode_request_count": len(events.get("decode_seq_ids", ())),
            }
            if events.get("step_start") is not None:
                starts.append(events["step_start"])
            if events.get("step_end") is not None:
                ends.append(events["step_end"])
            for field in sequence_fields:
                if field not in events:
                    continue
                merged.setdefault(field, []).extend(
                    self._global_request_id(worker_id, local_seq_id)
                    for local_seq_id in events[field]
                )
            for field in summed_fields:
                if events.get(field) is not None:
                    merged[field] = merged.get(field, 0) + events[field]
        for worker_id in self.worker_ids:
            merged["replica_queue_state"].setdefault(
                worker_id,
                {"running_queue_size": 0, "decode_request_count": 0},
            )
        if starts:
            merged["step_start"] = min(starts)
        if ends:
            merged["step_end"] = max(ends)
        return merged

    def step(self):
        worker_ids = self._active_workers()
        if not worker_ids:
            self.last_step_events = {}
            return [], 0
        responses = self.coordinator.step_all(worker_ids)
        outputs = []
        num_tokens = 0
        for worker_id in worker_ids:
            response = responses[worker_id]
            num_tokens += int(response.get("num_tokens", 0))
            for local_seq_id, token_ids in response.get("outputs", ()):
                request_id = self._global_request_id(worker_id, local_seq_id)
                outputs.append((request_id, token_ids))
                self._active_request_ids.discard(request_id)
        self.last_step_events = self._merge_step_events(responses)
        return outputs, num_tokens

    def is_finished(self):
        return not self._active_request_ids

    def reset_metrics(self):
        if not self.is_finished():
            raise RuntimeError("cannot reset metrics while requests are active")
        self.coordinator.reset_metrics_all()
        self._requests.clear()
        self._local_to_global.clear()
        self.last_step_events = {}

    def get_metrics(self):
        worker_metrics = self.coordinator.metrics_all()
        requests = []
        replicas = {}
        graph_totals = {"captured_graphs": 0, "replays": 0}
        graph_enabled = True
        for worker_id in self.worker_ids:
            metrics = worker_metrics[worker_id]
            replica_requests = metrics.get("requests", ())
            summary = deepcopy(metrics.get("summary", {}))
            replicas[worker_id] = {
                "request_count": len(replica_requests),
                "summary": summary,
            }
            graph = summary.get("cuda_graph", {})
            graph_enabled = graph_enabled and bool(graph.get("enabled", False))
            for name in graph_totals:
                graph_totals[name] += int(graph.get(name, 0) or 0)
            for request in replica_requests:
                request = deepcopy(request)
                request["seq_id"] = self._global_request_id(
                    worker_id, request["seq_id"]
                )
                request["replica_id"] = worker_id
                requests.append(request)
        assignments = Counter(self.request_assignments().values())
        return {
            "summary": {
                "replicas": replicas,
                "routing": {
                    "policy": "two_request_striped_round_robin",
                    "assigned_requests": {
                        worker_id: assignments[worker_id]
                        for worker_id in self.worker_ids
                    },
                },
                "worker_health": self.coordinator.worker_health(),
                "cuda_graph": {
                    "enabled": graph_enabled,
                    **graph_totals,
                    "replicas": {
                        worker_id: replicas[worker_id]["summary"].get(
                            "cuda_graph", {}
                        )
                        for worker_id in self.worker_ids
                    },
                },
            },
            "requests": sorted(requests, key=lambda request: request["seq_id"]),
        }

    def exit(self):
        self.coordinator.close()
