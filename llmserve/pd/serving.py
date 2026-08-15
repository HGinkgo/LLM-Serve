"""Continuous-batching adapter for dual-worker PD serving."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter

from llmserve.pd.observability import DecodeIdleReason
from llmserve.pd.protocol import RequestEnvelope, RequestLifecycle, RequestState


class PDServingEngine:
    """Expose a step-wise engine interface over Prefill/Decode workers.

    The adapter keeps the Prefill command asynchronous, while Decode remains
    the clock for one benchmark step. This preserves iteration-level batching
    and allows the existing serving benchmark to consume PD without treating
    a whole request batch as one opaque timing sample.
    """

    def __init__(
        self,
        coordinator,
        prefill_batch_size: int = 1,
        enable_latency_telemetry: bool = False,
        enable_transport_overlap: bool = False,
    ):
        if not isinstance(prefill_batch_size, int) or prefill_batch_size <= 0:
            raise ValueError("prefill_batch_size must be a positive integer")
        self.coordinator = coordinator
        self.prefill_batch_size = prefill_batch_size
        self.enable_latency_telemetry = enable_latency_telemetry
        self.enable_transport_overlap = enable_transport_overlap
        self._next_request_id = 0
        self._pending: deque[RequestEnvelope] = deque()
        self._requests: dict[int, dict] = {}
        self._active_by_decode_seq: dict[int, int] = {}
        self._prefill_future: Future | None = None
        self._prefill_future_meta: dict | None = None
        self._pending_handoff_batch: dict | None = None
        self._pending_transfer_acks: list[str] = []
        self._awaiting_transfer_completions: dict[str, int] = {}
        self._slot_release_samples: list[dict] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False
        self._fatal_error: str | None = None
        self._failure_worker_health: dict | None = None
        self.last_step_events: dict = {}
        self._prefill_batches: list[dict] = []
        self._queue_samples: list[dict] = []
        self._decode_idle_intervals: list[dict] = []
        self._run_started_at = perf_counter()
        self.coordinator.start()

    def add_request(self, prompt: str | list[int], sampling_params) -> int:
        if self._fatal_error is not None:
            raise RuntimeError(f"PD serving engine failed: {self._fatal_error}")
        if self._closed:
            raise RuntimeError("PD serving engine is closed")
        if isinstance(prompt, str):
            raise TypeError("PDServingEngine requires tokenized prompts")
        request_id = self._next_request_id
        self._next_request_id += 1
        token_ids = tuple(prompt)
        envelope = RequestEnvelope(
            request_id=request_id,
            prompt_token_ids=token_ids,
            max_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            ignore_eos=sampling_params.ignore_eos,
        )
        self._pending.append(envelope)
        self._requests[request_id] = {
            "request_id": request_id,
            "arrival_time": perf_counter(),
            "prompt_tokens": len(token_ids),
            "decode_seq_id": None,
            "finish_time": None,
            "lifecycle": RequestLifecycle(request_id),
            "timeline": None,
        }
        return request_id

    def record_benchmark_submit(self, request_id: int, submitted_at: float):
        if not self.enable_latency_telemetry:
            return
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(f"unknown request {request_id}")
        request["arrival_time"] = submitted_at
        request["timeline"] = {
            "t_submit": submitted_at,
            "writers": {"t_submit": "benchmark_harness"},
        }

    def abort_request(self, request_id: int) -> bool:
        """Cancel a PD request at a boundary between serving steps."""
        request = self._requests.get(request_id)
        if request is None:
            return False
        lifecycle = request["lifecycle"]
        if lifecycle.is_terminal:
            return False

        if lifecycle.state == RequestState.QUEUED:
            self._pending = deque(
                envelope
                for envelope in self._pending
                if envelope.request_id != request_id
            )
        elif lifecycle.state == RequestState.DECODING:
            decode_seq_id = request.get("decode_seq_id")
            if decode_seq_id is None:
                return False
            if not self.coordinator.abort_decode_request(decode_seq_id):
                return False
            self._active_by_decode_seq.pop(decode_seq_id, None)
        elif lifecycle.state == RequestState.HANDOFF:
            batch = self._pending_handoff_batch
            if batch is None:
                return False
            retained_handoffs = []
            removed = False
            for handoff in batch["handoffs"]:
                if handoff.request_id != request_id:
                    retained_handoffs.append(handoff)
                    continue
                removed = True
                descriptor = getattr(handoff, "descriptor", None)
                if getattr(descriptor, "transport", "inline") == "shared_slot":
                    self._pending_transfer_acks.append(descriptor.transfer_id)
            if not removed:
                return False
            if retained_handoffs:
                batch["handoffs"] = retained_handoffs
            else:
                self._pending_handoff_batch = None

        lifecycle.cancel()
        request["finish_time"] = perf_counter()
        return True

    def _start_prefill(self):
        if self._prefill_future is not None or not self._pending:
            return
        batch = []
        while self._pending and len(batch) < self.prefill_batch_size:
            envelope = self._pending.popleft()
            self._requests[envelope.request_id]["lifecycle"].transition(
                RequestState.PREFILLING
            )
            batch.append(envelope)
        submitted_at = perf_counter()
        release_transfer_ids = self._pending_transfer_acks
        self._pending_transfer_acks = []
        self._prefill_future = self._executor.submit(
            self.coordinator.prefill_batch,
            batch,
            release_transfer_ids=release_transfer_ids,
        )
        self._prefill_future_meta = {
            "request_ids": [envelope.request_id for envelope in batch],
            "batch_size": len(batch),
            "submitted_at": submitted_at,
            "released_transfer_ids": list(release_transfer_ids),
            "t_prefill_dispatch_parent": submitted_at,
        }

    def _flush_transfer_acks(self):
        if self._prefill_future is not None or not self._pending_transfer_acks:
            return
        transfer_ids = self._pending_transfer_acks
        stats = self.coordinator.release_prefill_transfers(transfer_ids)
        self._pending_transfer_acks = []
        self._slot_release_samples.append(
            {
                "transfer_ids": list(transfer_ids),
                "slot_stats": deepcopy(stats),
            }
        )

    def _record_completed_transfers(self, completions):
        for completion in completions:
            transfer_id = completion.get("transfer_id")
            request_id = self._awaiting_transfer_completions.pop(transfer_id, None)
            if request_id is None:
                raise RuntimeError("Decode Worker completed an unknown shared transfer")
            self._pending_transfer_acks.append(transfer_id)
            if not self.enable_latency_telemetry:
                continue
            request = self._requests.get(request_id)
            if request is None:
                raise RuntimeError("completed shared transfer has no request")
            timeline = request.get("timeline") or {}
            request["timeline"] = timeline
            telemetry = completion.get("telemetry") or {}
            timeline.update({
                key: telemetry.get(key)
                for key in (
                    "t_kv_import_completion_observed",
                    "copy_gpu_ms",
                    "decode_gpu_step_ms",
                    "copy_compute_overlap_ms",
                    "copy_compute_overlap_ratio",
                    "serial_gpu_ms",
                    "overlapped_makespan_gpu_ms",
                    "critical_path_reduction_gpu_ms",
                )
                if telemetry.get(key) is not None
            })
            if telemetry.get("kv_import_gpu_ms") is not None:
                timeline["kv_import_gpu_ms"] = telemetry["kv_import_gpu_ms"]
            timeline.setdefault("writers", {}).update(
                telemetry.get("writers") or {}
            )

    def _collect_completed_transfer_acks(self, *, wait: bool = False):
        collector = getattr(self.coordinator, "collect_completed_transfers", None)
        if collector is not None:
            self._record_completed_transfers(collector(wait=wait))

    def _collect_prefill(self):
        future = self._prefill_future
        if future is None or not future.done():
            return False
        meta = self._prefill_future_meta or {}
        handoffs = list(future.result())
        finished_at = perf_counter()
        prefill_rpc_timing = self.coordinator.last_rpc_timing("prefill")
        if len(handoffs) != meta.get("batch_size"):
            raise RuntimeError("Prefill Worker returned an incomplete batch")
        self._prefill_future = None
        self._prefill_future_meta = None
        self._start_prefill()
        admitted_handoffs = []
        for handoff in handoffs:
            request = self._requests.get(handoff.request_id)
            if request is None:
                raise RuntimeError("Prefill Worker returned an unknown request")
            lifecycle = request["lifecycle"]
            if lifecycle.state == RequestState.CANCELLED:
                descriptor = getattr(handoff, "descriptor", None)
                if getattr(descriptor, "transport", "inline") == "shared_slot":
                    self._pending_transfer_acks.append(descriptor.transfer_id)
                continue
            lifecycle.transition(RequestState.HANDOFF)
            admitted_handoffs.append(handoff)
        if not admitted_handoffs:
            if self._prefill_future is None and not self._pending:
                self._flush_transfer_acks()
            return True
        if self._pending_handoff_batch is not None:
            raise RuntimeError("PD has an unadmitted handoff batch")
        self._pending_handoff_batch = {
            "handoffs": admitted_handoffs,
            "meta": meta,
            "finished_at": finished_at,
            "prefill_rpc_timing": prefill_rpc_timing,
        }
        return True

    def _apply_handoff_admissions(
        self,
        batch: dict,
        admissions: list[dict],
        *,
        admit_started_at: float,
        admit_finished_at: float,
        decode_rpc_timing: dict,
        admission_timing: dict | None = None,
        combined_step_roundtrip_ms: float | None = None,
    ):
        handoffs = batch["handoffs"]
        meta = batch["meta"]
        finished_at = batch["finished_at"]
        prefill_rpc_timing = batch["prefill_rpc_timing"]
        if len(admissions) != len(handoffs):
            raise RuntimeError("Decode Worker returned an incomplete admission batch")
        for handoff, admission in zip(handoffs, admissions):
            request_id = handoff.request_id
            if request_id not in self._requests:
                raise RuntimeError("Decode Worker admitted an unknown request")
            request = self._requests[request_id]
            lifecycle = request["lifecycle"]
            lifecycle.transition(RequestState.DECODING)
            if self.enable_latency_telemetry:
                timeline = request.get("timeline") or {}
                request["timeline"] = timeline
                handoff_telemetry = getattr(handoff, "telemetry", {})
                timeline.update({
                    key: handoff_telemetry.get(key)
                    for key in (
                        "t_slot_acquired",
                        "t_kv_export_started",
                        "t_kv_export_finished",
                        "t_slot_ready",
                        "t_prefill_first_scheduled",
                        "t_prefill_finish",
                        "t_first_token",
                        "first_token_origin",
                    )
                    if handoff_telemetry.get(key) is not None
                })
                timeline["t_prefill_dispatch_parent"] = meta.get(
                    "t_prefill_dispatch_parent"
                )
                timeline.update({
                    key: admission.get("telemetry", {}).get(key)
                    for key in (
                        "t_decode_descriptor_received",
                        "t_slot_consuming",
                        "t_decode_admission_started",
                        "t_decode_admitted",
                        "t_handoff_finish",
                        "t_decode_admission_returned",
                        "t_kv_import_enqueued",
                    )
                    if admission.get("telemetry", {}).get(key) is not None
                })
                writers = timeline.setdefault("writers", {})
                writers.update(handoff_telemetry.get("writers", {}))
                writers.update(admission.get("telemetry", {}).get("writers", {}))
            descriptor = getattr(handoff, "descriptor", None)
            if getattr(descriptor, "transport", "inline") == "shared_slot":
                transfer_id = admission.get("transfer_id")
                if transfer_id != descriptor.transfer_id:
                    raise RuntimeError("Decode Worker returned an invalid transfer ACK")
                self._awaiting_transfer_completions[transfer_id] = request_id
            if admission.get("finished"):
                request["finished_without_step"] = list(
                    admission.get("output_token_ids") or ()
                )
                request["finish_time"] = admit_finished_at
                lifecycle.transition(RequestState.FINISHED)
                continue
            decode_seq_id = admission.get("seq_id")
            if decode_seq_id in self._active_by_decode_seq:
                raise RuntimeError("Decode Worker returned duplicate sequence ids")
            self._active_by_decode_seq[decode_seq_id] = request_id
            request["decode_seq_id"] = decode_seq_id
        prefill_timing = getattr(handoffs[0], "prefill_timing_ms", {})
        worker_total_ms = prefill_timing.get("worker_total_ms")
        admission_timing = admission_timing or {}
        actual_admit_started_at = admission_timing.get(
            "started_at", admit_started_at
        )
        actual_admit_finished_at = admission_timing.get(
            "finished_at", admit_finished_at
        )
        self._prefill_batches.append(
            {
                "request_ids": list(meta.get("request_ids", ())),
                "batch_size": meta.get("batch_size", 0),
                "prefill_roundtrip_ms": (
                    finished_at - meta.get("submitted_at", finished_at)
                ) * 1000,
                "admit_roundtrip_ms": (
                    actual_admit_finished_at - actual_admit_started_at
                ) * 1000,
                "handoff_path_ms": (
                    actual_admit_finished_at - finished_at
                ) * 1000,
                "combined_step_roundtrip_ms": combined_step_roundtrip_ms,
                "prefill_worker_ms": worker_total_ms,
                "prefill_model_forward_ms": prefill_timing.get(
                    "model_forward_ms"
                ),
                "prefill_model_forward_gpu_ms": prefill_timing.get(
                    "model_forward_gpu_ms"
                ),
                "prefill_kv_export_copy_ms": prefill_timing.get(
                    "kv_export_copy_ms"
                ),
                "prefill_forward_calls": prefill_timing.get("forward_calls"),
                "prefill_parent_overhead_ms": (
                    (finished_at - meta.get("submitted_at", finished_at)) * 1000
                    - worker_total_ms
                    if worker_total_ms is not None
                    else None
                ),
                "prefill_parent_queue_put_ms": prefill_rpc_timing.get(
                    "parent_queue_put_ms"
                ),
                "prefill_command_queue_ms": prefill_rpc_timing.get(
                    "command_queue_ms"
                ),
                "prefill_response_queue_ms": prefill_rpc_timing.get(
                    "response_queue_ms"
                ),
                "decode_parent_queue_put_ms": decode_rpc_timing.get(
                    "parent_queue_put_ms"
                ),
                "decode_command_queue_ms": decode_rpc_timing.get(
                    "command_queue_ms"
                ),
                "decode_worker_admit_ms": admission_timing.get(
                    "wall_ms", decode_rpc_timing.get("worker_service_ms")
                ),
                "decode_response_queue_ms": decode_rpc_timing.get(
                    "response_queue_ms"
                ),
                "shared_slot_id": (
                    getattr(getattr(handoffs[0], "descriptor", None), "slot_id", None)
                    if handoffs
                    else None
                ),
                "shared_slot_generation": (
                    getattr(
                        getattr(handoffs[0], "descriptor", None),
                        "slot_generation",
                        None,
                    )
                    if handoffs
                    else None
                ),
                "transports": [
                    getattr(getattr(handoff, "descriptor", None), "transport", None)
                    for handoff in handoffs
                ],
                "slot_stats_after_ready": getattr(
                    handoffs[0], "telemetry", {}
                ).get("slot_stats_after_ready") if handoffs else None,
                "slot_observability_after_ready": getattr(
                    handoffs[0], "telemetry", {}
                ).get("slot_observability") if handoffs else None,
                "slot_wait_count": getattr(
                    handoffs[0], "telemetry", {}
                ).get("slot_wait_count") if handoffs else None,
            }
        )
        self._pending_handoff_batch = None
        if self._prefill_future is None and not self._pending:
            self._flush_transfer_acks()

    def _admit_pending_handoff_batch(self):
        batch = self._pending_handoff_batch
        if batch is None:
            return False
        admit_started_at = perf_counter()
        admissions = list(self.coordinator.admit_batch(batch["handoffs"]))
        admit_finished_at = perf_counter()
        self._apply_handoff_admissions(
            batch,
            admissions,
            admit_started_at=admit_started_at,
            admit_finished_at=admit_finished_at,
            decode_rpc_timing=self.coordinator.last_rpc_timing("decode"),
        )
        return True

    def _record_queue_sample(self):
        self._queue_samples.append(
            {
                "elapsed_ms": (perf_counter() - self._run_started_at) * 1000,
                "pending_prefill_requests": len(self._pending),
                "active_decode_requests": len(self._active_by_decode_seq),
                "prefill_inflight": bool(
                    self._prefill_future is not None
                    and not self._prefill_future.done()
                ),
                "pending_transfer_acks": len(self._pending_transfer_acks),
                "pending_kv_imports": len(self._awaiting_transfer_completions),
            }
        )

    def _record_decode_idle_interval(
        self,
        reason: DecodeIdleReason,
        started_at: float,
        finished_at: float,
    ):
        if not self.enable_latency_telemetry:
            return
        self._decode_idle_intervals.append({
            "reason": reason.value,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": max(0.0, (finished_at - started_at) * 1000),
        })

    def _fail_all_requests(self, error: Exception):
        reason = f"{type(error).__name__}: {error}"
        self._fatal_error = reason
        try:
            self._failure_worker_health = self.coordinator.worker_health()
        except Exception:
            self._failure_worker_health = {}
        now = perf_counter()
        for request in self._requests.values():
            lifecycle = request["lifecycle"]
            if not lifecycle.is_terminal:
                lifecycle.fail(reason)
                request["finish_time"] = now
        self._pending.clear()
        self._active_by_decode_seq.clear()
        self._pending_handoff_batch = None
        self._pending_transfer_acks.clear()
        self._awaiting_transfer_completions.clear()
        if self._prefill_future is not None and (
            self._prefill_future.done() or self._prefill_future.cancel()
        ):
            self._prefill_future = None
            self._prefill_future_meta = None
        self.last_step_events = {
            "failed": True,
            "failure_reason": reason,
            "waiting_queue_size": 0,
            "running_queue_size": 0,
        }

    def step(self):
        if self._closed:
            if self._fatal_error is not None:
                raise RuntimeError(f"PD serving engine failed: {self._fatal_error}")
            raise RuntimeError("PD serving engine is closed")
        try:
            return self._step_once()
        except Exception as error:
            self._fail_all_requests(error)
            try:
                self.exit()
            except Exception:
                # Cleanup is best effort; the first Worker failure is authoritative.
                pass
            raise

    def _step_once(self):
        self._start_prefill()
        if self._prefill_future is not None and self._prefill_future.done():
            self._collect_prefill()
            self._start_prefill()

        self._collect_completed_transfer_acks()

        if not self._active_by_decode_seq:
            self._admit_pending_handoff_batch()
            if not self._active_by_decode_seq and self._prefill_future is not None:
                wait_started_at = perf_counter()
                self._prefill_future.result()
                wait_finished_at = perf_counter()
                self._record_decode_idle_interval(
                    DecodeIdleReason.WAITING_PREFILL_OUTPUT,
                    wait_started_at,
                    wait_finished_at,
                )
                self._collect_prefill()
                self._start_prefill()
                self._admit_pending_handoff_batch()
            if not self._active_by_decode_seq:
                if self._awaiting_transfer_completions:
                    self._collect_completed_transfer_acks(wait=True)
                self._flush_transfer_acks()
                finished = []
                for request_id, request in self._requests.items():
                    token_ids = request.pop("finished_without_step", None)
                    if token_ids is not None:
                        finished.append((request_id, token_ids))
                self._record_queue_sample()
                return finished, 0

        active_decode_before_step = len(self._active_by_decode_seq)
        decode_started_at = perf_counter()
        handoff_batch = self._pending_handoff_batch
        combined_step = getattr(self.coordinator, "decode_step_with_handoffs", None)
        if (
            self.enable_transport_overlap
            and handoff_batch is not None
            and combined_step is not None
        ):
            result = combined_step(handoff_batch["handoffs"])
            decode_finished_at = perf_counter()
            self._apply_handoff_admissions(
                handoff_batch,
                list(result.get("admissions") or ()),
                admit_started_at=decode_started_at,
                admit_finished_at=decode_finished_at,
                decode_rpc_timing=self.coordinator.last_rpc_timing("decode"),
                admission_timing=result.get("admission_timing"),
                combined_step_roundtrip_ms=(
                    decode_finished_at - decode_started_at
                ) * 1000,
            )
        else:
            self._admit_pending_handoff_batch()
            result = self.coordinator.decode_step()
            decode_finished_at = perf_counter()
        raw_events = result.get("last_step_events") or {}
        self._record_completed_transfers(result.get("completed_transfers") or ())
        step_diagnostics = result.get("step_diagnostics") or {}
        if step_diagnostics.get("idle_reason") == DecodeIdleReason.WAITING_KV_H2D.value:
            self._record_decode_idle_interval(
                DecodeIdleReason.WAITING_KV_H2D,
                step_diagnostics["started_at"],
                step_diagnostics["finished_at"],
            )
        elif (
            active_decode_before_step
            and not raw_events.get("scheduled_seq_ids")
        ):
            self._record_decode_idle_interval(
                DecodeIdleReason.SCHEDULER_NOT_SCHEDULED,
                decode_started_at,
                decode_finished_at,
            )
        outputs = []
        for decode_seq_id, token_ids in result.get("outputs", ()):
            request_id = self._active_by_decode_seq.pop(decode_seq_id, None)
            if request_id is None:
                raise RuntimeError("Decode Worker returned an unknown sequence id")
            request = self._requests[request_id]
            request["finish_time"] = perf_counter()
            if self.enable_latency_telemetry:
                timeline = request.get("timeline") or {}
                request["timeline"] = timeline
                timeline["t_finish"] = request[
                    "finish_time"
                ]
                timeline.setdefault("writers", {})[
                    "t_finish"
                ] = "pd_parent.decode_result"
            request["lifecycle"].transition(RequestState.FINISHED)
            outputs.append((request_id, list(token_ids)))
        events = deepcopy(raw_events)
        scheduled_ids = events.get("scheduled_seq_ids", ())
        events["scheduled_seq_ids"] = [
            self._active_by_decode_seq.get(seq_id, seq_id)
            for seq_id in scheduled_ids
        ]
        events["waiting_queue_size"] = events.get("waiting_queue_size", 0) + len(
            self._pending
        )
        events["pd_active_decode_requests"] = len(self._active_by_decode_seq)
        events["pd_pending_prefill_requests"] = len(self._pending)
        events["pd_pending_kv_imports"] = len(
            self._awaiting_transfer_completions
        )
        if self.enable_latency_telemetry:
            events["pd_slot_state"] = (
                deepcopy(self._prefill_batches[-1].get("slot_stats_after_ready"))
                if self._prefill_batches else None
            )
            events["pd_handoff_count"] = len(self._prefill_batches)
        self.last_step_events = events
        self._record_queue_sample()
        self._start_prefill()
        return outputs, int(result.get("num_tokens", 0))

    def is_finished(self):
        if self._fatal_error is not None:
            return all(
                request["lifecycle"].is_terminal
                for request in self._requests.values()
            )
        return (
            not self._pending
            and self._prefill_future is None
            and self._pending_handoff_batch is None
            and not self._active_by_decode_seq
            and not self._pending_transfer_acks
            and not self._awaiting_transfer_completions
        )

    def reset_metrics(self):
        if self._fatal_error is not None:
            raise RuntimeError("cannot reset metrics on a failed PD engine")
        if not self.is_finished():
            raise RuntimeError("cannot reset PD metrics while requests are active")
        self.coordinator.reset_decode_metrics()
        self._requests.clear()
        self._awaiting_transfer_completions.clear()
        self._pending_handoff_batch = None
        self._prefill_batches.clear()
        self._queue_samples.clear()
        self._decode_idle_intervals.clear()
        self.last_step_events = {}
        self._run_started_at = perf_counter()

    def get_metrics(self):
        worker_metrics = (
            {"requests": [], "summary": {}}
            if self._fatal_error is not None
            else self.coordinator.decode_metrics()
        )
        worker_requests = worker_metrics.get("requests", [])
        seq_to_request_id = {
            request["decode_seq_id"]: request_id
            for request_id, request in self._requests.items()
            if request.get("decode_seq_id") is not None
        }
        requests = []
        included_request_ids = set()
        for worker_request in worker_requests:
            request = dict(worker_request)
            request_id = seq_to_request_id.get(request.get("seq_id"))
            if request_id is None:
                continue
            request["seq_id"] = request_id
            local_request = self._requests[request_id]
            request["arrival_time"] = local_request["arrival_time"]
            if self.enable_latency_telemetry:
                timeline = deepcopy(local_request.get("timeline"))
                if timeline is not None:
                    timeline["t_finish"] = request.get("finish_time")
                    timeline.setdefault("writers", {})["t_finish"] = (
                        "decode_worker.engine"
                    )
                request["timeline"] = timeline
            request["status"] = local_request["lifecycle"].state.value
            request["cancelled"] = (
                local_request["lifecycle"].state == RequestState.CANCELLED
            )
            requests.append(request)
            included_request_ids.add(request_id)
        for request_id, request in self._requests.items():
            if request_id not in included_request_ids:
                lifecycle = request["lifecycle"]
                now = request.get("finish_time") or perf_counter()
                output_token_ids = request.get("finished_without_step") or ()
                success = lifecycle.state == RequestState.FINISHED
                cancelled = lifecycle.state == RequestState.CANCELLED
                requests.append(
                    {
                        "seq_id": request_id,
                        "prompt_tokens": request["prompt_tokens"],
                        "output_tokens": len(output_token_ids),
                        "success": success,
                        "cancelled": cancelled,
                        "status": lifecycle.state.value,
                        "failure_reason": lifecycle.terminal_reason,
                        "arrival_time": request["arrival_time"],
                        "first_token_time": now if output_token_ids else None,
                        "token_times": [now] if output_token_ids else [],
                        "output_event_times": [now] if output_token_ids else [],
                        "finish_time": request.get("finish_time"),
                        "timeline": deepcopy(request.get("timeline")),
                    }
                )
        summary = deepcopy(worker_metrics.get("summary", {}))
        summary["num_requests"] = len(requests)
        summary["num_finished"] = sum(
            1 for request in requests if request.get("success")
        )
        summary["num_cancelled"] = sum(
            1 for request in requests if request.get("cancelled")
        )
        summary["num_failed"] = (
            summary["num_requests"]
            - summary["num_finished"]
            - summary["num_cancelled"]
        )
        timing_fields = {
            "roundtrip_ms": "prefill_roundtrip_ms",
            "admit_ms": "admit_roundtrip_ms",
            "handoff_path_ms": "handoff_path_ms",
            "worker_ms": "prefill_worker_ms",
            "model_forward_ms": "prefill_model_forward_ms",
            "model_forward_gpu_ms": "prefill_model_forward_gpu_ms",
            "kv_export_copy_ms": "prefill_kv_export_copy_ms",
            "parent_overhead_ms": "prefill_parent_overhead_ms",
            "forward_calls": "prefill_forward_calls",
            "prefill_command_queue_ms": "prefill_command_queue_ms",
            "prefill_response_queue_ms": "prefill_response_queue_ms",
            "decode_command_queue_ms": "decode_command_queue_ms",
            "decode_worker_admit_ms": "decode_worker_admit_ms",
            "decode_response_queue_ms": "decode_response_queue_ms",
            "combined_step_roundtrip_ms": "combined_step_roundtrip_ms",
        }
        prefill_timing = {}
        for output_name, field_name in timing_fields.items():
            values = [
                batch[field_name]
                for batch in self._prefill_batches
                if batch.get(field_name) is not None
            ]
            prefill_timing[output_name] = (
                sum(values) / len(values) if values else None
            )
        summary["pd"] = {
            "prefill_batches": len(self._prefill_batches),
            "prefill_batch_size_mean": (
                sum(batch["batch_size"] for batch in self._prefill_batches)
                / len(self._prefill_batches)
                if self._prefill_batches else None
            ),
            "prefill_batches_detail": deepcopy(self._prefill_batches),
            "prefill_timing": prefill_timing,
            "queue_samples": deepcopy(self._queue_samples),
            "decode_idle": {
                "intervals": deepcopy(self._decode_idle_intervals),
                "counts": {
                    reason.value: sum(
                        1
                        for interval in self._decode_idle_intervals
                        if interval["reason"] == reason.value
                    )
                    for reason in DecodeIdleReason
                },
                "duration_ms": {
                    reason.value: sum(
                        interval["duration_ms"]
                        for interval in self._decode_idle_intervals
                        if interval["reason"] == reason.value
                    )
                    for reason in DecodeIdleReason
                },
            },
            "slot_release_samples": deepcopy(self._slot_release_samples),
            "worker_health": (
                deepcopy(self._failure_worker_health)
                if self._failure_worker_health is not None
                else self.coordinator.worker_health()
            ),
            "fatal_error": self._fatal_error,
        }
        return {"requests": sorted(requests, key=lambda item: item["seq_id"]), "summary": summary}

    def exit(self):
        if self._closed:
            return
        self._closed = True
        if self._prefill_future is not None:
            self._prefill_future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.coordinator.close()
