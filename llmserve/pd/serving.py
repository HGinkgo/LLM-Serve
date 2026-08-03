"""Continuous-batching adapter for dual-worker PD serving."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter

from llmserve.pd.protocol import RequestEnvelope


class PDServingEngine:
    """Expose a step-wise engine interface over Prefill/Decode workers.

    The adapter keeps the Prefill command asynchronous, while Decode remains
    the clock for one benchmark step. This preserves iteration-level batching
    and allows the existing serving benchmark to consume PD without treating
    a whole request batch as one opaque timing sample.
    """

    def __init__(self, coordinator, prefill_batch_size: int = 1):
        if not isinstance(prefill_batch_size, int) or prefill_batch_size <= 0:
            raise ValueError("prefill_batch_size must be a positive integer")
        self.coordinator = coordinator
        self.prefill_batch_size = prefill_batch_size
        self._next_request_id = 0
        self._pending: deque[RequestEnvelope] = deque()
        self._requests: dict[int, dict] = {}
        self._active_by_decode_seq: dict[int, int] = {}
        self._prefill_future: Future | None = None
        self._prefill_future_meta: dict | None = None
        self._pending_transfer_acks: list[str] = []
        self._slot_release_samples: list[dict] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False
        self.last_step_events: dict = {}
        self._prefill_batches: list[dict] = []
        self._queue_samples: list[dict] = []
        self._run_started_at = perf_counter()
        self.coordinator.start()

    def add_request(self, prompt: str | list[int], sampling_params) -> int:
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
            "decode_seq_id": None,
        }
        return request_id

    def _start_prefill(self):
        if self._prefill_future is not None or not self._pending:
            return
        batch = []
        while self._pending and len(batch) < self.prefill_batch_size:
            batch.append(self._pending.popleft())
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
        admit_started_at = perf_counter()
        admissions = list(self.coordinator.admit_batch(handoffs))
        admit_finished_at = perf_counter()
        decode_rpc_timing = self.coordinator.last_rpc_timing("decode")
        if len(admissions) != len(handoffs):
            raise RuntimeError("Decode Worker returned an incomplete admission batch")
        for handoff, admission in zip(handoffs, admissions):
            request_id = handoff.request_id
            if request_id not in self._requests:
                raise RuntimeError("Decode Worker admitted an unknown request")
            descriptor = getattr(handoff, "descriptor", None)
            if getattr(descriptor, "transport", "inline") == "shared_slot":
                transfer_id = admission.get("transfer_id")
                if transfer_id != descriptor.transfer_id:
                    raise RuntimeError("Decode Worker returned an invalid transfer ACK")
                self._pending_transfer_acks.append(transfer_id)
            if admission.get("finished"):
                self._requests[request_id]["finished_without_step"] = list(
                    admission.get("output_token_ids") or ()
                )
                continue
            decode_seq_id = admission.get("seq_id")
            if decode_seq_id in self._active_by_decode_seq:
                raise RuntimeError("Decode Worker returned duplicate sequence ids")
            self._active_by_decode_seq[decode_seq_id] = request_id
            self._requests[request_id]["decode_seq_id"] = decode_seq_id
        prefill_timing = getattr(handoffs[0], "prefill_timing_ms", {}) if handoffs else {}
        worker_total_ms = prefill_timing.get("worker_total_ms")
        self._prefill_batches.append(
            {
                "request_ids": list(meta.get("request_ids", ())),
                "batch_size": meta.get("batch_size", 0),
                "prefill_roundtrip_ms": (
                    finished_at - meta.get("submitted_at", finished_at)
                ) * 1000,
                "admit_roundtrip_ms": (
                    admit_finished_at - admit_started_at
                ) * 1000,
                "handoff_path_ms": (
                    admit_finished_at - finished_at
                ) * 1000,
                "prefill_worker_ms": worker_total_ms,
                "prefill_model_forward_ms": prefill_timing.get(
                    "model_forward_ms"
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
                "decode_worker_admit_ms": decode_rpc_timing.get(
                    "worker_service_ms"
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
            }
        )
        if self._prefill_future is None and not self._pending:
            self._flush_transfer_acks()
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
            }
        )

    def step(self):
        if self._closed:
            raise RuntimeError("PD serving engine is closed")
        self._start_prefill()
        if self._prefill_future is not None and self._prefill_future.done():
            self._collect_prefill()
            self._start_prefill()

        if not self._active_by_decode_seq:
            if self._prefill_future is not None:
                self._prefill_future.result()
                self._collect_prefill()
                self._start_prefill()
            if not self._active_by_decode_seq:
                self._flush_transfer_acks()
                finished = []
                for request_id, request in self._requests.items():
                    token_ids = request.pop("finished_without_step", None)
                    if token_ids is not None:
                        finished.append((request_id, token_ids))
                self._record_queue_sample()
                return finished, 0

        result = self.coordinator.decode_step()
        outputs = []
        for decode_seq_id, token_ids in result.get("outputs", ()):
            request_id = self._active_by_decode_seq.pop(decode_seq_id, None)
            if request_id is None:
                raise RuntimeError("Decode Worker returned an unknown sequence id")
            outputs.append((request_id, list(token_ids)))
        events = deepcopy(result.get("last_step_events") or {})
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
        self.last_step_events = events
        self._record_queue_sample()
        self._start_prefill()
        return outputs, int(result.get("num_tokens", 0))

    def is_finished(self):
        return (
            not self._pending
            and self._prefill_future is None
            and not self._active_by_decode_seq
            and not self._pending_transfer_acks
        )

    def reset_metrics(self):
        if not self.is_finished():
            raise RuntimeError("cannot reset PD metrics while requests are active")
        self.coordinator.reset_decode_metrics()
        self._requests.clear()
        self._prefill_batches.clear()
        self._queue_samples.clear()
        self.last_step_events = {}
        self._run_started_at = perf_counter()

    def get_metrics(self):
        worker_metrics = self.coordinator.decode_metrics()
        worker_requests = worker_metrics.get("requests", [])
        seq_to_request_id = {
            request["decode_seq_id"]: request_id
            for request_id, request in self._requests.items()
            if request.get("decode_seq_id") is not None
        }
        requests = []
        for worker_request in worker_requests:
            request = dict(worker_request)
            request_id = seq_to_request_id.get(request.get("seq_id"))
            if request_id is None:
                continue
            request["seq_id"] = request_id
            request["arrival_time"] = self._requests[request_id]["arrival_time"]
            requests.append(request)
        for request_id, request in self._requests.items():
            if request.get("finished_without_step") is not None and not any(
                item["seq_id"] == request_id for item in requests
            ):
                now = perf_counter()
                requests.append(
                    {
                        "seq_id": request_id,
                        "prompt_tokens": 0,
                        "output_tokens": len(request["finished_without_step"]),
                        "success": True,
                        "failure_reason": None,
                        "arrival_time": request["arrival_time"],
                        "first_token_time": now,
                        "token_times": [now],
                        "output_event_times": [now],
                        "finish_time": now,
                    }
                )
        summary = deepcopy(worker_metrics.get("summary", {}))
        timing_fields = {
            "roundtrip_ms": "prefill_roundtrip_ms",
            "admit_ms": "admit_roundtrip_ms",
            "handoff_path_ms": "handoff_path_ms",
            "worker_ms": "prefill_worker_ms",
            "model_forward_ms": "prefill_model_forward_ms",
            "kv_export_copy_ms": "prefill_kv_export_copy_ms",
            "parent_overhead_ms": "prefill_parent_overhead_ms",
            "forward_calls": "prefill_forward_calls",
            "prefill_command_queue_ms": "prefill_command_queue_ms",
            "prefill_response_queue_ms": "prefill_response_queue_ms",
            "decode_command_queue_ms": "decode_command_queue_ms",
            "decode_worker_admit_ms": "decode_worker_admit_ms",
            "decode_response_queue_ms": "decode_response_queue_ms",
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
            "slot_release_samples": deepcopy(self._slot_release_samples),
            "worker_health": self.coordinator.worker_health(),
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
