import time
from collections import deque
from collections.abc import Callable, Sequence
from hashlib import sha256
import json

from benchmarks.workloads import RequestSpec


_SCHEDULER_TELEMETRY_FIELDS = (
    "step_start", "step_end", "waiting_queue_size", "running_queue_size",
    "pd_pending_prefill_requests", "pd_active_decode_requests",
    "pd_active_decode_by_worker", "pd_pending_kv_imports_by_worker",
    "prefill_token_count", "decode_token_count", "prefill_request_count",
    "decode_request_count", "remaining_token_budget",
    "max_num_batched_tokens", "max_num_seqs", "chunked_prefill",
    "partial_prefill_seq_ids", "partial_prefill_chunk_count",
    "prefill_chunk_lengths", "model_forward_gpu_ms", "model_forward_stage",
    "pd_slot_state", "pd_handoff_count",
    "replica_events", "replica_queue_state",
)


def _submit_request(engine, prompt_token_ids, sampling_params, clock):
    submitted_at = clock()
    benchmark_submit = getattr(engine, "add_benchmark_request", None)
    if benchmark_submit is not None:
        return benchmark_submit(
            prompt_token_ids,
            sampling_params,
            submitted_at,
        )
    seq_id = engine.add_request(prompt_token_ids, sampling_params)
    recorder = getattr(engine, "record_benchmark_submit", None)
    if recorder is not None:
        recorder(seq_id, submitted_at)
    return seq_id


def _compact_scheduler_step(events):
    record = {
        name: events.get(name)
        for name in _SCHEDULER_TELEMETRY_FIELDS
        if name in events
    }
    record["scheduled_request_count"] = len(events.get("scheduled_seq_ids", ()))
    return record


def _request_trace_audit(seq_to_spec):
    """Audit deterministic request content without persisting prompt tokens."""
    entries = []
    for spec in sorted(seq_to_spec.values(), key=lambda item: item.request_id):
        prompt_payload = json.dumps(
            list(spec.prompt_token_ids), separators=(",", ":")
        ).encode()
        entries.append({
            "request_id": spec.request_id,
            "request_class": spec.request_class,
            "input_len": spec.input_len,
            "output_len": spec.output_len,
            "prompt_sha256": sha256(prompt_payload).hexdigest(),
        })
    digest_payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "entry_count": len(entries),
        "sha256": sha256(digest_payload).hexdigest(),
        "entries": entries,
    }


def run_poisson(
    engine,
    request_specs: Sequence[RequestSpec],
    arrival_times: Sequence[float],
    make_sampling_params: Callable[[RequestSpec], object],
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
):
    if len(request_specs) != len(arrival_times):
        raise ValueError("request_specs and arrival_times must have the same length")
    if not request_specs:
        raise ValueError("request_specs cannot be empty")

    pending = deque(zip(arrival_times, request_specs))
    seq_to_spec = {}
    seq_to_arrival = {}
    scheduled_batch_sizes = []
    speculative_batch_sizes = []
    waiting_queue_sizes = []
    running_queue_sizes = []
    scheduler_steps = []
    start = clock()

    while pending or not engine.is_finished():
        elapsed = clock() - start
        while pending and pending[0][0] <= elapsed:
            arrival_time, spec = pending.popleft()
            seq_id = _submit_request(
                engine,
                list(spec.prompt_token_ids),
                make_sampling_params(spec),
                clock,
            )
            seq_to_spec[seq_id] = spec
            seq_to_arrival[seq_id] = start + arrival_time

        if not engine.is_finished():
            engine.step()
            events = engine.last_step_events
            scheduled_batch_sizes.append(
                len(events.get("scheduled_seq_ids", []))
            )
            waiting_queue_sizes.append(events.get("waiting_queue_size", 0))
            running_queue_sizes.append(events.get("running_queue_size", 0))
            scheduler_steps.append(_compact_scheduler_step(events))
            if events.get("speculative"):
                speculative_batch_sizes.append(
                    events.get("speculative_batch_size", 0)
                )
            continue

        if pending:
            delay = pending[0][0] - (clock() - start)
            if delay > 0:
                sleep(delay)

    engine_metrics = engine.get_metrics()
    requests = []
    for request in engine_metrics["requests"]:
        request = dict(request)
        spec = seq_to_spec[request["seq_id"]]
        request["arrival_time"] = seq_to_arrival[request["seq_id"]]
        request["request_id"] = spec.request_id
        request["request_class"] = spec.request_class
        requests.append(request)
    requests.sort(key=lambda request: request["request_id"])

    first_arrival = min(request["arrival_time"] for request in requests)
    finished = [
        request["finish_time"]
        for request in requests
        if request["finish_time"] is not None
    ]
    end = max(finished) if finished else clock()
    return {
        "admitted": len(seq_to_spec),
        "duration": end - first_arrival,
        "requests": requests,
        "scheduled_batch_sizes": scheduled_batch_sizes,
        "speculative_batch_sizes": speculative_batch_sizes,
        "waiting_queue_sizes": waiting_queue_sizes,
        "running_queue_sizes": running_queue_sizes,
        "scheduler_steps": scheduler_steps,
        "engine_summary": engine_metrics.get("summary", {}),
        "request_trace": _request_trace_audit(seq_to_spec),
    }


def run_closed_loop(
    engine,
    request_specs,
    max_concurrency: int,
    warmup_seconds: float,
    measurement_seconds: float,
    make_sampling_params: Callable[[RequestSpec], object],
    clock: Callable[[], float] = time.perf_counter,
):
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be non-negative")
    if measurement_seconds <= 0:
        raise ValueError("measurement_seconds must be positive")

    request_specs = iter(request_specs)
    seq_to_spec = {}
    active_requests = 0

    def refill():
        nonlocal active_requests
        while active_requests < max_concurrency:
            spec = next(request_specs)
            seq_id = _submit_request(
                engine,
                list(spec.prompt_token_ids),
                make_sampling_params(spec),
                clock,
            )
            seq_to_spec[seq_id] = spec
            active_requests += 1

    refill()
    measurement_start = clock() + warmup_seconds
    measurement_end = measurement_start + measurement_seconds
    scheduled_batch_sizes = []
    speculative_batch_sizes = []
    waiting_queue_sizes = []
    running_queue_sizes = []
    scheduler_steps = []

    while clock() < measurement_end:
        outputs, _ = engine.step()
        active_requests -= len(outputs)
        events = engine.last_step_events
        step_end = events.get("step_end", clock())
        if measurement_start <= step_end < measurement_end:
            scheduled_batch_sizes.append(
                len(events.get("scheduled_seq_ids", []))
            )
            waiting_queue_sizes.append(events.get("waiting_queue_size", 0))
            running_queue_sizes.append(events.get("running_queue_size", 0))
            scheduler_steps.append(_compact_scheduler_step(events))
            if events.get("speculative"):
                speculative_batch_sizes.append(
                    events.get("speculative_batch_size", 0)
                )
        if step_end < measurement_end:
            refill()

    while not engine.is_finished():
        engine.step()

    engine_metrics = engine.get_metrics()
    requests = []
    for request in engine_metrics["requests"]:
        request = dict(request)
        spec = seq_to_spec[request["seq_id"]]
        request["request_id"] = spec.request_id
        request["request_class"] = spec.request_class
        requests.append(request)
    requests.sort(key=lambda request: request["request_id"])

    latency_requests = [
        request
        for request in requests
        if request["arrival_time"] >= measurement_start
        and request["finish_time"] is not None
        and request["finish_time"] < measurement_end
    ]
    window_completed = sum(
        request["finish_time"] is not None
        and measurement_start <= request["finish_time"] < measurement_end
        for request in requests
    )
    window_output_tokens = sum(
        measurement_start <= token_time < measurement_end
        for request in requests
        for token_time in request.get("token_times", [])
    )
    return {
        "admitted": len(seq_to_spec),
        "duration": measurement_seconds,
        "measurement_start": measurement_start,
        "measurement_end": measurement_end,
        "requests": requests,
        "latency_requests": latency_requests,
        "window_completed": window_completed,
        "window_output_tokens": window_output_tokens,
        "scheduled_batch_sizes": scheduled_batch_sizes,
        "speculative_batch_sizes": speculative_batch_sizes,
        "waiting_queue_sizes": waiting_queue_sizes,
        "running_queue_sizes": running_queue_sizes,
        "scheduler_steps": scheduler_steps,
        "engine_summary": engine_metrics.get("summary", {}),
        "request_trace": _request_trace_audit(seq_to_spec),
    }
