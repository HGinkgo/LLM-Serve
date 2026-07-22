import time
from collections import deque
from collections.abc import Callable, Sequence

from benchmarks.workloads import RequestSpec


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
    start = clock()

    while pending or not engine.is_finished():
        elapsed = clock() - start
        while pending and pending[0][0] <= elapsed:
            arrival_time, spec = pending.popleft()
            seq_id = engine.add_request(
                list(spec.prompt_token_ids),
                make_sampling_params(spec),
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
        "engine_summary": engine_metrics.get("summary", {}),
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
            seq_id = engine.add_request(
                list(spec.prompt_token_ids),
                make_sampling_params(spec),
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
        "engine_summary": engine_metrics.get("summary", {}),
    }
