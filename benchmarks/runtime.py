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
    scheduled_batch_sizes = []
    waiting_queue_sizes = []
    running_queue_sizes = []
    start = clock()

    while pending or not engine.is_finished():
        elapsed = clock() - start
        while pending and pending[0][0] <= elapsed:
            _, spec = pending.popleft()
            seq_id = engine.add_request(
                list(spec.prompt_token_ids),
                make_sampling_params(spec),
            )
            seq_to_spec[seq_id] = spec

        if not engine.is_finished():
            engine.step()
            events = engine.last_step_events
            scheduled_batch_sizes.append(
                len(events.get("scheduled_seq_ids", []))
            )
            waiting_queue_sizes.append(events.get("waiting_queue_size", 0))
            running_queue_sizes.append(events.get("running_queue_size", 0))
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
        "waiting_queue_sizes": waiting_queue_sizes,
        "running_queue_sizes": running_queue_sizes,
        "engine_summary": engine_metrics.get("summary", {}),
    }
