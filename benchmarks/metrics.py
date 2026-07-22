from collections.abc import Mapping, Sequence


def percentile(values: Sequence[float], percentile_value: float):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize_values(values: Sequence[float]):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def _request_latency(request: Mapping):
    arrival_time = request["arrival_time"]
    first_token_time = request["first_token_time"]
    finish_time = request["finish_time"]
    output_tokens = request["output_tokens"]
    token_times = request.get("token_times", [])

    ttft = first_token_time - arrival_time
    e2e = finish_time - arrival_time
    if output_tokens > 1:
        tpot = (finish_time - first_token_time) / (output_tokens - 1)
    else:
        tpot = 0.0
    burst_itl = request.get("burst_itl")
    if burst_itl is None:
        burst_itl = [
            token_times[index] - token_times[index - 1]
            for index in range(1, len(token_times))
        ]
    output_event_latency = request.get("output_event_latency")
    if output_event_latency is None:
        output_event_times = request.get("output_event_times", [])
        output_event_latency = [
            output_event_times[index] - output_event_times[index - 1]
            for index in range(1, len(output_event_times))
        ]
    return {
        "ttft": ttft,
        "tpot": tpot,
        "burst_itl": burst_itl,
        "output_event_latency": output_event_latency,
        "speculative_step_latency": request.get(
            "speculative_step_latency", []
        ),
        "e2e": e2e,
    }


def _latency_summary(requests: Sequence[Mapping]):
    scalar_names = ("ttft", "tpot", "e2e")
    sample_names = (
        "burst_itl",
        "output_event_latency",
        "speculative_step_latency",
    )
    values = {name: [] for name in scalar_names + sample_names}
    for request in requests:
        latency = _request_latency(request)
        for name in scalar_names:
            values[name].append(latency[name])
        for name in sample_names:
            values[name].extend(latency[name])
    return {name: summarize_values(samples) for name, samples in values.items()}


def summarize_speculative_requests(requests: Sequence[Mapping]):
    steps = sum(request.get("speculative_steps", 0) for request in requests)
    draft_tokens = sum(
        request.get("speculative_draft_tokens", 0) for request in requests
    )
    accepted_tokens = sum(
        request.get("speculative_accepted_tokens", 0) for request in requests
    )
    emitted_tokens = sum(
        request.get("speculative_emitted_tokens", 0) for request in requests
    )
    accept_all_count = sum(
        request.get("speculative_accept_all_count", 0) for request in requests
    )
    gamma_counts = {}
    timing_totals = {}
    for request in requests:
        for gamma, count in request.get("speculative_gamma_counts", {}).items():
            gamma_counts[gamma] = gamma_counts.get(gamma, 0) + count
        for name, value in request.get("speculative_timing", {}).items():
            timing_totals[name] = timing_totals.get(name, 0.0) + value

    return {
        "steps": steps,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "emitted_tokens": emitted_tokens,
        "acceptance_rate": (
            accepted_tokens / draft_tokens if draft_tokens > 0 else None
        ),
        "acceptance_length": emitted_tokens / steps if steps > 0 else None,
        "accepted_length": accepted_tokens / steps if steps > 0 else None,
        "draft_tokens_per_step": draft_tokens / steps if steps > 0 else None,
        "accept_all_count": accept_all_count,
        "gamma_counts": dict(
            sorted(gamma_counts.items(), key=lambda item: int(item[0]))
        ),
        "timing": {
            name: {
                "total_ms": total * 1000,
                "mean_ms": total * 1000 / steps if steps > 0 else None,
            }
            for name, total in sorted(timing_totals.items())
        },
    }


def _goodput_summary(
    requests: Sequence[Mapping],
    duration: float,
    slo_ms: Mapping[str, float],
):
    allowed = {"ttft", "tpot", "e2e"}
    unknown = set(slo_ms) - allowed
    if unknown:
        raise ValueError(f"unsupported SLO metrics: {sorted(unknown)}")
    thresholds = {name: value / 1000 for name, value in slo_ms.items()}
    good = 0
    for request in requests:
        latency = _request_latency(request)
        if all(latency[name] <= threshold for name, threshold in thresholds.items()):
            good += 1
    return {
        "completed": good,
        "requests_per_second": good / duration,
        "slo_ms": dict(slo_ms),
    }


def summarize_serving_run(
    requests: Sequence[Mapping],
    duration: float,
    slo_ms: Mapping[str, float] | None = None,
):
    if duration <= 0:
        raise ValueError("duration must be positive")
    successful = [request for request in requests if request["success"]]
    groups = {"overall": successful}
    for request in successful:
        groups.setdefault(request["request_class"], []).append(request)

    input_tokens = sum(request["prompt_tokens"] for request in successful)
    output_tokens = sum(request["output_tokens"] for request in successful)
    return {
        "completed": len(successful),
        "failed": len(requests) - len(successful),
        "duration": duration,
        "throughput": {
            "requests_per_second": len(successful) / duration,
            "input_tokens_per_second": input_tokens / duration,
            "output_tokens_per_second": output_tokens / duration,
            "total_tokens_per_second": (input_tokens + output_tokens) / duration,
        },
        "latency": {
            name: _latency_summary(group_requests)
            for name, group_requests in groups.items()
        },
        "goodput": (
            _goodput_summary(successful, duration, slo_ms)
            if slo_ms is not None
            else None
        ),
    }
