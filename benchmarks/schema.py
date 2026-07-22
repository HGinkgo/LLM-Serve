from benchmarks.metrics import summarize_values


def _milliseconds_summary(values):
    summary = summarize_values([value * 1000 for value in values])
    return summary


def compact_request_record(request: dict):
    first_token_time = request.get("first_token_time")
    finish_time = request.get("finish_time")
    arrival_time = request["arrival_time"]
    output_tokens = request["output_tokens"]
    token_times = request.get("token_times", [])
    output_event_times = request.get("output_event_times", [])

    ttft_ms = (
        (first_token_time - arrival_time) * 1000
        if first_token_time is not None
        else None
    )
    e2e_ms = (
        (finish_time - arrival_time) * 1000
        if finish_time is not None
        else None
    )
    if output_tokens > 1 and first_token_time is not None and finish_time is not None:
        tpot_ms = (finish_time - first_token_time) * 1000 / (output_tokens - 1)
    elif output_tokens == 1 and first_token_time is not None:
        tpot_ms = 0.0
    else:
        tpot_ms = None

    burst_itl = request.get("burst_itl")
    if burst_itl is None:
        burst_itl = [
            token_times[index] - token_times[index - 1]
            for index in range(1, len(token_times))
        ]
    output_event_latency = request.get("output_event_latency")
    if output_event_latency is None:
        output_event_latency = [
            output_event_times[index] - output_event_times[index - 1]
            for index in range(1, len(output_event_times))
        ]
    speculative_step_latency = request.get("speculative_step_latency", [])

    return {
        "seq_id": request["seq_id"],
        "request_id": request.get("request_id"),
        "request_class": request["request_class"],
        "prompt_tokens": request["prompt_tokens"],
        "output_tokens": output_tokens,
        "success": request["success"],
        "failure_reason": request.get("failure_reason"),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "burst_itl_ms": _milliseconds_summary(burst_itl),
        "output_event_latency_ms": _milliseconds_summary(output_event_latency),
        "speculative_step_latency_ms": _milliseconds_summary(
            speculative_step_latency
        ),
    }
