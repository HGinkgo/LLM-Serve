import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev


def _rate_label(request_rate: float):
    return str(request_rate).replace(".", "p")


def _point_dimensions(experiment: dict):
    arrival = experiment["arrival"]
    if arrival == "poisson":
        return (
            ("request_rate", value, f"rate-{_rate_label(value)}")
            for value in experiment["request_rates"]
        )
    if arrival == "closed-loop":
        return (
            ("max_concurrency", value, f"concurrency-{value}")
            for value in experiment["max_concurrencies"]
        )
    raise ValueError(f"unsupported arrival: {arrival}")


def expand_suite(suite: dict) -> list[dict]:
    if suite.get("schema_version") != 1:
        raise ValueError("unsupported suite schema_version")
    runs = suite.get("runs", 0)
    if runs <= 0:
        raise ValueError("suite runs must be positive")
    profiles = suite.get("profiles", {})
    points = []

    for experiment in suite.get("experiments", []):
        profile_name = experiment["profile"]
        if profile_name not in profiles:
            raise ValueError(f"unknown profile: {profile_name}")
        arrival = experiment["arrival"]
        runtime_defaults = deepcopy(experiment.get("runtime", {}))
        runtime_defaults.setdefault("enable_chunked_prefill", False)

        for dimension_name, dimension_value, dimension_label in _point_dimensions(experiment):
            for run in range(runs):
                for variant in experiment["variants"]:
                    runtime = deepcopy(runtime_defaults)
                    runtime.update(
                        {
                            key: value
                            for key, value in variant.items()
                            if key != "name"
                        }
                    )
                    variant_name = variant["name"]
                    point_id = (
                        f"{experiment['name']}-{variant_name}-"
                        f"{dimension_label}-r{run + 1}"
                    )
                    point = {
                        "point_id": point_id,
                        "suite": suite["name"],
                        "experiment": experiment["name"],
                        "arrival": arrival,
                        "variant": variant_name,
                        "run": run,
                        "workload_seed": run,
                        "arrival_seed": run,
                        "workload": deepcopy(profiles[profile_name]),
                        "runtime": runtime,
                        "slo_ms": deepcopy(experiment.get("slo_ms")),
                    }
                    point[dimension_name] = dimension_value
                    if arrival == "poisson":
                        point["num_requests"] = experiment["num_requests"]
                    else:
                        point["warmup_seconds"] = experiment["warmup_seconds"]
                        point["measurement_seconds"] = experiment[
                            "measurement_seconds"
                        ]
                    points.append(point)
    if len({point["point_id"] for point in points}) != len(points):
        raise ValueError("suite expands to duplicate point_id values")
    return points


def can_resume_result(path: Path, point: dict, git_commit: str) -> bool:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == 2
        and result.get("complete") is True
        and result.get("point_id") == point["point_id"]
        and result.get("git_commit") == git_commit
    )


def _nested_value(value: dict, path: str):
    for part in path.split("."):
        value = value[part]
    return float(value)


def _optional_nested_value(value: dict, path: str, scale: float = 1.0):
    try:
        for part in path.split("."):
            value = value[part]
    except (KeyError, TypeError):
        return None
    return value * scale if value is not None else None


def build_summary_rows(results: list[dict]) -> list[dict]:
    fields = {
        "request_throughput_rps": (
            "metrics.throughput.requests_per_second", 1.0
        ),
        "input_throughput_tps": (
            "metrics.throughput.input_tokens_per_second", 1.0
        ),
        "output_throughput_tps": (
            "metrics.throughput.output_tokens_per_second", 1.0
        ),
        "total_throughput_tps": (
            "metrics.throughput.total_tokens_per_second", 1.0
        ),
        "ttft_p50_ms": ("metrics.latency.overall.ttft.p50", 1000.0),
        "ttft_p99_ms": ("metrics.latency.overall.ttft.p99", 1000.0),
        "tpot_p50_ms": ("metrics.latency.overall.tpot.p50", 1000.0),
        "tpot_p99_ms": ("metrics.latency.overall.tpot.p99", 1000.0),
        "burst_itl_p50_ms": (
            "metrics.latency.overall.burst_itl.p50", 1000.0
        ),
        "burst_itl_p99_ms": (
            "metrics.latency.overall.burst_itl.p99", 1000.0
        ),
        "output_event_latency_p50_ms": (
            "metrics.latency.overall.output_event_latency.p50", 1000.0
        ),
        "output_event_latency_p99_ms": (
            "metrics.latency.overall.output_event_latency.p99", 1000.0
        ),
        "speculative_step_latency_p50_ms": (
            "metrics.latency.overall.speculative_step_latency.p50", 1000.0
        ),
        "speculative_step_latency_p99_ms": (
            "metrics.latency.overall.speculative_step_latency.p99", 1000.0
        ),
        "e2e_p50_ms": ("metrics.latency.overall.e2e.p50", 1000.0),
        "e2e_p99_ms": ("metrics.latency.overall.e2e.p99", 1000.0),
        "acceptance_rate": ("metrics.speculative.acceptance_rate", 1.0),
        "acceptance_length": (
            "metrics.speculative.acceptance_length", 1.0
        ),
        "scheduled_batch_size_mean": (
            "metrics.scheduled_batch_size.mean", 1.0
        ),
        "waiting_queue_size_p99": (
            "metrics.waiting_queue_size.p99", 1.0
        ),
        "running_queue_size_p99": (
            "metrics.running_queue_size.p99", 1.0
        ),
        "goodput_rps": ("metrics.goodput.requests_per_second", 1.0),
        "speculative_batch_size_mean": (
            "metrics.speculative_batch_size.mean", 1.0
        ),
        "kv_total_blocks": ("metrics.kv_cache.total_blocks", 1.0),
        "kv_peak_reserved_blocks": (
            "metrics.kv_cache.peak_reserved_blocks", 1.0
        ),
        "kv_preemptions": ("metrics.kv_cache.preemptions", 1.0),
        "kv_admission_deferrals": (
            "metrics.kv_cache.admission_deferrals", 1.0
        ),
        "kv_cache_gib": (
            "metrics.kv_cache.kv_cache_bytes", 1.0 / (1024 ** 3)
        ),
        "model_runtime_gib": (
            "metrics.kv_cache.model_runtime_bytes", 1.0 / (1024 ** 3)
        ),
    }
    for request_class in ("short", "long"):
        for latency_name in ("ttft", "tpot", "e2e"):
            for percentile_name in ("p50", "p99"):
                fields[
                    f"{request_class}_{latency_name}_{percentile_name}_ms"
                ] = (
                    "metrics.latency."
                    f"{request_class}.{latency_name}.{percentile_name}",
                    1000.0,
                )
    rows = []
    for result in results:
        config = result.get("config", {})
        row = {
            "point_id": result.get("point_id"),
            "complete": result.get("complete", False),
            "git_commit": result.get("git_commit"),
            "experiment": config.get("experiment"),
            "arrival": config.get("arrival"),
            "variant": config.get("variant"),
            "run": config.get("run"),
            "request_rate": config.get("request_rate"),
            "max_concurrency": config.get("max_concurrency"),
            "completed": _optional_nested_value(result, "metrics.completed"),
            "failed": _optional_nested_value(result, "metrics.failed"),
            "latency_sample_requests": _optional_nested_value(
                result, "metrics.latency_sample_requests"
            ),
        }
        row.update({
            name: _optional_nested_value(result, path, scale)
            for name, (path, scale) in fields.items()
        })
        rows.append(row)
    return rows


def aggregate_results(
    results: list[dict],
    metric_path: str,
    *,
    scale: float = 1.0,
    unit: str | None = None,
) -> list[dict]:
    groups = {}
    for result in results:
        if not result.get("complete"):
            continue
        config = result["config"]
        key = (
            config["experiment"],
            config["arrival"],
            config.get("request_rate"),
            config.get("max_concurrency"),
            config["variant"],
        )
        try:
            value = _nested_value(result, metric_path)
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(value * scale)

    baseline_means = {}
    for key, values in groups.items():
        experiment, arrival, request_rate, max_concurrency, variant = key
        if variant == "baseline":
            baseline_means[(experiment, arrival, request_rate, max_concurrency)] = mean(values)

    rows = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        experiment, arrival, request_rate, max_concurrency, variant = key
        values = groups[key]
        value_mean = mean(values)
        baseline_mean = baseline_means.get(
            (experiment, arrival, request_rate, max_concurrency)
        )
        rows.append({
            "experiment": experiment,
            "arrival": arrival,
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "variant": variant,
            "metric": metric_path,
            "unit": unit,
            "runs": len(values),
            "mean": value_mean,
            "stddev": stdev(values) if len(values) > 1 else 0.0,
            "ratio_to_baseline": (
                value_mean / baseline_mean
                if baseline_mean not in (None, 0)
                else None
            ),
        })
    return rows
