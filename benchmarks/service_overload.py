"""Service-level Poisson overload measurements for EngineServiceRuntime.

This module intentionally drives the service runtime directly instead of the
HTTP server.  It measures admission and event-stream behavior without adding
network-client variability; API 429 mapping is covered by service API tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from math import ceil
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Any

from benchmarks.arrivals import poisson_arrival_times
from benchmarks.environment import (
    atomic_write_json,
    build_environment_metadata,
    discover_model_revision,
)
from benchmarks.metrics import summarize_serving_run, summarize_values
from benchmarks.workloads import RequestSpec, WorkloadClass, build_request_specs
from llmserve.service.runtime import EngineServiceRuntime, ServiceOverloadedError


_TERMINAL_KINDS = frozenset({"completed", "cancelled", "failed", "timed_out"})


def parse_inflight_limit_variants(value: str) -> tuple[tuple[str, int | None], ...]:
    """Parse an explicit ``unbounded,64`` admission-control comparison."""
    variants = []
    seen = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "unbounded":
            name, limit = "unbounded", None
        else:
            try:
                limit = int(item)
            except ValueError as error:
                raise ValueError(
                    "inflight limits must be 'unbounded' or positive integers"
                ) from error
            if limit <= 0:
                raise ValueError("inflight limits must be positive integers")
            name = f"limit-{limit}"
        if name in seen:
            raise ValueError("inflight limit variants must be unique")
        seen.add(name)
        variants.append((name, limit))
    if not variants:
        raise ValueError("at least one inflight limit variant is required")
    return tuple(variants)


def run_service_poisson(
    runtime: Any,
    *,
    request_specs: Sequence[RequestSpec],
    arrival_times: Sequence[float],
    sampling_params_factory: Callable[[RequestSpec], Any],
    warmup_seconds: float,
    measurement_seconds: float,
    drain_timeout_seconds: float,
    snapshot_interval_seconds: float = 0.25,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = sleep,
) -> dict[str, Any]:
    """Run a fixed Poisson trace against one service runtime.

    Each scheduled trace arrival owns a daemon client submitter so a blocked
    admission cannot delay later arrivals. Latency starts immediately before
    that submitter calls ``runtime.submit`` and ends when the benchmark client
    observes output events. Arrival cohorts are defined by the fixed trace's
    scheduled arrival time; token and completion throughput are event counts
    observed inside the measurement interval.
    """
    if len(request_specs) != len(arrival_times):
        raise ValueError("request_specs and arrival_times must have the same length")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be non-negative")
    if measurement_seconds <= 0:
        raise ValueError("measurement_seconds must be positive")
    if drain_timeout_seconds < 0:
        raise ValueError("drain_timeout_seconds must be non-negative")
    if snapshot_interval_seconds <= 0:
        raise ValueError("snapshot_interval_seconds must be positive")
    if any(arrival < 0 for arrival in arrival_times):
        raise ValueError("arrival_times must be non-negative")
    if any(
        later < earlier
        for earlier, later in zip(arrival_times, arrival_times[1:])
    ):
        raise ValueError("arrival_times must be sorted")

    started_at = clock()
    measurement_started_at = started_at + warmup_seconds
    measurement_finished_at = measurement_started_at + measurement_seconds
    next_snapshot_at = started_at
    next_arrival_index = 0
    active: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    state_lock = Lock()
    submission_closed = Event()
    submission_threads: list[Thread] = []
    submission_errors: list[BaseException] = []
    queue_samples: list[dict[str, Any]] = []
    output_tokens_in_window = 0
    completed_in_window = 0
    measurement_outcomes = {
        kind: 0
        for kind in _TERMINAL_KINDS
    }
    drain_started_at: float | None = None

    def submit_request(
        spec: RequestSpec,
        scheduled_at: float,
        phase: str,
        record: dict[str, Any],
    ):
        submitted_at = clock()
        with state_lock:
            record["arrival_time"] = submitted_at
            record["arrival_lag"] = submitted_at - scheduled_at
        try:
            request = runtime.submit(
                list(spec.prompt_token_ids),
                sampling_params_factory(spec),
            )
        except ServiceOverloadedError:
            with state_lock:
                if not submission_closed.is_set():
                    record.update({
                        "status": "rejected",
                        "rejected_at": clock(),
                    })
        except BaseException as error:
            with state_lock:
                if not submission_closed.is_set():
                    record.update({
                        "status": "failed",
                        "finish_time": clock(),
                        "error": _exception_snapshot(error),
                    })
                    submission_errors.append(error)
        else:
            with state_lock:
                if not submission_closed.is_set():
                    record.update({
                        "admitted_at": clock(),
                        "admitted": True,
                        "request": request,
                    })
                    active[spec.request_id] = record

    while True:
        now = clock()
        if now >= next_snapshot_at:
            snapshot = dict(runtime.metrics_snapshot())
            snapshot_phase = (
                "warmup"
                if now < measurement_started_at
                else "measurement"
                if now < measurement_finished_at
                else "drain"
            )
            queue_samples.append({
                "observed_at": now,
                "phase": snapshot_phase,
                **snapshot,
            })
            next_snapshot_at = now + snapshot_interval_seconds

        while (
            next_arrival_index < len(arrival_times)
            and started_at + arrival_times[next_arrival_index] <= now
            and now < measurement_finished_at
        ):
            spec = request_specs[next_arrival_index]
            scheduled_at = started_at + arrival_times[next_arrival_index]
            phase = (
                "warmup"
                if scheduled_at < measurement_started_at
                else "measurement"
            )
            record = {
                "request_id": spec.request_id,
                "request_class": spec.request_class,
                "prompt_tokens": spec.input_len,
                "output_tokens": 0,
                "scheduled_arrival_time": scheduled_at,
                "arrival_lag": None,
                "arrival_time": None,
                "phase": phase,
                "status": "submitting",
                "admitted": False,
                "success": False,
                "cancelled": False,
                "token_times": [],
                "output_event_times": [],
                "measurement_output_tokens": 0,
            }
            with state_lock:
                records.append(record)
            submission_thread = Thread(
                target=submit_request,
                args=(spec, scheduled_at, phase, record),
                name=f"service-overload-submit-{spec.request_id}",
                daemon=True,
            )
            submission_threads.append(submission_thread)
            submission_thread.start()
            next_arrival_index += 1
            now = clock()

        with state_lock:
            active_records = list(active.items())
        for request_id, record in active_records:
            request = record["request"]
            while (event := request.poll_event()) is not None:
                observed_at = clock()
                if event.kind == "token":
                    token_count = len(event.token_ids)
                    if token_count == 0:
                        continue
                    record["output_tokens"] += token_count
                    record["token_times"].extend([observed_at] * token_count)
                    record["output_event_times"].append(observed_at)
                    record.setdefault("first_token_time", observed_at)
                    record["last_token_time"] = observed_at
                    if (
                        measurement_started_at <= observed_at
                        < measurement_finished_at
                    ):
                        output_tokens_in_window += token_count
                        record["measurement_output_tokens"] += token_count
                    continue
                if event.kind not in _TERMINAL_KINDS:
                    continue
                record.update({
                    "status": event.kind,
                    "finish_time": observed_at,
                    "success": event.kind == "completed",
                    "cancelled": event.kind in {"cancelled", "timed_out"},
                })
                if (
                    measurement_started_at <= observed_at < measurement_finished_at
                ):
                    measurement_outcomes[event.kind] += 1
                    if event.kind == "completed":
                        completed_in_window += 1
                with state_lock:
                    active.pop(request_id, None)
                break

        now = clock()
        measurement_finished = now >= measurement_finished_at
        if measurement_finished and drain_started_at is None:
            drain_started_at = now
        with state_lock:
            active_empty = not active
        submissions_pending = any(
            submission_thread.is_alive()
            for submission_thread in submission_threads
        )
        if not active_empty and measurement_finished and drain_started_at is None:
            drain_started_at = now
        if not submissions_pending and active_empty and measurement_finished:
            break
        if (
            drain_started_at is not None
            and now - drain_started_at >= drain_timeout_seconds
        ):
            break

        next_arrival_at = (
            started_at + arrival_times[next_arrival_index]
            if next_arrival_index < len(arrival_times)
            else None
        )
        wait_seconds = 0.002
        if next_arrival_at is not None:
            wait_seconds = min(wait_seconds, max(0.0, next_arrival_at - now))
        sleeper(wait_seconds)

    submission_closed.set()
    with state_lock:
        active_records = list(active.values())
    for record in active_records:
        try:
            runtime.cancel(record["request"])
        except Exception:
            pass
        record.update({
            "status": "unfinished",
            "success": False,
            "cancelled": True,
            "finish_time": clock(),
        })
    with state_lock:
        active.clear()
        for record in records:
            if record["status"] == "submitting":
                record.update({
                    "status": "unfinished",
                    "success": False,
                    "cancelled": True,
                    "finish_time": clock(),
                })

    final_snapshot = dict(runtime.metrics_snapshot())
    final_observed_at = clock()
    final_snapshot_phase = (
        "warmup"
        if final_observed_at < measurement_started_at
        else "measurement"
        if final_observed_at < measurement_finished_at
        else "drain"
    )
    queue_samples.append({
        "observed_at": final_observed_at,
        "phase": final_snapshot_phase,
        **final_snapshot,
    })
    arrival_cohort = [
        record
        for record in records
        if record["phase"] == "measurement"
    ]
    latency_cohort = [
        record for record in arrival_cohort if record["status"] == "completed"
    ]
    metrics = summarize_serving_run(
        latency_cohort,
        measurement_seconds,
        include_e2e=False,
        include_auxiliary_latency=False,
    )
    metrics.pop("goodput", None)
    metrics["throughput"] = {
        "requests_per_second": completed_in_window / measurement_seconds,
        "output_tokens_per_second": output_tokens_in_window / measurement_seconds,
    }
    admission = {
        "offered": len(records),
        "accepted": sum(record["admitted"] for record in records),
        "rejected": sum(record["status"] == "rejected" for record in records),
        "measurement_offered": sum(
            record["phase"] == "measurement" for record in records
        ),
        "measurement_accepted": sum(
            record["phase"] == "measurement" and record["admitted"]
            for record in records
        ),
        "measurement_rejected": sum(
            record["phase"] == "measurement" and record["status"] == "rejected"
            for record in records
        ),
        "unfinished_after_drain": sum(
            record["status"] == "unfinished" for record in records
        ),
    }
    measurement_queue_samples = [
        sample for sample in queue_samples if sample["phase"] == "measurement"
    ]
    queue_depth = {
        "window": "measurement",
        "sample_interval_seconds": snapshot_interval_seconds,
        "samples": len(measurement_queue_samples),
        "raw_samples": len(queue_samples),
        "waiting": summarize_values([
            float(sample.get("queue_waiting", 0))
            for sample in measurement_queue_samples
        ]),
        "running": summarize_values([
            float(sample.get("queue_running", 0))
            for sample in measurement_queue_samples
        ]),
        "inflight": summarize_values([
            float(sample.get("inflight_requests", 0))
            for sample in measurement_queue_samples
        ]),
        "admission_reserved": summarize_values([
            float(sample.get("admission_reserved", 0))
            for sample in measurement_queue_samples
        ]),
    }
    for record in records:
        record.pop("request", None)
    if submission_errors:
        raise submission_errors[0]
    return {
        "measurement": {
            "warmup_seconds": warmup_seconds,
            "measurement_seconds": measurement_seconds,
            "drain_timeout_seconds": drain_timeout_seconds,
            "started_at": started_at,
            "measurement_started_at": measurement_started_at,
            "measurement_finished_at": measurement_finished_at,
        },
        "admission": admission,
        "outcomes": {
            "measurement_window": measurement_outcomes,
            "arrival_cohort": {
                kind: sum(record["status"] == kind for record in arrival_cohort)
                for kind in _TERMINAL_KINDS
            },
            "arrival_cohort_unfinished_after_drain": sum(
                record["status"] == "unfinished" for record in arrival_cohort
            ),
        },
        "metrics": metrics,
        "queue_depth": queue_depth,
        "service_metrics_final": final_snapshot,
        "requests": records,
        "queue_samples": queue_samples,
    }


def _parse_request_rates(value: str) -> tuple[float, ...]:
    try:
        rates = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("request rates must be comma-separated numbers") from error
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("request rates must be positive")
    return rates


def _trace_for_duration(
    *,
    request_rate: float,
    duration_seconds: float,
    seed: int,
) -> tuple[float, ...]:
    request_count = max(64, int(ceil(request_rate * duration_seconds * 2)) + 32)
    while True:
        arrivals = poisson_arrival_times(request_count, request_rate, seed)
        if arrivals[-1] >= duration_seconds:
            return tuple(arrival for arrival in arrivals if arrival < duration_seconds)
        request_count *= 2


def _request_trace_hash(
    request_specs: Sequence[RequestSpec],
    arrival_times: Sequence[float],
) -> str:
    payload = {
        "arrivals": list(arrival_times),
        "requests": [
            {
                "request_id": spec.request_id,
                "request_class": spec.request_class,
                "input_len": spec.input_len,
                "output_len": spec.output_len,
                "prompt_token_ids": list(spec.prompt_token_ids),
            }
            for spec in request_specs
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _summary_value(mapping: Mapping[str, Any], path: Sequence[str]):
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _milliseconds(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    observation = result["observation"]
    metrics = observation["metrics"]
    latency = metrics["latency"].get("overall", {})
    outcomes = observation.get("outcomes", {})
    window_outcomes = outcomes.get("measurement_window", {})
    cohort_outcomes = outcomes.get("arrival_cohort", {})
    return {
        "point_id": result["point_id"],
        "request_rate": result["config"]["request_rate"],
        "variant": result["config"]["variant"],
        "inflight_limit": result["config"]["max_inflight_requests"],
        "trace_hash": result["trace_hash"],
        "offered": observation["admission"]["measurement_offered"],
        "accepted": observation["admission"]["measurement_accepted"],
        "rejected": observation["admission"]["measurement_rejected"],
        "completed_in_window": window_outcomes.get("completed"),
        "timed_out_in_window": window_outcomes.get("timed_out"),
        "completed_cohort": metrics["completed"],
        "timed_out_cohort": cohort_outcomes.get("timed_out"),
        "unfinished_after_drain": observation["admission"]["unfinished_after_drain"],
        "request_throughput_rps": metrics["throughput"]["requests_per_second"],
        "output_throughput_tps": metrics["throughput"]["output_tokens_per_second"],
        "ttft_p50_ms": _milliseconds(_summary_value(latency, ("ttft", "p50"))),
        "ttft_p99_ms": _milliseconds(_summary_value(latency, ("ttft", "p99"))),
        "tpot_p50_ms": _milliseconds(_summary_value(latency, ("tpot", "p50"))),
        "tpot_p99_ms": _milliseconds(_summary_value(latency, ("tpot", "p99"))),
        "queue_waiting_p99": _summary_value(
            observation, ("queue_depth", "waiting", "p99")
        ),
        "queue_running_p99": _summary_value(
            observation, ("queue_depth", "running", "p99")
        ),
        "service_inflight_p99": _summary_value(
            observation, ("queue_depth", "inflight", "p99")
        ),
    }


def _build_service_runtime(args: argparse.Namespace, inflight_limit: int | None):
    from transformers import AutoTokenizer

    from llmserve.service.factory import ServiceLaunchConfig, build_engine_factory

    config = ServiceLaunchConfig(
        model=args.model,
        mode="collocated",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_kv_capacity_admission=True,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return EngineServiceRuntime(
        build_engine_factory(config),
        tokenizer=tokenizer,
        max_inflight_requests=inflight_limit,
    ), config


def _exception_snapshot(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _load_trace(
    trace_path: Path,
    expected_hash: str,
) -> tuple[tuple[RequestSpec, ...], tuple[float, ...]]:
    """Load a persisted trace and reject a changed workload before execution."""
    with Path(trace_path).open() as trace_file:
        trace = json.load(trace_file)
    request_specs = tuple(
        RequestSpec(
            request_id=int(request["request_id"]),
            request_class=str(request["request_class"]),
            input_len=int(request["input_len"]),
            output_len=int(request["output_len"]),
            prompt_token_ids=tuple(int(token) for token in request["prompt_token_ids"]),
        )
        for request in trace["requests"]
    )
    arrival_times = tuple(float(arrival) for arrival in trace["arrival_times"])
    trace_hash = _request_trace_hash(request_specs, arrival_times)
    if trace.get("trace_hash") != trace_hash or trace_hash != expected_hash:
        raise RuntimeError(f"trace hash mismatch: {trace_path}")
    return request_specs, arrival_times


def _point_config(
    args: argparse.Namespace,
    *,
    point_id: str,
    variant_name: str,
    inflight_limit: int | None,
    request_rate: float,
    trace: Mapping[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "point_id": point_id,
        "model": args.model,
        "trace_path": str(Path(trace["path"]).resolve()),
        "trace_hash": trace["hash"],
        "result_path": str(result_path.resolve()),
        "config": {
            "variant": variant_name,
            "max_inflight_requests": inflight_limit,
            "request_rate": request_rate,
            "trace_seed": trace["seed"],
            "input_len": args.input_len,
            "output_len": args.output_len,
            "temperature": 0.01,
            "ignore_eos": True,
            "warmup_seconds": args.warmup_seconds,
            "measurement_seconds": args.measurement_seconds,
            "drain_timeout_seconds": args.drain_timeout_seconds,
            "snapshot_interval_seconds": args.snapshot_interval_seconds,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_chunked_prefill": True,
            "enable_kv_capacity_admission": True,
            "enforce_eager": args.enforce_eager,
        },
    }


def execute_overload_point(point: Mapping[str, Any]) -> dict[str, Any]:
    """Execute exactly one point in the worker process that owns its CUDA state."""
    from llmserve.sampling_params import SamplingParams

    config = dict(point["config"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "point_id": point["point_id"],
        "metadata": build_environment_metadata(),
        "model": point["model"],
        "trace_hash": point["trace_hash"],
        "trace_path": point["trace_path"],
        "config": config,
        "runtime_fatal_error": None,
    }
    runtime = None
    try:
        request_specs, arrival_times = _load_trace(
            Path(point["trace_path"]),
            str(point["trace_hash"]),
        )
        runtime_args = argparse.Namespace(
            model=point["model"],
            max_model_len=config["max_model_len"],
            max_num_batched_tokens=config["max_num_batched_tokens"],
            max_num_seqs=config["max_num_seqs"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            enforce_eager=config["enforce_eager"],
        )
        runtime, _ = _build_service_runtime(
            runtime_args,
            config["max_inflight_requests"],
        )
        runtime.start()
        result["observation"] = run_service_poisson(
            runtime,
            request_specs=request_specs,
            arrival_times=arrival_times,
            sampling_params_factory=lambda spec: SamplingParams(
                temperature=config["temperature"],
                max_tokens=spec.output_len,
                ignore_eos=config["ignore_eos"],
            ),
            warmup_seconds=config["warmup_seconds"],
            measurement_seconds=config["measurement_seconds"],
            drain_timeout_seconds=config["drain_timeout_seconds"],
            snapshot_interval_seconds=config["snapshot_interval_seconds"],
        )
        result["complete"] = True
    except Exception as error:
        result["error"] = _exception_snapshot(error)
    finally:
        if runtime is not None:
            fatal_error = runtime.fatal_error
            if fatal_error is not None:
                result["runtime_fatal_error"] = _exception_snapshot(fatal_error)
            try:
                runtime.close()
            except Exception as error:
                if "error" not in result:
                    result["error"] = _exception_snapshot(error)
                    result["complete"] = False
    return result


def _run_point_worker(
    point_config_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Launch a fresh Python process so CUDA allocator state cannot leak across points."""
    return runner(
        [
            sys.executable,
            "-m",
            "benchmarks.service_overload",
            "--worker-config",
            str(Path(point_config_path).resolve()),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


def _execute_worker(point_config_path: Path) -> int:
    with Path(point_config_path).open() as config_file:
        point = json.load(config_file)
    result_path = Path(point["result_path"])
    try:
        result = execute_overload_point(point)
    except Exception as error:
        result = {
            "schema_version": 1,
            "complete": False,
            "point_id": point.get("point_id"),
            "trace_hash": point.get("trace_hash"),
            "config": point.get("config", {}),
            "error": _exception_snapshot(error),
        }
    atomic_write_json(result_path, result)
    return 0 if result["complete"] else 1


def execute_overload_scan(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_environment_metadata()
    if metadata["git_dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing dirty worktree; pass --allow-dirty to override")
    rates = _parse_request_rates(args.request_rates)
    variants = parse_inflight_limit_variants(args.inflight_limits)
    durations = args.warmup_seconds + args.measurement_seconds
    points = [
        (rate, variant_name, limit)
        for rate in rates
        for variant_name, limit in variants
    ]
    trace_dir = output_dir / "traces"
    traces_by_rate = {}
    for rate in rates:
        trace_seed = args.trace_seed + rates.index(rate)
        arrivals = _trace_for_duration(
            request_rate=rate,
            duration_seconds=durations,
            seed=trace_seed,
        )
        request_specs = build_request_specs(
            (WorkloadClass("short", 1, args.input_len, args.output_len),),
            len(arrivals),
            trace_seed,
        )
        trace_hash = _request_trace_hash(request_specs, arrivals)
        trace_path = trace_dir / f"rate-{rate:g}-seed-{trace_seed}.json"
        if trace_path.exists():
            with trace_path.open() as trace_file:
                existing_trace = json.load(trace_file)
            if existing_trace.get("trace_hash") != trace_hash:
                raise RuntimeError(
                    f"existing trace does not match regenerated trace: {trace_path}"
                )
        else:
            atomic_write_json(trace_path, {
                "request_rate": rate,
                "seed": trace_seed,
                "trace_hash": trace_hash,
                "arrival_times": list(arrivals),
                "requests": [
                    {
                        "request_id": spec.request_id,
                        "request_class": spec.request_class,
                        "input_len": spec.input_len,
                        "output_len": spec.output_len,
                        "prompt_token_ids": list(spec.prompt_token_ids),
                    }
                    for spec in request_specs
                ],
            })
        traces_by_rate[rate] = {
            "arrivals": arrivals,
            "request_specs": request_specs,
            "seed": trace_seed,
            "hash": trace_hash,
            "path": trace_path,
        }
    manifest = {
        "schema_version": 1,
        "complete": False,
        "metadata": metadata,
        "model": Path(args.model).name,
        "model_revision": discover_model_revision(args.model),
        "mode": "collocated",
        "client_path": "EngineServiceRuntime direct event polling; HTTP API mapping is covered by unit tests",
        "execution_model": "one isolated Python worker process per rate and admission variant",
        "total_points": len(points),
        "completed_points": 0,
        "failed_points": 0,
        "request_rates": rates,
        "inflight_limit_variants": [
            {"name": name, "limit": limit} for name, limit in variants
        ],
        "traces": [
            {
                "request_rate": rate,
                "seed": traces_by_rate[rate]["seed"],
                "trace_hash": traces_by_rate[rate]["hash"],
                "trace_path": str(traces_by_rate[rate]["path"].relative_to(output_dir)),
            }
            for rate in rates
        ],
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    results_dir = output_dir / "runs"
    result_rows = []
    for point_index, (rate, variant_name, limit) in enumerate(points):
        trace = traces_by_rate[rate]
        point_id = f"service-overload-{variant_name}-rate-{rate:g}"
        result_path = results_dir / f"{point_id}.json"
        point = _point_config(
            args,
            point_id=point_id,
            variant_name=variant_name,
            inflight_limit=limit,
            request_rate=rate,
            trace=trace,
            result_path=result_path,
        )
        point_config_path = results_dir / f"{point_id}.config.json"
        atomic_write_json(point_config_path, point)
        worker = _run_point_worker(point_config_path)
        if result_path.exists():
            with result_path.open() as result_file:
                result = json.load(result_file)
        else:
            result = {
                "schema_version": 1,
                "complete": False,
                "point_id": point_id,
                "trace_hash": trace["hash"],
                "config": point["config"],
                "error": {
                    "type": "WorkerProcessError",
                    "message": "worker exited without writing a result: "
                    f"returncode={worker.returncode}; stderr={worker.stderr[-2000:]}",
                },
            }
            atomic_write_json(result_path, result)
        if worker.returncode != 0 and result.get("complete"):
            result["complete"] = False
            result["error"] = {
                "type": "WorkerProcessError",
                "message": "worker reported success data but exited non-zero: "
                f"returncode={worker.returncode}; stderr={worker.stderr[-2000:]}",
            }
            atomic_write_json(result_path, result)
        if result["complete"]:
            result_rows.append(_summary_row(result))
            manifest["completed_points"] += 1
        else:
            manifest["failed_points"] += 1
        atomic_write_json(output_dir / "manifest.json", manifest)
        print(
            f"[{point_index + 1}/{len(points)}] {point_id}: "
            f"{'ok' if result['complete'] else 'failed'}",
            flush=True,
        )
    _write_csv(output_dir / "summary.csv", result_rows)
    manifest["complete"] = manifest["failed_points"] == 0
    atomic_write_json(output_dir / "manifest.json", manifest)
    return 0 if manifest["complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a service-level Poisson overload scan on one Collocated Runtime."
    )
    parser.add_argument("--model", required=True, help="local Qwen model directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-rates", default="24")
    parser.add_argument("--inflight-limits", default="unbounded,64")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--measurement-seconds", type=float, default=20.0)
    parser.add_argument("--drain-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=0.25)
    parser.add_argument("--trace-seed", type=int, default=20260818)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    worker_parser = argparse.ArgumentParser(add_help=False)
    worker_parser.add_argument("--worker-config")
    worker_args, remaining = worker_parser.parse_known_args(argv)
    if worker_args.worker_config is not None:
        if remaining:
            raise ValueError("--worker-config cannot be combined with benchmark options")
        return _execute_worker(Path(worker_args.worker_config))

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.input_len <= 0 or args.output_len <= 0:
            raise ValueError("input and output lengths must be positive")
        if args.max_model_len <= 0 or args.max_num_batched_tokens <= 0:
            raise ValueError("model length and token budget must be positive")
        if args.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if not 0 < args.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        return execute_overload_scan(args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
