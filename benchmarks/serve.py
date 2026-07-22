import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from random import Random

from benchmarks.arrivals import poisson_arrival_times
from benchmarks.environment import (
    atomic_write_json,
    build_environment_metadata,
    discover_model_revision,
)
from benchmarks.metrics import (
    summarize_serving_run,
    summarize_speculative_requests,
    summarize_values,
)
from benchmarks.runtime import run_closed_loop, run_poisson
from benchmarks.schema import compact_request_record
from benchmarks.workloads import (
    WorkloadClass,
    RequestSpec,
    build_request_specs,
    iter_request_specs,
)


def _warmup_engine(engine, classes, make_sampling_params):
    rng = Random(0)
    for index, workload_class in enumerate(classes):
        output_len = min(workload_class.output_len, 8)
        spec = RequestSpec(
            request_id=-(index + 1),
            request_class=workload_class.name,
            input_len=workload_class.input_len,
            output_len=output_len,
            prompt_token_ids=tuple(
                rng.randint(0, 10000)
                for _ in range(workload_class.input_len)
            ),
        )
        engine.add_request(
            list(spec.prompt_token_ids),
            make_sampling_params(spec),
        )
    while not engine.is_finished():
        engine.step()
    engine.reset_metrics()


def _default_engine_factory(model, **kwargs):
    from llmserve import LLM

    return LLM(model, **kwargs)


def _default_sampling_params(spec):
    from llmserve import SamplingParams

    return SamplingParams(
        temperature=0.01,
        max_tokens=spec.output_len,
        ignore_eos=True,
    )


class ArgmaxSampler:
    def __call__(self, logits, temperatures):
        return logits.argmax(dim=-1)


def _workload_classes(point: dict):
    return [
        WorkloadClass(**workload_class)
        for workload_class in point["workload"]["classes"]
    ]


def _closed_loop_metrics(observation: dict, slo_ms):
    metrics = summarize_serving_run(
        observation["latency_requests"],
        observation["duration"],
        slo_ms=slo_ms,
    )
    completed_requests = [
        request
        for request in observation["requests"]
        if request["finish_time"] is not None
        and observation["measurement_start"] <= request["finish_time"]
        < observation["measurement_end"]
    ]
    input_tokens = sum(
        request["prompt_tokens"] for request in completed_requests
    )
    metrics["completed"] = observation["window_completed"]
    metrics["latency_sample_requests"] = len(
        observation["latency_requests"]
    )
    metrics["failed"] = sum(
        not request["success"] for request in completed_requests
    )
    metrics["throughput"] = {
        "requests_per_second": (
            observation["window_completed"] / observation["duration"]
        ),
        "input_tokens_per_second": input_tokens / observation["duration"],
        "output_tokens_per_second": (
            observation["window_output_tokens"] / observation["duration"]
        ),
        "total_tokens_per_second": (
            input_tokens + observation["window_output_tokens"]
        ) / observation["duration"],
    }
    return metrics


def run_point(
    point: dict,
    model: str,
    speculative_model: str | None = None,
    distributed_init_method: str | None = None,
    *,
    engine_factory=None,
    make_sampling_params=None,
    clock=time.perf_counter,
    sleep=time.sleep,
    metadata: dict | None = None,
    model_revision: str | None = None,
    speculative_model_revision: str | None = None,
):
    runtime = point["runtime"]
    enable_speculative = runtime.get("enable_speculative", False)
    if enable_speculative and not speculative_model:
        raise ValueError("speculative_model is required for speculative variants")

    engine_factory = engine_factory or _default_engine_factory
    make_sampling_params = make_sampling_params or _default_sampling_params
    active_speculative_model = speculative_model if enable_speculative else None
    engine_kwargs = {
        "enforce_eager": runtime.get("enforce_eager", True),
        "enable_chunked_prefill": runtime.get("enable_chunked_prefill", False),
        "max_model_len": runtime["max_model_len"],
        "max_num_batched_tokens": runtime["max_num_batched_tokens"],
        "speculative_model": active_speculative_model,
        "speculative_gamma": runtime.get("speculative_gamma", 3),
        "speculative_tree_nodes": runtime.get("speculative_tree_nodes", 0),
        "speculative_accept_mode": runtime.get(
            "speculative_accept_mode", "greedy"
        ),
        "speculative_trace": runtime.get("speculative_trace", False),
    }
    if distributed_init_method is not None:
        engine_kwargs["distributed_init_method"] = distributed_init_method
    engine = engine_factory(model, **engine_kwargs)
    if runtime.get("argmax_sampler", False):
        engine.model_runner.sampler = ArgmaxSampler()
    classes = _workload_classes(point)
    try:
        if point["arrival"] == "poisson":
            if runtime.get("warmup", False):
                _warmup_engine(engine, classes, make_sampling_params)
            specs = build_request_specs(
                classes,
                num_requests=point["num_requests"],
                seed=point["workload_seed"],
            )
            arrivals = poisson_arrival_times(
                num_requests=point["num_requests"],
                request_rate=point["request_rate"],
                seed=point["arrival_seed"],
            )
            observation = run_poisson(
                engine,
                specs,
                arrivals,
                make_sampling_params=make_sampling_params,
                clock=clock,
                sleep=sleep,
            )
            metrics = summarize_serving_run(
                observation["requests"],
                observation["duration"],
                slo_ms=point.get("slo_ms"),
            )
            metrics["offered_request_rate"] = point["request_rate"]
            metric_requests = observation["requests"]
        elif point["arrival"] == "closed-loop":
            observation = run_closed_loop(
                engine,
                iter_request_specs(classes, seed=point["workload_seed"]),
                max_concurrency=point["max_concurrency"],
                warmup_seconds=point["warmup_seconds"],
                measurement_seconds=point["measurement_seconds"],
                make_sampling_params=make_sampling_params,
                clock=clock,
            )
            metrics = _closed_loop_metrics(observation, point.get("slo_ms"))
            metrics["max_concurrency"] = point["max_concurrency"]
            metric_requests = observation["latency_requests"]
        else:
            raise ValueError(f"unsupported arrival: {point['arrival']}")
    finally:
        if hasattr(engine, "exit"):
            engine.exit()

    metrics["scheduled_batch_size"] = summarize_values(
        observation["scheduled_batch_sizes"]
    )
    metrics["speculative_batch_size"] = summarize_values(
        observation["speculative_batch_sizes"]
    )
    metrics["waiting_queue_size"] = summarize_values(
        observation["waiting_queue_sizes"]
    )
    metrics["running_queue_size"] = summarize_values(
        observation["running_queue_sizes"]
    )
    metrics["speculative"] = summarize_speculative_requests(metric_requests)
    engine_speculative = observation["engine_summary"].get("speculative", {})
    for name in ("batch_calls", "mean_batch_size", "max_batch_size"):
        if name in engine_speculative:
            metrics["speculative"][name] = engine_speculative[name]

    public_config = deepcopy(point)
    public_config["model"] = Path(model).name
    public_config["model_revision"] = model_revision
    public_config["speculative_model"] = (
        Path(active_speculative_model).name
        if active_speculative_model
        else None
    )
    public_config["speculative_model_revision"] = (
        speculative_model_revision if active_speculative_model else None
    )
    metadata = dict(metadata or {})
    complete = metrics["failed"] == 0
    return {
        "schema_version": 2,
        "complete": complete,
        "point_id": point["point_id"],
        "git_commit": metadata.get("git_commit"),
        "metadata": metadata,
        "config": public_config,
        "metrics": metrics,
        "requests": [
            compact_request_record(request)
            for request in observation["requests"]
        ],
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one in-process LLM-Serve benchmark point"
    )
    parser.add_argument("--point-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--speculative-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--speculative-model-revision")
    parser.add_argument("--expected-git-commit")
    parser.add_argument(
        "--distributed-init-method",
        default="tcp://localhost:2333",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _redact_error(message: str, paths):
    for path in paths:
        if path:
            message = message.replace(path, Path(path).name)
    return message


def _failure_result(point, metadata, error, redacted_paths=()):
    public_point = deepcopy(point)
    return {
        "schema_version": 2,
        "complete": False,
        "point_id": point.get("point_id"),
        "git_commit": metadata.get("git_commit"),
        "metadata": metadata,
        "config": public_point,
        "metrics": None,
        "requests": [],
        "error": {
            "type": type(error).__name__,
            "message": _redact_error(str(error), redacted_paths),
        },
    }


def main(argv=None):
    args = _parse_args(argv)
    point = json.loads(Path(args.point_config).read_text())
    metadata = build_environment_metadata()
    output_path = Path(args.output)
    if (
        args.expected_git_commit
        and metadata.get("git_commit") != args.expected_git_commit
    ):
        error = RuntimeError(
            "worker git commit does not match suite manifest"
        )
        atomic_write_json(output_path, _failure_result(point, metadata, error))
        return 1

    model = os.path.expanduser(args.model)
    speculative_model = (
        os.path.expanduser(args.speculative_model)
        if args.speculative_model
        else None
    )
    try:
        result = run_point(
            point,
            model=model,
            speculative_model=speculative_model,
            distributed_init_method=args.distributed_init_method,
            metadata=metadata,
            model_revision=(
                args.model_revision or discover_model_revision(model)
            ),
            speculative_model_revision=(
                args.speculative_model_revision
                or discover_model_revision(speculative_model)
                if speculative_model
                else None
            ),
        )
    except Exception as error:
        result = _failure_result(
            point, metadata, error, (model, speculative_model)
        )
        atomic_write_json(output_path, result)
        print(f"FAILED {point.get('point_id')}: {error}", file=sys.stderr)
        return 1

    atomic_write_json(output_path, result)
    throughput = result["metrics"]["throughput"]["output_tokens_per_second"]
    print(f"PASS {point['point_id']}: {throughput:.3f} output tok/s")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
