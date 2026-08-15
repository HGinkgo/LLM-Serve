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
from benchmarks.gpu_clocks import GPUClockSampler
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


def _next_worker_endpoint(endpoint: str | None):
    if endpoint and endpoint.startswith("tcp://"):
        try:
            prefix, port = endpoint.rsplit(":", 1)
            return f"{prefix}:{int(port) + 1}"
        except (TypeError, ValueError):
            pass
    return "tcp://127.0.0.1:24432"


def _default_pd_engine_factory(model, **kwargs):
    from llmserve.pd import PDConfig, PDCoordinator, PDServingEngine

    distributed_init_method = kwargs.pop("distributed_init_method", None)
    prefill_init_method = kwargs.pop("prefill_init_method", None)
    decode_init_method = kwargs.pop("decode_init_method", None)
    if prefill_init_method is None:
        prefill_init_method = distributed_init_method or "tcp://127.0.0.1:24431"
    if decode_init_method is None:
        decode_init_method = _next_worker_endpoint(prefill_init_method)
    prefill_gpu = kwargs.pop("prefill_gpu", 0)
    decode_gpu = kwargs.pop("decode_gpu", 1)
    decode_gpus = tuple(kwargs.pop("decode_gpus", ()))
    decode_init_methods = tuple(kwargs.pop("decode_init_methods", ()))
    prefill_batch_size = kwargs.pop("prefill_batch_size", 1)
    prefill_enforce_eager = kwargs.pop("prefill_enforce_eager", True)
    decode_enforce_eager = kwargs.pop("decode_enforce_eager", True)
    kv_slot_count = kwargs.pop("kv_slot_count", 2)
    kv_slot_capacity_tokens = kwargs.pop("kv_slot_capacity_tokens", 1024)
    enable_transport_overlap = kwargs.pop("enable_pd_transport_overlap", False)
    coordinator = PDCoordinator(
        PDConfig(
            model=model,
            prefill_gpu=prefill_gpu,
            decode_gpu=decode_gpu,
            decode_gpus=decode_gpus,
            prefill_enforce_eager=prefill_enforce_eager,
            decode_enforce_eager=decode_enforce_eager,
            prefill_init_method=prefill_init_method,
            decode_init_method=decode_init_method,
            decode_init_methods=decode_init_methods,
            kv_slot_count=kv_slot_count,
            kv_slot_capacity_tokens=kv_slot_capacity_tokens,
            engine_kwargs=kwargs,
        )
    )
    return PDServingEngine(
        coordinator,
        prefill_batch_size=prefill_batch_size,
        enable_latency_telemetry=kwargs.get("enable_latency_telemetry", False),
        enable_transport_overlap=enable_transport_overlap,
    )


def _effective_runtime_config(engine):
    """Read selected values from the constructed runtime, not only the suite."""
    fields = (
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
        "gpu_memory_utilization",
        "enable_chunked_prefill",
        "enable_kv_capacity_admission",
        "enforce_eager",
        "enable_speculative_cuda_graph",
        "enable_latency_telemetry",
        "random_seed",
    )
    coordinator = getattr(engine, "coordinator", None)
    if coordinator is not None:
        config = coordinator.config
        return {
            "kind": "pd",
            "prefill_gpu": config.prefill_gpu,
            "decode_gpu": config.decode_gpu,
            "decode_gpus": list(config.decode_gpus),
            "decode_worker_ids": list(config.decode_worker_ids),
            "decode_init_methods": list(config.decode_init_methods),
            "prefill_enforce_eager": config.prefill_enforce_eager,
            "decode_enforce_eager": config.decode_enforce_eager,
            "kv_slot_count": config.kv_slot_count,
            "kv_slot_capacity_tokens": config.kv_slot_capacity_tokens,
            "prefill_batch_size": engine.prefill_batch_size,
            "enable_pd_transport_overlap": engine.enable_transport_overlap,
            "engine_kwargs": {
                name: config.engine_kwargs.get(name) for name in fields
            },
        }
    config = getattr(engine, "config", None)
    return {
        "kind": "collocated",
        "visible_gpu": 0,
        "engine_config": {name: getattr(config, name, None) for name in fields},
    }


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
    if runtime.get("pd", False) and enable_speculative:
        raise ValueError("PD benchmark currently requires speculative decoding to be disabled")

    engine_factory = engine_factory or (
        _default_pd_engine_factory
        if runtime.get("pd", False)
        else _default_engine_factory
    )
    make_sampling_params = make_sampling_params or _default_sampling_params
    active_speculative_model = speculative_model if enable_speculative else None
    engine_kwargs = {
        "enforce_eager": runtime.get("enforce_eager", True),
        "awq_backend": runtime.get("awq_backend", "cuda"),
        "enable_chunked_prefill": runtime.get("enable_chunked_prefill", False),
        "enable_kv_capacity_admission": runtime.get(
            "enable_kv_capacity_admission", False
        ),
        "max_model_len": runtime["max_model_len"],
        "max_num_batched_tokens": runtime["max_num_batched_tokens"],
        "max_num_seqs": runtime.get("max_num_seqs", 512),
        "gpu_memory_utilization": runtime.get(
            "gpu_memory_utilization", 0.9
        ),
        "speculative_model": active_speculative_model,
        "speculative_gamma": runtime.get("speculative_gamma", 3),
        "speculative_accept_mode": runtime.get(
            "speculative_accept_mode", "greedy"
        ),
        "speculative_trace": runtime.get("speculative_trace", False),
        "enable_speculative_cuda_graph": runtime.get(
            "enable_speculative_cuda_graph", False
        ),
        "enable_latency_telemetry": runtime.get(
            "enable_latency_telemetry", False
        ),
        "random_seed": runtime.get("random_seed"),
    }
    if runtime.get("pd", False):
        engine_kwargs.update(
            {
                "prefill_gpu": runtime.get("prefill_gpu", 0),
                "decode_gpu": runtime.get("decode_gpu", 1),
                "decode_gpus": tuple(runtime.get("decode_gpus", ())),
                "prefill_batch_size": runtime.get("prefill_batch_size", 1),
                "prefill_enforce_eager": runtime.get(
                    "prefill_enforce_eager", True
                ),
                "decode_enforce_eager": runtime.get(
                    "decode_enforce_eager", True
                ),
                "prefill_init_method": runtime.get("prefill_init_method"),
                "decode_init_method": runtime.get("decode_init_method"),
                "decode_init_methods": tuple(
                    runtime.get("decode_init_methods", ())
                ),
                "kv_slot_count": runtime.get("kv_slot_count", 2),
                "kv_slot_capacity_tokens": runtime.get(
                    "kv_slot_capacity_tokens", 1024
                ),
                "enable_pd_transport_overlap": runtime.get(
                    "enable_pd_transport_overlap", False
                ),
            }
        )
    if distributed_init_method is not None:
        engine_kwargs["distributed_init_method"] = distributed_init_method
    engine = engine_factory(model, **engine_kwargs)
    effective_runtime = _effective_runtime_config(engine)
    clock_sampler = (
        GPUClockSampler().start()
        if runtime.get("sample_gpu_clocks", False)
        else None
    )
    clock_summary = None
    if runtime.get("argmax_sampler", False) and hasattr(engine, "model_runner"):
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
        if clock_sampler is not None:
            clock_summary = clock_sampler.stop()
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
    metrics["kv_cache"] = observation["engine_summary"].get("kv_cache", {})
    metrics["pd"] = observation["engine_summary"].get("pd", {})
    metrics["cuda_graph"] = observation["engine_summary"].get(
        "cuda_graph", {}
    )

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
        "telemetry": {
            "scheduler_steps": observation.get("scheduler_steps", []),
            "gpu_clocks": clock_summary,
            "measurement_window": {
                "start": observation.get("measurement_start"),
                "end": observation.get("measurement_end"),
            },
            "effective_runtime": effective_runtime,
        },
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
