import argparse
import copy
import csv
import json
import os
from pathlib import Path


CSV_FIELDS = [
    "source",
    "git_commit",
    "git_dirty",
    "generated_at_utc",
    "python_version",
    "torch_version",
    "cuda_runtime",
    "gpu_name",
    "model",
    "model_revision",
    "speculative_model",
    "speculative_model_revision",
    "workload_name",
    "mode",
    "metric_scope",
    "arrival",
    "num_requests",
    "max_concurrency",
    "input_len",
    "long_input_len",
    "injection_delay",
    "injected_request_index",
    "output_len",
    "gamma",
    "chunked_prefill",
    "throughput_tok_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "burst_itl_p50_ms",
    "burst_itl_p99_ms",
    "output_event_latency_p50_ms",
    "output_event_latency_p99_ms",
    "speculative_step_latency_p50_ms",
    "speculative_step_latency_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "request_latency_p50_ms",
    "request_latency_p99_ms",
    "acceptance_rate",
    "acceptance_length",
    "draft_proposal_mean_ms",
    "target_verify_mean_ms",
    "kv_update_mean_ms",
    "speculative_total_mean_ms",
]


def _public_path(value: str | None, *, keep_remote_id: bool = True):
    if not value:
        return value
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded) or value.startswith(("~", ".")):
        return Path(expanded).name
    if keep_remote_id:
        return value
    return Path(value).name


def sanitize_result(result: dict) -> dict:
    sanitized = copy.deepcopy(result)
    config = sanitized.get("config", {})
    config["model"] = _public_path(config.get("model"))
    config["speculative_model"] = _public_path(config.get("speculative_model"))
    config["prompt_file"] = _public_path(
        config.get("prompt_file"),
        keep_remote_id=False,
    )
    return sanitized


def _stat_ms(summary: dict, name: str, percentile: str):
    stats = summary.get(name)
    if stats is None and name == "burst_itl":
        stats = summary.get("itl")
    if not stats or stats.get(percentile) is None:
        return ""
    return float(stats[percentile]) * 1000


def _timing_mean_ms(speculative: dict, name: str):
    timing = speculative.get("timing", {}).get(name)
    if not timing or timing.get("mean") is None:
        return ""
    return float(timing["mean"]) * 1000


def build_row(source: Path, result: dict) -> dict:
    config = sanitize_result(result).get("config", {})
    metadata = result.get("metadata", {})
    metrics = result.get("metrics", {})
    if "steady_state" in metrics:
        summary = metrics["steady_state"]
        metric_scope = "steady_state"
    else:
        summary = metrics.get("summary", {})
        metric_scope = "overall"
    speculative = summary.get("speculative", {})
    speculative_model = config.get("speculative_model")

    return {
        "source": source.name,
        "git_commit": metadata.get("git_commit", ""),
        "git_dirty": metadata.get("git_dirty", ""),
        "generated_at_utc": metadata.get("generated_at_utc", ""),
        "python_version": metadata.get("python_version", ""),
        "torch_version": metadata.get("torch_version", ""),
        "cuda_runtime": metadata.get("cuda_runtime", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "model": config.get("model", ""),
        "model_revision": config.get("model_revision", ""),
        "speculative_model": speculative_model or "",
        "speculative_model_revision": config.get(
            "speculative_model_revision", ""
        ),
        "workload_name": config.get("workload_name", ""),
        "mode": "eagle" if speculative_model else "baseline",
        "metric_scope": metric_scope,
        "arrival": config.get("arrival", ""),
        "num_requests": config.get("num_requests", ""),
        "max_concurrency": config.get("max_concurrency", ""),
        "input_len": config.get("input_len", ""),
        "long_input_len": config.get("long_input_len", ""),
        "injection_delay": config.get("injection_delay", ""),
        "injected_request_index": config.get("injected_request_index", ""),
        "output_len": config.get("output_len", ""),
        "gamma": config.get("speculative_gamma", ""),
        "chunked_prefill": config.get("enable_chunked_prefill", False),
        "throughput_tok_s": summary.get("throughput", ""),
        "ttft_p50_ms": _stat_ms(summary, "ttft", "p50"),
        "ttft_p99_ms": _stat_ms(summary, "ttft", "p99"),
        "burst_itl_p50_ms": _stat_ms(summary, "burst_itl", "p50"),
        "burst_itl_p99_ms": _stat_ms(summary, "burst_itl", "p99"),
        "output_event_latency_p50_ms": _stat_ms(
            summary, "output_event_latency", "p50"
        ),
        "output_event_latency_p99_ms": _stat_ms(
            summary, "output_event_latency", "p99"
        ),
        "speculative_step_latency_p50_ms": _stat_ms(
            summary, "speculative_step_latency", "p50"
        ),
        "speculative_step_latency_p99_ms": _stat_ms(
            summary, "speculative_step_latency", "p99"
        ),
        "tpot_p50_ms": _stat_ms(summary, "tpot", "p50"),
        "tpot_p99_ms": _stat_ms(summary, "tpot", "p99"),
        "request_latency_p50_ms": _stat_ms(summary, "request_latency", "p50"),
        "request_latency_p99_ms": _stat_ms(summary, "request_latency", "p99"),
        "acceptance_rate": speculative.get("acceptance_rate", ""),
        "acceptance_length": speculative.get("acceptance_length", ""),
        "draft_proposal_mean_ms": _timing_mean_ms(
            speculative, "draft_proposal_time"
        ),
        "target_verify_mean_ms": _timing_mean_ms(
            speculative, "target_verify_time"
        ),
        "kv_update_mean_ms": _timing_mean_ms(speculative, "kv_update_time"),
        "speculative_total_mean_ms": _timing_mean_ms(speculative, "total_time"),
    }


def load_result(path: Path) -> dict:
    with path.open() as f:
        result = json.load(f)
    if "config" not in result or "metrics" not in result:
        raise ValueError(f"{path} is not a bench_serving result")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and sanitize LLM-Serve benchmark JSON files"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--representative-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    loaded = [(path, load_result(path)) for path in args.inputs]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(build_row(path, result) for path, result in loaded)

    if args.representative_dir is not None:
        args.representative_dir.mkdir(parents=True, exist_ok=True)
        names = [path.name for path, _ in loaded]
        if len(names) != len(set(names)):
            raise ValueError("representative JSON inputs must have unique file names")
        for path, result in loaded:
            output_path = args.representative_dir / path.name
            with output_path.open("w") as f:
                json.dump(sanitize_result(result), f, indent=2)

    print(f"Wrote {len(loaded)} rows to {args.csv}")


if __name__ == "__main__":
    main()
