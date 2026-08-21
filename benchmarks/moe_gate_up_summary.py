"""Create an auditable A/B summary from MoE Gate/Up raw benchmark reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path

from benchmarks.environment import atomic_write_json
from benchmarks.metrics import summarize_values
from benchmarks.moe_baseline import build_vllm_marlin_baseline


_FUSION_FLAG = "enable_moe_gate_up_fusion"


def _require_equal(control: object, optimized: object, *, name: str) -> None:
    if control != optimized:
        raise ValueError(f"control and optimized {name} must match")


def _runtime_contract(report: Mapping) -> dict:
    config = dict(report["runtime_config"])
    config.pop(_FUSION_FLAG, None)
    return config


def _validate_reports(control: Mapping, optimized: Mapping) -> None:
    _require_equal(control.get("model"), optimized.get("model"), name="model")
    control_input = control.get("input", {})
    optimized_input = optimized.get("input", {})
    _require_equal(
        control_input.get("prompt_token_sha256"),
        optimized_input.get("prompt_token_sha256"),
        name="prompt token trace",
    )
    _require_equal(
        control_input.get("cases"), optimized_input.get("cases"), name="prompt cases"
    )
    _require_equal(
        _runtime_contract(control),
        _runtime_contract(optimized),
        name="runtime configuration",
    )
    if control.get("runtime_config", {}).get(_FUSION_FLAG) is not False:
        raise ValueError("control report must disable Gate/Up fusion")
    if optimized.get("runtime_config", {}).get(_FUSION_FLAG) is not True:
        raise ValueError("optimized report must enable Gate/Up fusion")


def _group_results(report: Mapping) -> dict[tuple[str, int], list[Mapping]]:
    grouped: dict[tuple[str, int], list[Mapping]] = defaultdict(list)
    for result in report.get("results", []):
        backend = result.get("backend")
        batch_size = result.get("batch_size")
        if not isinstance(backend, str) or not isinstance(batch_size, int):
            raise ValueError("each benchmark result needs backend and batch_size")
        grouped[(backend, batch_size)].append(result)
    if not grouped:
        raise ValueError("benchmark report has no results")
    return dict(grouped)


def _aggregate(rows: Sequence[Mapping]) -> dict:
    wall_time = sum(float(row.get("wall_time_seconds", 0.0)) for row in rows)
    if wall_time <= 0:
        raise ValueError("benchmark result wall_time_seconds must be positive")
    request_count = sum(int(row.get("request_count", 0)) for row in rows)
    completed = sum(int(row.get("completed", 0)) for row in rows)
    failed = sum(int(row.get("failed", 0)) for row in rows)
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in rows)
    requests = [
        request
        for row in rows
        for request in row.get("requests", [])
    ]
    ttft_ms = [
        float(request["ttft_ms"])
        for request in requests
        if request.get("ttft_ms") is not None
    ]
    tpot_ms = [
        float(request["tpot_ms"])
        for request in requests
        if request.get("tpot_ms") is not None
    ]
    return {
        "run_count": len(rows),
        "request_count": request_count,
        "completed": completed,
        "failed": failed,
        "output_tokens": output_tokens,
        "wall_time_seconds": wall_time,
        "request_throughput": completed / wall_time,
        "output_throughput": output_tokens / wall_time,
        "ttft_ms": summarize_values(ttft_ms),
        "tpot_ms": summarize_values(tpot_ms),
        "peak_allocated_bytes": max(
            int(row.get("peak_allocated_bytes", 0)) for row in rows
        ),
        "peak_reserved_bytes": max(
            int(row.get("peak_reserved_bytes", 0)) for row in rows
        ),
    }


def _percent_change(control: float | int | None, optimized: float | int | None):
    if control in (None, 0) or optimized is None:
        return None
    return (optimized / control - 1.0) * 100.0


def _comparison_delta(control: Mapping, optimized: Mapping) -> dict:
    return {
        "output_throughput_percent": _percent_change(
            control["output_throughput"], optimized["output_throughput"]
        ),
        "request_throughput_percent": _percent_change(
            control["request_throughput"], optimized["request_throughput"]
        ),
        "ttft_p50_percent": _percent_change(
            control["ttft_ms"]["p50"], optimized["ttft_ms"]["p50"]
        ),
        "ttft_p99_percent": _percent_change(
            control["ttft_ms"]["p99"], optimized["ttft_ms"]["p99"]
        ),
        "tpot_p50_percent": _percent_change(
            control["tpot_ms"]["p50"], optimized["tpot_ms"]["p50"]
        ),
        "tpot_p99_percent": _percent_change(
            control["tpot_ms"]["p99"], optimized["tpot_ms"]["p99"]
        ),
        "peak_allocated_bytes_percent": _percent_change(
            control["peak_allocated_bytes"], optimized["peak_allocated_bytes"]
        ),
        "peak_reserved_bytes_percent": _percent_change(
            control["peak_reserved_bytes"], optimized["peak_reserved_bytes"]
        ),
    }


def summarize_ab_report(*, control: Mapping, optimized: Mapping) -> dict:
    """Aggregate matching unfused and fused raw reports without mutating them."""
    _validate_reports(control, optimized)
    control_groups = _group_results(control)
    optimized_groups = _group_results(optimized)
    _require_equal(
        set(control_groups), set(optimized_groups), name="backend and batch-size points"
    )
    if not any(backend == "marlin" for backend, _ in control_groups):
        raise ValueError("MoE A/B summary requires a Marlin measurement point")

    config = control["runtime_config"]
    comparisons = []
    for backend, batch_size in sorted(control_groups):
        control_summary = _aggregate(control_groups[(backend, batch_size)])
        optimized_summary = _aggregate(optimized_groups[(backend, batch_size)])
        comparisons.append({
            "backend": backend,
            "batch_size": batch_size,
            "control": control_summary,
            "optimized": optimized_summary,
            "delta": _comparison_delta(control_summary, optimized_summary),
        })

    return {
        "schema_version": 1,
        "report_kind": "moe_gate_up_fusion_ab_summary",
        "external_baseline": build_vllm_marlin_baseline(
            source=config.get("marlin_library"),
            role="external_reference_and_token_baseline",
        ),
        "candidate_ab_control": {
            "runtime": "llmserve",
            "enable_moe_gate_up_fusion": False,
        },
        "candidate_ab_optimized": {
            "runtime": "llmserve",
            "enable_moe_gate_up_fusion": True,
        },
        "metadata": dict(control.get("metadata", {})),
        "workload": {
            "model": control.get("model"),
            "case_count": control["input"].get("case_count"),
            "prompt_token_sha256": control["input"].get("prompt_token_sha256"),
            "cases": control["input"].get("cases", []),
            "runtime_config": _runtime_contract(control),
        },
        "aggregation": {
            "throughput": "sum(output_tokens or completed) / sum(wall_time_seconds)",
            "latency": "percentiles over all compact request summaries across repeats",
            "peak_memory": "maximum CUDA peak across repeats",
        },
        "comparisons": comparisons,
    }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def run_summary(args: argparse.Namespace) -> dict:
    control_path = Path(args.control).expanduser().resolve()
    optimized_path = Path(args.optimized).expanduser().resolve()
    if not control_path.is_file():
        raise ValueError(f"control report does not exist: {control_path}")
    if not optimized_path.is_file():
        raise ValueError(f"optimized report does not exist: {optimized_path}")
    report = summarize_ab_report(
        control=_load_json(control_path),
        optimized=_load_json(optimized_path),
    )
    report["sources"] = {
        "control": _source_record(control_path),
        "optimized": _source_record(optimized_path),
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_summary(args)
    atomic_write_json(args.output, report)
    print(json.dumps({"comparison_count": len(report["comparisons"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
