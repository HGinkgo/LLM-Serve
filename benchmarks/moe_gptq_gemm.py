"""Microbenchmark the production GPTQ linear path at fixed M values."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import torch
from safetensors.torch import safe_open

from benchmarks.environment import atomic_write_json, build_environment_metadata
from benchmarks.metrics import summarize_values
from benchmarks.moe_backend import SUPPORTED_BACKENDS, parse_backends
from llmserve.layers.quantized import GPTQLinear


DEFAULT_M_VALUES = (1, 4, 8, 16, 32)
DEFAULT_LAYER = "model.layers.0.mlp.experts.0.gate_proj"
GPTQ_TENSOR_NAMES = ("qweight", "qzeros", "scales", "g_idx")


def parse_m_values(value: str) -> tuple[int, ...]:
    values = []
    for item in value.split(","):
        try:
            parsed = int(item.strip())
        except ValueError as error:
            raise ValueError("m-values must contain positive integers") from error
        if parsed < 1:
            raise ValueError("m-values must contain positive integers")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError("m-values must not be empty")
    return tuple(values)


def summarize_timings(
    timings_ms: Sequence[float],
    *,
    m: int,
    k: int,
    n: int,
) -> dict:
    if not timings_ms:
        raise ValueError("at least one timing sample is required")
    summary = summarize_values(list(timings_ms))
    mean_ms = summary["mean"]
    assert mean_ms is not None
    return {
        "count": summary["count"],
        "mean_ms": mean_ms,
        "p50_ms": summary["p50"],
        "p99_ms": summary["p99"],
        "min_ms": min(timings_ms),
        "max_ms": summary["max"],
        "effective_tflops": (2 * m * k * n) / (mean_ms / 1000 * 1e12),
    }


def build_result_row(
    *,
    backend: str,
    m: int,
    k: int,
    n: int,
    group_size: int,
    timings_ms: Sequence[float],
    peak_allocated_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
) -> dict:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown GPTQ backend: {backend}")
    row = {
        "backend": backend,
        "shape": {"m": m, "k": k, "n": n, "group_size": group_size},
        "timing_ms": summarize_timings(timings_ms, m=m, k=k, n=n),
    }
    if peak_allocated_bytes is not None:
        row["peak_allocated_bytes"] = int(peak_allocated_bytes)
    if peak_reserved_bytes is not None:
        row["peak_reserved_bytes"] = int(peak_reserved_bytes)
    return row


def _read_checkpoint_tensors(model_file: Path, layer: str) -> dict[str, torch.Tensor]:
    prefix = layer.rstrip(".") + "."
    with safe_open(str(model_file), framework="pt", device="cpu") as checkpoint:
        tensors = {}
        for name in GPTQ_TENSOR_NAMES:
            key = prefix + name
            if key not in checkpoint.keys():
                raise ValueError(f"checkpoint is missing tensor: {key}")
            tensors[name] = checkpoint.get_tensor(key)
    return tensors


def _build_linear(
    tensors: Mapping[str, torch.Tensor],
    *,
    backend: str,
    marlin_library: str | None,
    device: torch.device,
) -> tuple[GPTQLinear, int, int, int]:
    qweight = tensors["qweight"]
    scales = tensors["scales"]
    input_size = qweight.shape[0] * 8
    output_size = qweight.shape[1]
    group_size = input_size // scales.shape[0]
    if group_size <= 0 or input_size % group_size:
        raise ValueError("checkpoint has an invalid GPTQ group layout")
    linear = GPTQLinear(
        input_size,
        output_size,
        group_size=group_size,
        backend=backend,
        marlin_library=marlin_library,
    ).to(device)
    for name in GPTQ_TENSOR_NAMES:
        getattr(linear, name).data.copy_(tensors[name])
    linear.prepare_for_runtime()
    return linear, input_size, output_size, group_size


@torch.inference_mode()
def _measure_forward(
    linear: GPTQLinear,
    inputs: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        linear(inputs)
    torch.cuda.synchronize(inputs.device)
    starts = []
    ends = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        linear(inputs)
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize(inputs.device)
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def run_microbenchmark(args: argparse.Namespace) -> dict:
    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    model_file = model / "model.safetensors"
    if not model_file.is_file():
        raise ValueError(f"checkpoint file does not exist: {model_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPTQ GEMM microbenchmark")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")

    backends = parse_backends(args.backends)
    m_values = parse_m_values(args.m_values)
    marlin_library = None
    if "marlin" in backends:
        if not args.marlin_library:
            raise ValueError("--marlin-library is required when using Marlin")
        marlin_library = str(Path(args.marlin_library).expanduser().resolve())
        if not Path(marlin_library).is_file():
            raise ValueError(f"Marlin library does not exist: {marlin_library}")
    elif args.marlin_library:
        raise ValueError("--marlin-library requires the marlin backend")

    tensors = _read_checkpoint_tensors(model_file, args.layer)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    max_m = max(m_values)
    input_bank = torch.randn(
        max_m,
        tensors["qweight"].shape[0] * 8,
        dtype=torch.float16,
        device=device,
    )

    results = []
    for backend in backends:
        linear = None
        try:
            linear, input_size, output_size, group_size = _build_linear(
                tensors,
                backend=backend,
                marlin_library=marlin_library,
                device=device,
            )
            torch.cuda.synchronize(device)
            for m in m_values:
                torch.cuda.reset_peak_memory_stats(device)
                timings_ms = _measure_forward(
                    linear,
                    input_bank[:m],
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                results.append(build_result_row(
                    backend=backend,
                    m=m,
                    k=input_size,
                    n=output_size,
                    group_size=group_size,
                    timings_ms=timings_ms,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                ))
        finally:
            del linear
            torch.cuda.empty_cache()

    return {
        "schema_version": 1,
        "benchmark": "moe_gptq_gemm",
        "model": str(model),
        "metadata": build_environment_metadata(),
        "workload": {
            "layer": args.layer,
            "checkpoint_tensor": "model.safetensors",
            "input_dtype": "float16",
            "m_values": list(m_values),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "backends": list(backends),
            "marlin_library": marlin_library,
        },
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--backends", default="tinygemm,marlin")
    parser.add_argument("--marlin-library")
    parser.add_argument("--m-values", default=",".join(map(str, DEFAULT_M_VALUES)))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_microbenchmark(args)
    atomic_write_json(args.output, report)
    print(json.dumps({"result_count": len(report["results"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
