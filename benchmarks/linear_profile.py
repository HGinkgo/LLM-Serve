import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.environment import (
    atomic_write_json,
    build_environment_metadata,
    discover_model_revision,
)


PROJECTION_NAMES = ("qkv", "o", "gate_up", "down")


@dataclass(frozen=True)
class LinearShape:
    name: str
    input_size: int
    output_size: int


def qwen3_projection_shapes(config) -> tuple[LinearShape, ...]:
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        if hidden_size % num_heads:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        head_dim = hidden_size // num_heads
    head_dim = int(head_dim)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    return (
        LinearShape("qkv", hidden_size, q_size + 2 * kv_size),
        LinearShape("o", q_size, hidden_size),
        LinearShape("gate_up", hidden_size, 2 * intermediate_size),
        LinearShape("down", intermediate_size, hidden_size),
    )


def summarize_latencies(
    latencies_ms: dict[str, float],
    *,
    num_hidden_layers: int,
) -> dict:
    missing = [name for name in PROJECTION_NAMES if name not in latencies_ms]
    if missing:
        raise ValueError(f"missing projection latency: {', '.join(missing)}")
    per_layer_total_ms = sum(
        float(latencies_ms[name]) for name in PROJECTION_NAMES
    )
    if per_layer_total_ms <= 0:
        raise ValueError("projection latency total must be positive")
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    return {
        "per_layer_total_ms": per_layer_total_ms,
        "model_projection_total_ms": per_layer_total_ms * num_hidden_layers,
        "share_pct": {
            name: float(latencies_ms[name]) / per_layer_total_ms * 100.0
            for name in PROJECTION_NAMES
        },
    }


def parse_num_tokens(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("num_tokens must be comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("num_tokens must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("num_tokens values must be unique")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Qwen3 dense projection shapes with CUDA events."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--num-tokens",
        type=parse_num_tokens,
        default=parse_num_tokens("1,4,8,16"),
        help="Comma-separated GEMM M values, not request concurrency.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser


def profile_projection_shapes(
    shapes: tuple[LinearShape, ...],
    *,
    num_hidden_layers: int,
    num_tokens: tuple[int, ...],
    measure_latency,
) -> list[dict]:
    results = []
    for token_count in num_tokens:
        latencies_ms = {
            shape.name: float(measure_latency(shape, token_count))
            for shape in shapes
        }
        results.append(
            {
                "num_tokens": token_count,
                "latency_ms": latencies_ms,
                "summary": summarize_latencies(
                    latencies_ms,
                    num_hidden_layers=num_hidden_layers,
                ),
            }
        )
    return results


def measure_dense_cuda(
    shape: LinearShape,
    *,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    if device.type != "cuda":
        raise ValueError("dense latency profiling requires a CUDA device")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    with torch.cuda.device(device), torch.inference_mode():
        inputs = torch.empty(
            (num_tokens, shape.input_size),
            dtype=dtype,
            device=device,
        )
        weight = torch.empty(
            (shape.output_size, shape.input_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(warmup):
            F.linear(inputs, weight)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            F.linear(inputs, weight)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeats


def main(argv=None):
    from transformers import AutoConfig

    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense Linear profiling")
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != "qwen3":
        raise ValueError(f"expected a Qwen3 config, got {config.model_type!r}")
    shapes = qwen3_projection_shapes(config)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    device = torch.device(args.device)

    def measure(shape, num_tokens):
        return measure_dense_cuda(
            shape,
            num_tokens=num_tokens,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    element_size = torch.empty((), dtype=dtype).element_size()
    document = {
        "schema_version": 1,
        "benchmark": "qwen3-dense-linear",
        "metadata": build_environment_metadata(),
        "model": Path(args.model).name,
        "model_revision": discover_model_revision(args.model),
        "config": {
            "dtype": args.dtype,
            "device": str(device),
            "num_tokens": list(args.num_tokens),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "num_hidden_layers": int(config.num_hidden_layers),
        },
        "shapes": [
            {
                "name": shape.name,
                "input_size": shape.input_size,
                "output_size": shape.output_size,
                "weight_elements": shape.input_size * shape.output_size,
                "dense_weight_bytes_per_layer": (
                    shape.input_size * shape.output_size * element_size
                ),
            }
            for shape in shapes
        ],
        "measurements": profile_projection_shapes(
            shapes,
            num_hidden_layers=int(config.num_hidden_layers),
            num_tokens=args.num_tokens,
            measure_latency=measure,
        ),
    }
    if args.output is not None:
        atomic_write_json(args.output, document)
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
