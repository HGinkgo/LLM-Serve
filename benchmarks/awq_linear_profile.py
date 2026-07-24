import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.environment import (
    atomic_write_json,
    build_environment_metadata,
    discover_model_revision,
)
from benchmarks.linear_profile import PROJECTION_NAMES, parse_num_tokens
from llmserve.layers.quantization.awq import (
    AWQRuntimeConfig,
    awq_reference_linear,
    dequantize_awq_gemm,
)
from llmserve.layers.quantization.awq_triton import awq_triton_linear
from llmserve.layers.quantization.awq_cuda import (
    awq_cuda_linear,
    repack_awq_qweight_cuda,
    reorder_awq_qzeros_cuda,
    select_awq_cuda_split_k,
)


PROJECTION_COMPONENTS = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "o": ("o_proj",),
    "gate_up": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
}
AWQ_TENSOR_NAMES = ("qweight", "qzeros", "scales")


@dataclass(frozen=True)
class AWQProjectionTensors:
    name: str
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor

    @property
    def input_size(self) -> int:
        return self.qweight.shape[0]

    @property
    def output_size(self) -> int:
        return self.scales.shape[1]

    @property
    def payload_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.qweight, self.qzeros, self.scales)
        )

    def to(self, device: torch.device) -> "AWQProjectionTensors":
        return AWQProjectionTensors(
            self.name,
            self.qweight.to(device),
            self.qzeros.to(device),
            self.scales.to(device),
        )


def merge_projection_tensors(
    name: str,
    parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> AWQProjectionTensors:
    if not parts:
        raise ValueError("projection must contain at least one checkpoint module")
    input_group_shapes = {
        (qweight.shape[0], qzeros.shape[0], scales.shape[0])
        for qweight, qzeros, scales in parts
    }
    if len(input_group_shapes) != 1:
        raise ValueError("projection parts must share input/group dimensions")
    projection = AWQProjectionTensors(
        name=name,
        qweight=torch.cat([part[0] for part in parts], dim=1),
        qzeros=torch.cat([part[1] for part in parts], dim=1),
        scales=torch.cat([part[2] for part in parts], dim=1),
    )
    if projection.qweight.shape[1] * 8 != projection.output_size:
        raise ValueError("qweight packed output does not match scales")
    if projection.qzeros.shape[1] * 8 != projection.output_size:
        raise ValueError("qzeros packed output does not match scales")
    return projection


def load_layer_projections(
    model_path: str | Path,
    *,
    layer_id: int,
) -> tuple[AWQProjectionTensors, ...]:
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("AWQ profiler requires model.safetensors.index.json")
    with index_path.open(encoding="utf-8") as file:
        weight_map = json.load(file).get("weight_map", {})

    prefix = f"model.layers.{layer_id}."
    required_keys = []
    for component_names in PROJECTION_COMPONENTS.values():
        for component_name in component_names:
            module_prefix = (
                f"{prefix}self_attn.{component_name}"
                if component_name in {"q_proj", "k_proj", "v_proj", "o_proj"}
                else f"{prefix}mlp.{component_name}"
            )
            required_keys.extend(
                f"{module_prefix}.{tensor_name}"
                for tensor_name in AWQ_TENSOR_NAMES
            )
    missing = [key for key in required_keys if key not in weight_map]
    if missing:
        raise ValueError(f"checkpoint index is missing AWQ tensors: {missing}")

    tensors = {}
    keys_by_file = {}
    for key in required_keys:
        keys_by_file.setdefault(weight_map[key], []).append(key)
    for filename, keys in keys_by_file.items():
        with safe_open(model_path / filename, framework="pt", device="cpu") as file:
            for key in keys:
                tensors[key] = file.get_tensor(key)

    projections = []
    for projection_name in PROJECTION_NAMES:
        parts = []
        for component_name in PROJECTION_COMPONENTS[projection_name]:
            module_prefix = (
                f"{prefix}self_attn.{component_name}"
                if component_name in {"q_proj", "k_proj", "v_proj", "o_proj"}
                else f"{prefix}mlp.{component_name}"
            )
            parts.append(
                tuple(
                    tensors[f"{module_prefix}.{tensor_name}"]
                    for tensor_name in AWQ_TENSOR_NAMES
                )
            )
        projections.append(merge_projection_tensors(projection_name, parts))
    return tuple(projections)


def summarize_stage_latencies(
    measurements: dict[str, dict[str, float]],
    *,
    num_hidden_layers: int,
) -> dict:
    missing = [name for name in PROJECTION_NAMES if name not in measurements]
    if missing:
        raise ValueError(f"missing projection latency: {', '.join(missing)}")
    stage_names = ("dequantize", "matmul", "reference", "triton", "cuda")
    per_layer_ms = {
        stage: sum(float(measurements[name][stage]) for name in PROJECTION_NAMES)
        for stage in stage_names
    }
    component_total = per_layer_ms["dequantize"] + per_layer_ms["matmul"]
    if any(per_layer_ms[stage] <= 0 for stage in stage_names):
        raise ValueError("AWQ stage latency totals must be positive")
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    return {
        "per_layer_ms": per_layer_ms,
        "dequantize_component_share_pct": (
            per_layer_ms["dequantize"] / component_total * 100.0
        ),
        "model_reference_total_ms": (
            per_layer_ms["reference"] * num_hidden_layers
        ),
        "model_triton_total_ms": per_layer_ms["triton"] * num_hidden_layers,
        "model_cuda_total_ms": per_layer_ms["cuda"] * num_hidden_layers,
        "triton_speedup_vs_reference": (
            per_layer_ms["reference"] / per_layer_ms["triton"]
        ),
        "cuda_speedup_vs_reference": (
            per_layer_ms["reference"] / per_layer_ms["cuda"]
        ),
    }


def measure_cuda_operation(operation, *, warmup: int, repeats: int) -> float:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = operation()
    end.record()
    end.synchronize()
    if output is None:
        raise RuntimeError("CUDA operation did not produce an output")
    return start.elapsed_time(end) / repeats


def measure_projection_cuda(
    projection: AWQProjectionTensors,
    *,
    num_tokens: int,
    group_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    if device.type != "cuda":
        raise ValueError("AWQ latency profiling requires a CUDA device")
    with torch.cuda.device(device), torch.inference_mode():
        projection = projection.to(device)
        inputs = torch.zeros(
            (num_tokens, projection.input_size),
            dtype=projection.scales.dtype,
            device=device,
        )
        weight = dequantize_awq_gemm(
            projection.qweight,
            projection.qzeros,
            projection.scales,
            group_size=group_size,
        )
        repacked_qweight = repack_awq_qweight_cuda(projection.qweight)
        reordered_qzeros = reorder_awq_qzeros_cuda(projection.qzeros)
        split_k = select_awq_cuda_split_k(
            num_tokens,
            projection.input_size,
            projection.output_size,
        )
        num_groups = projection.input_size // group_size
        while split_k > 1 and (
            split_k > num_groups or num_groups % split_k
        ):
            split_k //= 2
        workspace = torch.empty(
            (split_k, num_tokens, projection.output_size)
            if split_k > 1
            else (0,),
            dtype=torch.float32,
            device=device,
        )
        measurements = {
            "dequantize": measure_cuda_operation(
                lambda: dequantize_awq_gemm(
                    projection.qweight,
                    projection.qzeros,
                    projection.scales,
                    group_size=group_size,
                ),
                warmup=warmup,
                repeats=repeats,
            ),
            "matmul": measure_cuda_operation(
                lambda: torch.matmul(inputs, weight),
                warmup=warmup,
                repeats=repeats,
            ),
            "reference": measure_cuda_operation(
                lambda: awq_reference_linear(
                    inputs,
                    projection.qweight,
                    projection.qzeros,
                    projection.scales,
                    group_size=group_size,
                ),
                warmup=warmup,
                repeats=repeats,
            ),
            "triton": measure_cuda_operation(
                lambda: awq_triton_linear(
                    inputs,
                    projection.qweight,
                    projection.qzeros,
                    projection.scales,
                    group_size=group_size,
                ),
                warmup=warmup,
                repeats=repeats,
            ),
            "cuda": measure_cuda_operation(
                lambda: awq_cuda_linear(
                    inputs,
                    repacked_qweight,
                    reordered_qzeros,
                    projection.scales,
                    group_size=group_size,
                    split_k=split_k,
                    workspace=workspace,
                ),
                warmup=warmup,
                repeats=repeats,
            ),
        }
    del (
        projection,
        inputs,
        weight,
        repacked_qweight,
        reordered_qzeros,
        workspace,
    )
    torch.cuda.empty_cache()
    return measurements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Qwen3 AWQ reference dequantize and matmul stages."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument(
        "--num-tokens",
        type=parse_num_tokens,
        default=parse_num_tokens("1,4,8,16"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    from transformers import AutoConfig

    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AWQ Linear profiling")
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != "qwen3":
        raise ValueError(f"expected a Qwen3 config, got {config.model_type!r}")
    awq_config = AWQRuntimeConfig.from_hf_config(config)
    projections = load_layer_projections(args.model, layer_id=args.layer_id)
    device = torch.device(args.device)
    measurements = []
    for num_tokens in args.num_tokens:
        projection_measurements = {
            projection.name: measure_projection_cuda(
                projection,
                num_tokens=num_tokens,
                group_size=awq_config.group_size,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            for projection in projections
        }
        measurements.append(
            {
                "num_tokens": num_tokens,
                "projection_latency_ms": projection_measurements,
                "summary": summarize_stage_latencies(
                    projection_measurements,
                    num_hidden_layers=int(config.num_hidden_layers),
                ),
            }
        )
    document = {
        "schema_version": 1,
        "benchmark": "qwen3-awq-linear",
        "metadata": build_environment_metadata(),
        "model": Path(args.model).name,
        "model_revision": discover_model_revision(args.model),
        "config": {
            "device": str(device),
            "layer_id": args.layer_id,
            "num_tokens": list(args.num_tokens),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "group_size": awq_config.group_size,
            "activation_dtype": str(awq_config.activation_dtype),
            "num_hidden_layers": int(config.num_hidden_layers),
        },
        "shapes": [
            {
                "name": projection.name,
                "input_size": projection.input_size,
                "output_size": projection.output_size,
                "payload_bytes_per_layer": projection.payload_bytes,
            }
            for projection in projections
        ],
        "measurements": measurements,
    }
    if args.output is not None:
        atomic_write_json(args.output, document)
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
