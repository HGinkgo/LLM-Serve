from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file
from torch import nn

from llmserve.layers.quantization.awq import dequantize_awq_gemm
from llmserve.quantization.awq_calibration import (
    PackedAWQWeight,
    apply_awq_clip,
    fold_linear_scale,
    fold_rmsnorm_scale,
    quantize_awq_weight,
    search_awq_clip,
    search_awq_scale,
)


QWEN3_QUANTIZED_LINEAR_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class Qwen3LayerCalibrationResult:
    hidden_states: torch.Tensor
    packed_weights: dict[str, PackedAWQWeight]
    clipped_group_count: int
    evaluated_clip_group_count: int


def _run_layer(layer: nn.Module, hidden_states: torch.Tensor, kwargs: dict) -> torch.Tensor:
    output = layer(hidden_states, **kwargs)
    if isinstance(output, tuple):
        return output[0]
    return output


@torch.no_grad()
def _capture_inputs(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    kwargs: dict,
) -> dict[str, torch.Tensor]:
    capture_names = (
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    )
    captured = {}
    handles = []

    for name in capture_names:
        module = layer.get_submodule(name)

        def capture(_module, args, capture_name=name):
            captured[capture_name] = args[0].detach()

        handles.append(module.register_forward_pre_hook(capture))
    try:
        _run_layer(layer, hidden_states, kwargs)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in capture_names if name not in captured]
    if missing:
        raise RuntimeError(f"Qwen3 calibration did not observe inputs for: {missing}")
    return captured


@torch.no_grad()
def calibrate_qwen3_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    forward_kwargs: dict | None = None,
    group_size: int = 128,
    n_grid: int = 20,
    max_tokens: int = 512,
    apply_clip: bool = True,
) -> Qwen3LayerCalibrationResult:
    kwargs = {} if forward_kwargs is None else forward_kwargs
    captured = _capture_inputs(layer, hidden_states, kwargs)

    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj
    qkv_scale = search_awq_scale(
        captured["self_attn.q_proj"],
        [q_proj.weight, k_proj.weight, v_proj.weight],
        group_size=group_size,
        n_grid=n_grid,
        max_tokens=max_tokens,
    )
    fold_rmsnorm_scale(
        layer.input_layernorm,
        [q_proj, k_proj, v_proj],
        qkv_scale,
    )

    gate_proj = layer.mlp.gate_proj
    up_proj = layer.mlp.up_proj
    gate_up_scale = search_awq_scale(
        captured["mlp.gate_proj"],
        [gate_proj.weight, up_proj.weight],
        group_size=group_size,
        n_grid=n_grid,
        max_tokens=max_tokens,
    )
    fold_rmsnorm_scale(
        layer.post_attention_layernorm,
        [gate_proj, up_proj],
        gate_up_scale,
    )

    down_proj = layer.mlp.down_proj
    down_scale = search_awq_scale(
        captured["mlp.down_proj"],
        [down_proj.weight],
        group_size=group_size,
        n_grid=n_grid,
        max_tokens=max_tokens,
    )
    fold_linear_scale(up_proj, [down_proj], down_scale)

    clipped_group_count = 0
    evaluated_clip_group_count = 0
    if apply_clip:
        adjusted_inputs = {
            "self_attn.v_proj": captured["self_attn.q_proj"] / qkv_scale,
            "self_attn.o_proj": captured["self_attn.o_proj"],
            "mlp.gate_proj": captured["mlp.gate_proj"] / gate_up_scale,
            "mlp.up_proj": captured["mlp.gate_proj"] / gate_up_scale,
            "mlp.down_proj": captured["mlp.down_proj"] / down_scale,
        }
        for name, inputs in adjusted_inputs.items():
            module = layer.get_submodule(name)
            original_max = module.weight.reshape(
                module.weight.shape[0],
                -1,
                group_size,
            ).abs().amax(dim=-1, keepdim=True)
            clip_values = search_awq_clip(
                module.weight,
                inputs,
                group_size=group_size,
                n_grid=n_grid,
                max_tokens=max_tokens,
            )
            evaluated_clip_group_count += clip_values.numel()
            clipped_group_count += int(torch.count_nonzero(clip_values < original_max).item())
            apply_awq_clip(module.weight, clip_values, group_size=group_size)

    packed_weights = {}
    for name in QWEN3_QUANTIZED_LINEAR_NAMES:
        module = layer.get_submodule(name)
        packed = quantize_awq_weight(module.weight, group_size=group_size)
        packed_weights[name] = packed
        module.weight.copy_(
            dequantize_awq_gemm(
                packed.qweight,
                packed.qzeros,
                packed.scales,
                group_size=group_size,
            ).t()
        )

    output = _run_layer(layer, hidden_states, kwargs)
    return Qwen3LayerCalibrationResult(
        hidden_states=output,
        packed_weights=packed_weights,
        clipped_group_count=clipped_group_count,
        evaluated_clip_group_count=evaluated_clip_group_count,
    )


def _cache_tensors(
    layer: nn.Module,
    packed_weights: dict[str, PackedAWQWeight],
) -> dict[str, torch.Tensor]:
    tensors = {
        "input_layernorm.weight": layer.input_layernorm.weight.detach(),
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight.detach(),
    }
    for name, packed in packed_weights.items():
        tensors[f"{name}.qweight"] = packed.qweight
        tensors[f"{name}.qzeros"] = packed.qzeros
        tensors[f"{name}.scales"] = packed.scales
    return {name: tensor.cpu().contiguous() for name, tensor in tensors.items()}


def save_qwen3_layer_cache(
    layer: nn.Module,
    packed_weights: dict[str, PackedAWQWeight],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        _cache_tensors(layer, packed_weights),
        str(path),
        metadata={"format": "llmserve-qwen3-awq-layer-v1"},
    )


def load_qwen3_layer_cache(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")


@torch.no_grad()
def apply_qwen3_layer_cache(
    layer: nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    group_size: int = 128,
) -> dict[str, PackedAWQWeight]:
    for norm_name in ("input_layernorm", "post_attention_layernorm"):
        norm = layer.get_submodule(norm_name)
        cached = tensors[f"{norm_name}.weight"]
        norm.weight.copy_(cached.to(device=norm.weight.device, dtype=norm.weight.dtype))

    packed_weights = {}
    for name in QWEN3_QUANTIZED_LINEAR_NAMES:
        module = layer.get_submodule(name)
        packed = PackedAWQWeight(
            qweight=tensors[f"{name}.qweight"].to(module.weight.device),
            qzeros=tensors[f"{name}.qzeros"].to(module.weight.device),
            scales=tensors[f"{name}.scales"].to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            ),
        )
        packed_weights[name] = packed
        module.weight.copy_(
            dequantize_awq_gemm(
                packed.qweight,
                packed.qzeros,
                packed.scales,
                group_size=group_size,
            ).t()
        )
    return packed_weights


def _copy_checkpoint_metadata(source: Path, output: Path) -> dict:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name == "config.json":
            continue
        if path.name.startswith("model") and (
            path.suffix == ".safetensors" or path.name.endswith(".index.json")
        ):
            continue
        shutil.copy2(path, output / path.name)
    with (source / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_awq_state_dict(
    model: nn.Module,
    layer_cache_paths: list[str | Path],
) -> dict[str, torch.Tensor]:
    quantized_weights = {
        f"model.layers.{layer_index}.{name}.weight"
        for layer_index in range(len(layer_cache_paths))
        for name in QWEN3_QUANTIZED_LINEAR_NAMES
    }
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if name not in quantized_weights
    }
    for layer_index, cache_path in enumerate(layer_cache_paths):
        cached = load_qwen3_layer_cache(cache_path)
        for name in QWEN3_QUANTIZED_LINEAR_NAMES:
            prefix = f"model.layers.{layer_index}.{name}"
            for suffix in ("qweight", "qzeros", "scales"):
                state_dict[f"{prefix}.{suffix}"] = cached[
                    f"{name}.{suffix}"
                ].contiguous()
    return state_dict


def export_qwen3_awq_checkpoint(
    model: nn.Module,
    layer_cache_paths: list[str | Path],
    source_model_dir: str | Path,
    output_dir: str | Path,
    *,
    calibration_metadata: dict,
    group_size: int = 128,
    max_shard_size: str | int = "4GB",
) -> None:
    source = Path(source_model_dir)
    output = Path(output_dir)
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"missing source config: {source / 'config.json'}")
    if len(layer_cache_paths) != len(model.model.layers):
        raise ValueError("one AWQ cache is required for every Qwen3 decoder layer")
    output.mkdir(parents=True, exist_ok=True)

    config = _copy_checkpoint_metadata(source, output)
    config["quantization_config"] = {
        "backend": "autoawq",
        "bits": 4,
        "do_fuse": False,
        "group_size": group_size,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
    }
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    state_dict = _build_awq_state_dict(model, layer_cache_paths)
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    for old_path in output.glob("model*.safetensors"):
        old_path.unlink()
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {name: state_dict[name] for name in tensor_names}
        save_file(shard, str(output / filename), metadata={"format": "pt"})
    if split.is_sharded:
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metadata": split.metadata,
                    "weight_map": split.tensor_to_filename,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    manifest = {
        "format": "autoawq-gemm",
        "quant_method": "activation-aware-w4a16",
        "bits": 4,
        "group_size": group_size,
        "zero_point": True,
        "source_model": str(source.resolve()),
        "num_layers": len(layer_cache_paths),
        "calibration": calibration_metadata,
        "checkpoint_bytes": split.metadata["total_size"],
    }
    with (output / "quantization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
