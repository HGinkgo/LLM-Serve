from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


_AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


@dataclass(frozen=True)
class PackedAWQWeight:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("INT4 packing expects a 2D tensor")
    if values.shape[1] % 8 != 0:
        raise ValueError("INT4 packing requires the output dimension to be divisible by 8")
    if values.is_floating_point():
        raise TypeError("INT4 packing expects an integer tensor")
    if torch.any((values < 0) | (values > 15)):
        raise ValueError("INT4 values must be in [0, 15]")

    ordered = values.reshape(values.shape[0], -1, 8)[..., _AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, 4, device=values.device, dtype=torch.int64)
    packed = torch.sum(ordered.to(torch.int64) << shifts, dim=-1)
    return packed.to(torch.int32)


def quantize_awq_weight(weight: torch.Tensor, group_size: int = 128) -> PackedAWQWeight:
    """Quantize an [out_features, in_features] weight to AutoAWQ GEMM layout."""
    if weight.ndim != 2:
        raise ValueError("AWQ quantization expects a two-dimensional weight tensor")
    if not weight.is_floating_point():
        raise ValueError("AWQ quantization expects a floating-point weight tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError("input size must be divisible by group_size")
    if out_features % 8 != 0:
        raise ValueError("output size must be divisible by 8")

    groups = in_features // group_size
    grouped = weight.to(torch.float32).reshape(out_features, groups, group_size)
    group_min = grouped.amin(dim=-1)
    group_max = grouped.amax(dim=-1)
    scales = ((group_max - group_min).clamp(min=1e-5) / 15.0)
    zeros = (-torch.round(group_min / scales)).clamp(0, 15)
    quantized = torch.round(grouped / scales.unsqueeze(-1)) + zeros.unsqueeze(-1)
    quantized = quantized.clamp(0, 15).to(torch.int32).reshape(out_features, in_features)

    qweight = _pack_int4(quantized.t().contiguous())
    qzeros = _pack_int4(zeros.to(torch.int32).t().contiguous())
    runtime_scales = scales.t().contiguous().to(weight.dtype)
    return PackedAWQWeight(qweight=qweight, qzeros=qzeros, scales=runtime_scales)


def fake_quantize_awq_weight(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Return the dequantized value produced by group-wise asymmetric INT4."""
    if weight.ndim != 2:
        raise ValueError("AWQ fake quantization expects a two-dimensional weight tensor")
    if not weight.is_floating_point():
        raise ValueError("AWQ fake quantization expects a floating-point weight tensor")
    if group_size <= 0 or weight.shape[1] % group_size != 0:
        raise ValueError("input size must be divisible by a positive group_size")

    original_dtype = weight.dtype
    grouped = weight.to(torch.float32).reshape(-1, group_size)
    group_min = grouped.amin(dim=-1, keepdim=True)
    group_max = grouped.amax(dim=-1, keepdim=True)
    scales = ((group_max - group_min).clamp(min=1e-5) / 15.0)
    zeros = (-torch.round(group_min / scales)).clamp(0, 15)
    quantized = (torch.round(grouped / scales) + zeros).clamp(0, 15)
    dequantized = (quantized - zeros) * scales
    return dequantized.reshape_as(weight).to(original_dtype)


def _normalized_weight_mean(
    weights: Sequence[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    input_size = weights[0].shape[1]
    total = torch.zeros(input_size, dtype=torch.float32, device=weights[0].device)
    output_rows = 0
    for weight in weights:
        for chunk in weight.detach().split(256, dim=0):
            grouped = chunk.to(torch.float32).abs().reshape(-1, group_size)
            grouped = grouped / (grouped.amax(dim=-1, keepdim=True) + 1e-6)
            total += grouped.reshape(chunk.shape[0], input_size).sum(dim=0)
        output_rows += weight.shape[0]
    return total / output_rows


def _linear_reconstruction_error(
    inputs: torch.Tensor,
    weights: Sequence[torch.Tensor],
    scales: torch.Tensor,
    group_size: int,
) -> float:
    inverse_scales = scales.reciprocal().view(1, -1)
    loss = torch.zeros((), dtype=torch.float32, device=inputs.device)
    elements = 0
    for weight in weights:
        reference = F.linear(inputs, weight)
        quantized = fake_quantize_awq_weight(
            weight * scales.view(1, -1),
            group_size=group_size,
        )
        candidate = F.linear(inputs, quantized * inverse_scales)
        loss += (reference - candidate).float().pow(2).sum()
        elements += reference.numel()
    return (loss / elements).item()


@torch.no_grad()
def search_awq_scale(
    inputs: torch.Tensor,
    weights: Sequence[torch.Tensor],
    *,
    group_size: int = 128,
    n_grid: int = 20,
    duo_scaling: bool = True,
    max_tokens: int = 512,
) -> torch.Tensor:
    """Search an activation-aware per-input-channel scale for related linears."""
    if inputs.ndim < 2 or not inputs.is_floating_point():
        raise ValueError("inputs must be a floating tensor with a channel dimension")
    if not weights:
        raise ValueError("at least one weight tensor is required")
    if n_grid <= 0 or max_tokens <= 0:
        raise ValueError("n_grid and max_tokens must be positive")

    input_size = inputs.shape[-1]
    for weight in weights:
        if weight.ndim != 2 or weight.shape[1] != input_size:
            raise ValueError("all weights must share the input channel dimension")
        if weight.device != inputs.device or weight.dtype != inputs.dtype:
            raise ValueError("inputs and weights must share device and dtype")
        if input_size % group_size != 0:
            raise ValueError("input size must be divisible by group_size")

    flattened = inputs.reshape(-1, input_size)
    if flattened.shape[0] > max_tokens:
        indices = torch.linspace(
            0,
            flattened.shape[0] - 1,
            max_tokens,
            device=flattened.device,
        ).long()
        flattened = flattened.index_select(0, indices)

    x_mean = flattened.to(torch.float32).abs().mean(dim=0).clamp(min=1e-4)
    w_mean = _normalized_weight_mean(weights, group_size).clamp(min=1e-4)
    identity = torch.ones(input_size, dtype=torch.float32, device=inputs.device)
    best_scales = identity
    best_error = _linear_reconstruction_error(
        flattened,
        weights,
        identity.to(inputs.dtype),
        group_size,
    )

    for grid_index in range(n_grid):
        ratio = grid_index / n_grid
        if duo_scaling:
            scales = x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)
        else:
            scales = x_mean.pow(ratio)
        scales = scales.clamp(min=1e-4)
        scales = scales / torch.sqrt(scales.max() * scales.min())
        scales = torch.where(torch.isfinite(scales), scales, identity)
        candidate = scales.to(inputs.dtype)
        error = _linear_reconstruction_error(
            flattened,
            weights,
            candidate,
            group_size,
        )
        if error < best_error:
            best_error = error
            best_scales = scales.clone()

    return best_scales.to(inputs.dtype)


@torch.no_grad()
def search_awq_clip(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    *,
    group_size: int = 128,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    max_tokens: int = 512,
    output_chunk_size: int = 256,
) -> torch.Tensor:
    """Search symmetric clipping thresholds per output channel and INT4 group."""
    if weight.ndim != 2 or inputs.ndim < 2:
        raise ValueError("weight and inputs must have channel dimensions")
    if not weight.is_floating_point() or not inputs.is_floating_point():
        raise ValueError("weight and inputs must be floating tensors")
    if weight.shape[1] != inputs.shape[-1]:
        raise ValueError("weight and input channel dimensions must match")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight and inputs must share device and dtype")
    if group_size <= 0 or weight.shape[1] % group_size != 0:
        raise ValueError("input size must be divisible by a positive group_size")
    if n_grid <= 0 or max_tokens <= 0 or output_chunk_size <= 0:
        raise ValueError("search sizes must be positive")
    if not 0 <= max_shrink < 1:
        raise ValueError("max_shrink must be in [0, 1)")

    input_size = weight.shape[1]
    groups = input_size // group_size
    flattened = inputs.reshape(-1, input_size)
    if flattened.shape[0] > max_tokens:
        step = max(1, flattened.shape[0] // max_tokens)
        flattened = flattened[::step][:max_tokens]
    grouped_inputs = flattened.reshape(-1, groups, group_size)

    result = []
    search_steps = max(1, int(max_shrink * n_grid))
    for weight_chunk in weight.split(output_chunk_size, dim=0):
        grouped_weight = weight_chunk.reshape(-1, groups, group_size)
        original_max = grouped_weight.abs().amax(dim=-1, keepdim=True)
        reference = torch.einsum("tgi,ogi->otg", grouped_inputs, grouped_weight)

        baseline_quantized = fake_quantize_awq_weight(
            weight_chunk,
            group_size=group_size,
        ).reshape_as(grouped_weight)
        baseline_output = torch.einsum(
            "tgi,ogi->otg",
            grouped_inputs,
            baseline_quantized,
        )
        best_errors = (reference - baseline_output).float().pow(2).mean(dim=1)
        best_max = original_max.clone()

        for grid_index in range(1, search_steps + 1):
            candidate_max = original_max * (1 - grid_index / n_grid)
            clipped = grouped_weight.clamp(-candidate_max, candidate_max)
            quantized = fake_quantize_awq_weight(
                clipped.reshape_as(weight_chunk),
                group_size=group_size,
            ).reshape_as(grouped_weight)
            candidate_output = torch.einsum(
                "tgi,ogi->otg",
                grouped_inputs,
                quantized,
            )
            errors = (reference - candidate_output).float().pow(2).mean(dim=1)
            improved = errors < best_errors
            best_errors = torch.where(improved, errors, best_errors)
            best_max = torch.where(improved.unsqueeze(-1), candidate_max, best_max)
        result.append(best_max)

    return torch.cat(result, dim=0).to(weight.dtype)


@torch.no_grad()
def apply_awq_clip(
    weight: torch.Tensor,
    max_values: torch.Tensor,
    *,
    group_size: int = 128,
) -> None:
    if weight.ndim != 2 or weight.shape[1] % group_size != 0:
        raise ValueError("weight input size must be divisible by group_size")
    expected_shape = (weight.shape[0], weight.shape[1] // group_size, 1)
    if max_values.shape != expected_shape:
        raise ValueError(f"clip values must have shape {expected_shape}")
    grouped = weight.reshape(weight.shape[0], -1, group_size)
    limits = max_values.to(device=weight.device, dtype=weight.dtype)
    grouped.clamp_(min=-limits, max=limits)


@torch.no_grad()
def fold_rmsnorm_scale(
    norm: nn.Module,
    linears: Sequence[nn.Module],
    scales: torch.Tensor,
) -> None:
    if not hasattr(norm, "weight") or norm.weight.numel() != scales.numel():
        raise ValueError("RMSNorm weight and scale sizes must match")
    norm.weight.div_(scales.to(device=norm.weight.device, dtype=norm.weight.dtype))
    bias = getattr(norm, "bias", None)
    if bias is not None:
        bias.div_(scales.to(device=bias.device, dtype=bias.dtype))
    for linear in linears:
        if linear.weight.shape[1] != scales.numel():
            raise ValueError("linear input size and scale size must match")
        linear.weight.mul_(
            scales.to(device=linear.weight.device, dtype=linear.weight.dtype).view(1, -1)
        )


@torch.no_grad()
def fold_linear_scale(
    previous: nn.Module,
    following: Sequence[nn.Module],
    scales: torch.Tensor,
) -> None:
    if previous.weight.shape[0] != scales.numel():
        raise ValueError("previous linear output size and scale size must match")
    previous_scale = scales.to(device=previous.weight.device, dtype=previous.weight.dtype)
    previous.weight.div_(previous_scale.view(-1, 1))
    bias = getattr(previous, "bias", None)
    if bias is not None:
        bias.div_(previous_scale)
    for linear in following:
        if linear.weight.shape[1] != scales.numel():
            raise ValueError("following linear input size and scale size must match")
        linear.weight.mul_(
            scales.to(device=linear.weight.device, dtype=linear.weight.dtype).view(1, -1)
        )
