from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


@dataclass(frozen=True)
class AWQRuntimeConfig:
    bits: int
    group_size: int
    zero_point: bool
    version: str
    backend: str
    activation_dtype: torch.dtype
    quant_method: str = "awq"
    execution_backend: str = "reference"

    @property
    def pack_factor(self) -> int:
        return 32 // self.bits

    @classmethod
    def from_hf_config(cls, hf_config) -> "AWQRuntimeConfig":
        raw = getattr(hf_config, "quantization_config", None)
        if not isinstance(raw, Mapping):
            raise ValueError("quantization_config must be an AWQ mapping")
        values = {
            "quant_method": str(raw.get("quant_method", "")).lower(),
            "backend": str(raw.get("backend", "")).lower(),
            "bits": raw.get("bits"),
            "group_size": raw.get("group_size"),
            "version": str(raw.get("version", "")).lower(),
            "zero_point": raw.get("zero_point"),
        }
        expected = {
            "quant_method": "awq",
            "backend": "autoawq",
            "bits": 4,
            "group_size": 128,
            "version": "gemm",
            "zero_point": True,
        }
        for field, expected_value in expected.items():
            if values[field] != expected_value:
                raise ValueError(
                    f"unsupported AWQ {field}: expected {expected_value!r}, "
                    f"got {values[field]!r}"
                )
        activation_dtype = getattr(hf_config, "dtype", None)
        if activation_dtype != torch.bfloat16:
            raise ValueError(
                "unsupported AWQ dtype: expected torch.bfloat16, "
                f"got {activation_dtype!r}"
            )
        return cls(
            bits=values["bits"],
            group_size=values["group_size"],
            zero_point=values["zero_point"],
            version=values["version"],
            backend=values["backend"],
            activation_dtype=activation_dtype,
            quant_method=values["quant_method"],
        )


class AWQLinearMethod:

    def __init__(self, config: AWQRuntimeConfig):
        self.config = config
        self._cuda_linear = None

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
    ) -> None:
        if getattr(layer, "tp_size", 1) != 1:
            raise ValueError("AWQ tensor parallel is not supported")
        if input_size % self.config.group_size:
            raise ValueError("AWQ input size must be divisible by group_size")
        if output_size % self.config.pack_factor:
            raise ValueError("AWQ output size must be divisible by pack_factor")

        num_groups = input_size // self.config.group_size
        packed_output_size = output_size // self.config.pack_factor
        layer.qweight = nn.Parameter(
            torch.empty(input_size, packed_output_size, dtype=torch.int32),
            requires_grad=False,
        )
        layer.qzeros = nn.Parameter(
            torch.empty(num_groups, packed_output_size, dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = nn.Parameter(
            torch.empty(
                num_groups,
                output_size,
                dtype=self.config.activation_dtype,
            ),
            requires_grad=False,
        )
        for param in (layer.qweight, layer.qzeros):
            param.output_dim = 1
            param.input_dim = 0
            param.packed_dim = 1
            param.pack_factor = self.config.pack_factor
        layer.scales.output_dim = 1
        layer.scales.input_dim = 0

    def apply(self, layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        if self.config.execution_backend == "cuda":
            if not hasattr(layer, "awq_qweight"):
                raise RuntimeError(
                    "AWQ CUDA weights must be processed after checkpoint loading"
                )
            return self._cuda_linear(
                inputs,
                layer.awq_qweight,
                layer.awq_qzeros,
                layer.scales,
                group_size=self.config.group_size,
                workspace=layer.awq_workspace,
            )
        if (
            self.config.execution_backend == "triton"
            and inputs.device.type == "cuda"
            and inputs.ndim == 2
            and 0 < inputs.shape[0] <= 16
        ):
            from llmserve.layers.quantization.awq_triton import (
                awq_triton_linear,
                is_awq_triton_available,
            )

            if is_awq_triton_available():
                return awq_triton_linear(
                    inputs,
                    layer.qweight,
                    layer.qzeros,
                    layer.scales,
                    group_size=self.config.group_size,
                )
        return awq_reference_linear(
            inputs,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            group_size=self.config.group_size,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if self.config.execution_backend != "cuda":
            return
        from llmserve.layers.quantization.awq_cuda import (
            awq_cuda_linear,
            repack_awq_qweight_cuda,
            reorder_awq_qzeros_cuda,
            select_awq_cuda_split_k,
        )

        repacked_qweight = repack_awq_qweight_cuda(layer.qweight)
        reordered_qzeros = reorder_awq_qzeros_cuda(layer.qzeros)
        k = repacked_qweight.shape[0] * 16
        n = repacked_qweight.shape[1] * 64
        split_k = select_awq_cuda_split_k(16, k, n)
        workspace = torch.empty(
            (split_k, 16, n) if split_k > 1 else (0,),
            dtype=torch.float32,
            device=repacked_qweight.device,
        )
        delattr(layer, "qweight")
        delattr(layer, "qzeros")
        layer.register_buffer("awq_qweight", repacked_qweight, persistent=False)
        layer.register_buffer("awq_qzeros", reordered_qzeros, persistent=False)
        layer.register_buffer("awq_workspace", workspace, persistent=False)
        self._cuda_linear = awq_cuda_linear


def unpack_awq_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError("packed AWQ tensor must be two-dimensional")
    if packed.dtype != torch.int32:
        raise ValueError("packed AWQ tensor must use int32")
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    unpacked = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(-1), shifts),
        0xF,
    )
    unpacked = unpacked.view(packed.shape[0], -1, 8)
    reverse_order = torch.tensor(
        AWQ_REVERSE_ORDER,
        dtype=torch.long,
        device=packed.device,
    )
    return unpacked[:, :, reverse_order].reshape(packed.shape[0], -1)


def _validate_awq_tensors(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    if qweight.ndim != 2 or qweight.dtype != torch.int32:
        raise ValueError("qweight must be a two-dimensional int32 tensor")
    if qzeros.ndim != 2 or qzeros.dtype != torch.int32:
        raise ValueError("qzeros must be a two-dimensional int32 tensor")
    if scales.ndim != 2 or not scales.is_floating_point():
        raise ValueError("scales must be a two-dimensional floating tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    input_size = qweight.shape[0]
    if input_size % group_size:
        raise ValueError("qweight input size must be divisible by group_size")
    num_groups = input_size // group_size
    output_size = qweight.shape[1] * 8
    if qzeros.shape != (num_groups, qweight.shape[1]):
        raise ValueError(
            "qzeros shape must be [input_size / group_size, output_size / 8]"
        )
    if scales.shape != (num_groups, output_size):
        raise ValueError(
            "scales shape must be [input_size / group_size, output_size]"
        )
    devices = {qweight.device, qzeros.device, scales.device}
    if len(devices) != 1:
        raise ValueError("qweight, qzeros, and scales must share one device")


def dequantize_awq_gemm(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    _validate_awq_tensors(qweight, qzeros, scales, group_size)
    quantized_weight = unpack_awq_int4(qweight)
    zeros = unpack_awq_int4(qzeros).repeat_interleave(group_size, dim=0)
    expanded_scales = scales.repeat_interleave(group_size, dim=0)
    return (
        quantized_weight.to(scales.dtype) - zeros.to(scales.dtype)
    ) * expanded_scales


def awq_reference_linear(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    if not inputs.is_floating_point():
        raise ValueError("AWQ activations must use a floating dtype")
    if inputs.dtype != scales.dtype:
        raise ValueError(
            "AWQ activation dtype must match checkpoint scales dtype"
        )
    if inputs.device != qweight.device:
        raise ValueError("AWQ activations and weights must share one device")
    if inputs.shape[-1] != qweight.shape[0]:
        raise ValueError("AWQ activation K must match qweight input size")
    weight = dequantize_awq_gemm(
        qweight,
        qzeros,
        scales,
        group_size=group_size,
    )
    output = torch.matmul(inputs.to(weight.dtype), weight)
    return output.to(inputs.dtype)
