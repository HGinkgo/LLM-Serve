"""GPTQ checkpoint conversion for PyTorch's CUDA int4 TinyGEMM operator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


GPTQ_PACK_FACTOR = 8
TINY_GEMM_INNER_K_TILES = 8


def _unpack_int32_nibbles(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(
        GPTQ_PACK_FACTOR,
        device=packed.device,
        dtype=torch.int32,
    ) * 4
    mask = torch.tensor(0xF, device=packed.device, dtype=torch.int32)
    return ((packed.unsqueeze(-1) >> shifts) & mask).flatten(-2)


def unpack_gptq_qweight(
    packed: torch.Tensor,
    *,
    input_size: int,
    output_size: int,
) -> torch.Tensor:
    """Return GPTQ's input-packed qweight as [input_size, output_size]."""
    if tuple(packed.shape) != (input_size // GPTQ_PACK_FACTOR, output_size):
        raise ValueError(
            "unexpected GPTQ qweight shape: "
            f"expected {(input_size // GPTQ_PACK_FACTOR, output_size)}, "
            f"got {tuple(packed.shape)}"
        )
    shifts = torch.arange(
        GPTQ_PACK_FACTOR,
        device=packed.device,
        dtype=torch.int32,
    ) * 4
    mask = torch.tensor(0xF, device=packed.device, dtype=torch.int32)
    return ((packed.unsqueeze(1) >> shifts.view(1, -1, 1)) & mask).reshape(
        input_size,
        output_size,
    )


def unpack_gptq_qzeros(
    packed: torch.Tensor,
    *,
    num_groups: int,
    output_size: int,
) -> torch.Tensor:
    if tuple(packed.shape) != (num_groups, output_size // GPTQ_PACK_FACTOR):
        raise ValueError(
            "unexpected GPTQ qzeros shape: "
            f"expected {(num_groups, output_size // GPTQ_PACK_FACTOR)}, "
            f"got {tuple(packed.shape)}"
        )
    return _unpack_int32_nibbles(packed)


def gptq_qzeros_to_tinygemm_offset(
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Translate ``(q - zero) * scale`` to TinyGEMM's centered form.

    TinyGEMM evaluates ``(q - 8) * scale + offset``. GPTQ checkpoints
    represent the zero point separately, so the additive offset is
    ``(8 - zero_point) * scale``.
    """
    if scales.shape != zero_points.shape:
        raise ValueError("GPTQ scales and zero_points must have the same shape")
    return (8 - zero_points.to(scales.dtype)) * scales


class GPTQLinear(nn.Module):
    """GPTQ W4A16 linear with a CUDA TinyGEMM serving path.

    The checkpoint tensors are retained only until ``prepare_for_runtime``.
    That conversion uses PyTorch's native CUDA int4 packed GEMM provider and
    releases qweight/qzeros/scales afterwards. The unprepared forward is a
    correctness oracle for unit tests and must not be used for serving.
    """

    def __init__(self, input_size: int, output_size: int, *, group_size: int):
        super().__init__()
        if input_size % group_size != 0:
            raise ValueError("input_size must be divisible by group_size")
        if input_size % GPTQ_PACK_FACTOR != 0:
            raise ValueError("input_size must be divisible by GPTQ pack_factor")
        if output_size % GPTQ_PACK_FACTOR != 0:
            raise ValueError("output_size must be divisible by GPTQ pack_factor")
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size
        num_groups = input_size // group_size

        # Use Parameters to keep the generic safetensors loader simple.
        self.qweight = nn.Parameter(
            torch.empty(input_size // GPTQ_PACK_FACTOR, output_size, dtype=torch.int32),
            requires_grad=False,
        )
        self.qzeros = nn.Parameter(
            torch.empty(num_groups, output_size // GPTQ_PACK_FACTOR, dtype=torch.int32),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty(num_groups, output_size),
            requires_grad=False,
        )
        self.g_idx = nn.Parameter(
            torch.arange(input_size, dtype=torch.int32) // group_size,
            requires_grad=False,
        )
        self.register_buffer("packed_weight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("scales_and_zeros", torch.empty(0, dtype=torch.bfloat16))
        self.register_buffer("runtime_ready", torch.tensor(False), persistent=False)

    @property
    def num_groups(self) -> int:
        return self.input_size // self.group_size

    def validate_group_mapping(self) -> None:
        expected = torch.arange(
            self.input_size,
            device=self.g_idx.device,
            dtype=torch.int32,
        ) // self.group_size
        if self.g_idx.numel() != self.input_size or not torch.equal(self.g_idx, expected):
            raise ValueError(
                "GPTQ TinyGEMM requires contiguous groups; "
                "act-order/g_idx permutations are unsupported"
            )

    def _zero_points(self) -> torch.Tensor:
        zeros = unpack_gptq_qzeros(
            self.qzeros,
            num_groups=self.num_groups,
            output_size=self.output_size,
        )
        # GPTQ serializes zero_point - 1 so it can reserve the all-zero code.
        return zeros + 1

    def dequantize_weight(self) -> torch.Tensor:
        """Reference dequantization for numerical tests only."""
        self.validate_group_mapping()
        codes = unpack_gptq_qweight(
            self.qweight,
            input_size=self.input_size,
            output_size=self.output_size,
        )
        zero_points = self._zero_points()
        groups = self.g_idx.to(torch.long)
        scales = self.scales.index_select(0, groups)
        zeros = zero_points.index_select(0, groups)
        return (codes.to(scales.dtype) - zeros.to(scales.dtype)) * scales

    @torch.no_grad()
    def prepare_for_runtime(self) -> None:
        """Convert loaded GPTQ tensors to the native CUDA TinyGEMM layout."""
        if bool(self.runtime_ready):
            return
        if self.qweight.device.type != "cuda":
            raise RuntimeError("GPTQ TinyGEMM preparation requires CUDA checkpoint tensors")
        if self.input_size % (TINY_GEMM_INNER_K_TILES * 16) != 0:
            raise ValueError(
                "GPTQ TinyGEMM requires input_size divisible by "
                f"{TINY_GEMM_INNER_K_TILES * 16}"
            )
        self.validate_group_mapping()

        # GPTQ packs eight K values per int32. TinyGEMM packs two adjacent K
        # values per byte, with the even K code in the high nibble.
        codes = unpack_gptq_qweight(
            self.qweight,
            input_size=self.input_size,
            output_size=self.output_size,
        ).transpose(0, 1).to(torch.uint8)
        tinygemm_input = ((codes[:, 0::2] << 4) | codes[:, 1::2]).contiguous()
        packed_weight = torch.ops.aten._convert_weight_to_int4pack(
            tinygemm_input,
            TINY_GEMM_INNER_K_TILES,
        )

        scales = self.scales.to(torch.bfloat16)
        offsets = gptq_qzeros_to_tinygemm_offset(scales, self._zero_points())
        scales_and_zeros = torch.stack((scales, offsets), dim=-1).contiguous()

        self.packed_weight = packed_weight
        self.scales_and_zeros = scales_and_zeros
        # The packed representation replaces, rather than supplements, source tensors.
        self.qweight.data = torch.empty(0, dtype=torch.int32, device=self.qweight.device)
        self.qzeros.data = torch.empty(0, dtype=torch.int32, device=self.qzeros.device)
        self.scales.data = torch.empty(0, dtype=self.scales.dtype, device=self.scales.device)
        self.g_idx.data = torch.empty(0, dtype=torch.int32, device=self.g_idx.device)
        self.runtime_ready.fill_(True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not bool(self.runtime_ready):
            weight = self.dequantize_weight().to(hidden_states.dtype)
            return F.linear(hidden_states, weight.transpose(0, 1))
        original_shape = hidden_states.shape
        if original_shape[-1] != self.input_size:
            raise ValueError(
                f"expected hidden size {self.input_size}, got {original_shape[-1]}"
            )
        inputs = hidden_states.reshape(-1, self.input_size).to(torch.bfloat16).contiguous()
        output = torch.ops.aten._weight_int4pack_mm(
            inputs,
            self.packed_weight,
            self.group_size,
            self.scales_and_zeros,
        )
        return output.reshape(*original_shape[:-1], self.output_size).to(hidden_states.dtype)
