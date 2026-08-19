"""GPTQ checkpoint conversion for CUDA W4A16 serving backends."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F
from torch import nn


GPTQ_PACK_FACTOR = 8
TINY_GEMM_INNER_K_TILES = 8
MARLIN_MIN_CAPABILITY = 8
MARLIN_MIN_INPUT_SIZE = 128
MARLIN_MIN_OUTPUT_SIZE = 64
MARLIN_SYMMETRIC_QZERO_PACKED = 0x77777777
# vLLM 0.9.1 scalar_types.uint4b8.id. This provider intentionally accepts only
# that ABI instead of importing the vLLM Python serving runtime.
VLLM_UINT4B8_TYPE_ID = 1125899907892224
_MARLIN_SCALE_PERMUTATION = tuple(
    index + 8 * group for index in range(8) for group in range(8)
)
_MARLIN_SINGLE_GROUP_SCALE_PERMUTATION = tuple(
    2 * index + offset
    for index in range(4)
    for offset in (0, 1, 8, 9, 16, 17, 24, 25)
)
_marlin_load_lock = Lock()
_loaded_marlin_libraries: set[str] = set()


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


def marlin_permute_scales(
    scales: torch.Tensor,
    *,
    input_size: int,
    output_size: int,
    group_size: int,
) -> torch.Tensor:
    """Reorder grouped GPTQ scales into the vLLM Marlin tile layout."""
    if group_size != 128:
        raise ValueError("Marlin supports group_size=128 for this GPTQ backend")
    if input_size % MARLIN_MIN_INPUT_SIZE != 0:
        raise ValueError("Marlin requires input_size divisible by 128")
    if output_size % MARLIN_MIN_OUTPUT_SIZE != 0:
        raise ValueError("Marlin requires output_size divisible by 64")
    expected_shape = (input_size // group_size, output_size)
    if tuple(scales.shape) != expected_shape:
        raise ValueError(
            "unexpected GPTQ scale shape for Marlin: "
            f"expected {expected_shape}, got {tuple(scales.shape)}"
        )
    scale_permutation = (
        _MARLIN_SCALE_PERMUTATION
        if group_size < input_size
        else _MARLIN_SINGLE_GROUP_SCALE_PERMUTATION
    )
    permutation = torch.tensor(
        scale_permutation,
        dtype=torch.long,
        device=scales.device,
    )
    return scales.reshape(-1, permutation.numel()).index_select(
        1,
        permutation,
    ).reshape(expected_shape).contiguous()


def _load_marlin_library(library_path: str) -> None:
    path = Path(library_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Marlin provider library does not exist: {path}")
    path_string = str(path)
    with _marlin_load_lock:
        if path_string not in _loaded_marlin_libraries:
            torch.ops.load_library(path_string)
            _loaded_marlin_libraries.add(path_string)
    try:
        torch.ops._C.gptq_marlin_repack.default._schema
        torch.ops._C.gptq_marlin_gemm.default._schema
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "Marlin provider does not expose the vLLM 0.9.1 GPTQ-Marlin ABI"
        ) from error


class GPTQLinear(nn.Module):
    """GPTQ W4A16 linear with CUDA TinyGEMM or vLLM Marlin execution.

    The checkpoint tensors are retained only until ``prepare_for_runtime``.
    The selected conversion releases qweight/qzeros/scales afterwards. The
    unprepared forward is a correctness oracle for unit tests and must not be
    used for serving.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        group_size: int,
        backend: str = "tinygemm",
        marlin_library: str | None = None,
    ):
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
        self.configure_backend(backend, marlin_library)
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
        self.register_buffer("marlin_weight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("marlin_scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("marlin_workspace", torch.empty(0, dtype=torch.int32))
        self.register_buffer("marlin_empty", torch.empty(0, dtype=torch.int32))
        self.register_buffer("runtime_ready", torch.tensor(False), persistent=False)

    def configure_backend(self, backend: str, marlin_library: str | None) -> None:
        if backend not in {"tinygemm", "marlin"}:
            raise ValueError("GPTQ backend must be 'tinygemm' or 'marlin'")
        if backend == "marlin" and not marlin_library:
            raise ValueError("Marlin backend requires marlin_library")
        if hasattr(self, "runtime_ready") and bool(self.runtime_ready):
            raise RuntimeError("cannot change a prepared GPTQ backend")
        self.backend = backend
        self.marlin_library = marlin_library

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
        """Convert loaded GPTQ tensors to the selected CUDA serving layout."""
        if bool(self.runtime_ready):
            return
        if self.qweight.device.type != "cuda":
            raise RuntimeError("GPTQ runtime preparation requires CUDA checkpoint tensors")
        if self.backend == "tinygemm":
            self._prepare_tinygemm()
        else:
            self._prepare_marlin()
        self._release_source_tensors()
        self.runtime_ready.fill_(True)

    def _prepare_tinygemm(self) -> None:
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

    def _prepare_marlin(self) -> None:
        if self.input_size % MARLIN_MIN_INPUT_SIZE != 0:
            raise ValueError("Marlin requires input_size divisible by 128")
        if self.output_size % MARLIN_MIN_OUTPUT_SIZE != 0:
            raise ValueError("Marlin requires output_size divisible by 64")
        if torch.cuda.get_device_capability(self.qweight.device)[0] < MARLIN_MIN_CAPABILITY:
            raise RuntimeError("Marlin requires CUDA compute capability >= 8.0")
        self.validate_group_mapping()
        if not bool(torch.all(self.qzeros == MARLIN_SYMMETRIC_QZERO_PACKED)):
            raise ValueError("Marlin backend requires symmetric GPTQ zero points")
        assert self.marlin_library is not None
        _load_marlin_library(self.marlin_library)

        empty = torch.empty(0, dtype=torch.int32, device=self.qweight.device)
        self.marlin_weight = torch.ops._C.gptq_marlin_repack(
            self.qweight.contiguous(),
            empty,
            self.input_size,
            self.output_size,
            4,
        )
        scale_dtype = self.scales.dtype
        if scale_dtype not in {torch.float16, torch.bfloat16}:
            scale_dtype = torch.float16
        self.marlin_scales = marlin_permute_scales(
            self.scales.to(scale_dtype).contiguous(),
            input_size=self.input_size,
            output_size=self.output_size,
            group_size=self.group_size,
        )
        self.marlin_workspace = torch.zeros(
            torch.cuda.get_device_properties(self.qweight.device).multi_processor_count,
            dtype=torch.int32,
            device=self.qweight.device,
        )
        self.marlin_empty = empty

    def _release_source_tensors(self) -> None:
        # The packed representation replaces, rather than supplements, source tensors.
        self.qweight.data = torch.empty(0, dtype=torch.int32, device=self.qweight.device)
        self.qzeros.data = torch.empty(0, dtype=torch.int32, device=self.qzeros.device)
        self.scales.data = torch.empty(0, dtype=self.scales.dtype, device=self.scales.device)
        self.g_idx.data = torch.empty(0, dtype=torch.int32, device=self.g_idx.device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not bool(self.runtime_ready):
            weight = self.dequantize_weight().to(hidden_states.dtype)
            return F.linear(hidden_states, weight.transpose(0, 1))
        original_shape = hidden_states.shape
        if original_shape[-1] != self.input_size:
            raise ValueError(
                f"expected hidden size {self.input_size}, got {original_shape[-1]}"
            )
        if self.backend == "tinygemm":
            inputs = hidden_states.reshape(-1, self.input_size).to(torch.bfloat16).contiguous()
            output = torch.ops.aten._weight_int4pack_mm(
                inputs,
                self.packed_weight,
                self.group_size,
                self.scales_and_zeros,
            )
        else:
            inputs = hidden_states.reshape(-1, self.input_size).to(
                self.marlin_scales.dtype,
            ).contiguous()
            output = torch.ops._C.gptq_marlin_gemm(
                inputs,
                None,
                self.marlin_weight,
                self.marlin_scales,
                None,
                self.marlin_empty,
                self.marlin_empty,
                self.marlin_empty,
                self.marlin_workspace,
                VLLM_UINT4B8_TYPE_ID,
                inputs.shape[0],
                self.output_size,
                self.input_size,
                True,
                False,
                True,
                False,
            )
        return output.reshape(*original_shape[:-1], self.output_size).to(hidden_states.dtype)
