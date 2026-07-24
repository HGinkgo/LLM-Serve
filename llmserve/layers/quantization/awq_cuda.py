from __future__ import annotations

import os
from pathlib import Path
import sys

import torch


AWQ_LOGICAL_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
CUDA_TILE_K = 16
CUDA_TILE_N = 64

_EXTENSION = None


def select_awq_cuda_split_k(m: int, k: int, n: int) -> int:
    if m <= 0 or k <= 0 or n <= 0:
        raise ValueError("AWQ CUDA M, K, and N must be positive")
    if m > 16:
        return 1
    if n >= 16384:
        return 1
    if k >= 8192:
        return 8
    return 4


def _reorder_autoawq_words(packed: torch.Tensor) -> torch.Tensor:
    reordered = torch.zeros_like(packed)
    for logical_index, source_shift in enumerate(AWQ_LOGICAL_SHIFTS):
        value = torch.bitwise_and(
            torch.bitwise_right_shift(packed, source_shift),
            0xF,
        )
        reordered = torch.bitwise_or(reordered, value << (4 * logical_index))
    return reordered


def repack_awq_qweight_reference(qweight: torch.Tensor) -> torch.Tensor:
    """Build the per-thread INT4 words consumed by two N8 MMA operations."""
    if qweight.ndim != 2 or qweight.dtype != torch.int32:
        raise ValueError("qweight must be a two-dimensional int32 tensor")
    k = qweight.shape[0]
    n = qweight.shape[1] * 8
    if k % CUDA_TILE_K:
        raise ValueError(f"qweight K must be divisible by {CUDA_TILE_K}")
    if n % CUDA_TILE_N:
        raise ValueError(f"qweight N must be divisible by {CUDA_TILE_N}")

    sequential = _reorder_autoawq_words(qweight).to(torch.int64)
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int64)
    values = ((sequential.unsqueeze(-1) >> shifts) & 0xF).reshape(k, n)
    output = torch.empty(
        (k // CUDA_TILE_K, n // CUDA_TILE_N, 4, 32),
        dtype=torch.int32,
        device=qweight.device,
    )
    pack_shifts = torch.arange(
        0,
        32,
        4,
        device=qweight.device,
        dtype=torch.int64,
    )
    for k_tile in range(k // CUDA_TILE_K):
        for n_tile in range(n // CUDA_TILE_N):
            for warp in range(4):
                for lane in range(32):
                    group = lane // 4
                    thread = lane % 4
                    rows = torch.tensor(
                        [
                            k_tile * 16 + thread * 2,
                            k_tile * 16 + thread * 2 + 1,
                            k_tile * 16 + thread * 2 + 8,
                            k_tile * 16 + thread * 2 + 9,
                        ],
                        device=qweight.device,
                    )
                    first_col = n_tile * 64 + warp * 16 + group
                    cols = torch.tensor(
                        [first_col, first_col + 8],
                        device=qweight.device,
                    )
                    fragment_values = torch.cat(
                        (values[rows, cols[0]], values[rows, cols[1]])
                    )
                    output[k_tile, n_tile, warp, lane] = torch.sum(
                        fragment_values << pack_shifts
                    ).to(torch.int32)
    return output


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("the AWQ CUDA backend requires CUDA")

    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(
            "the AWQ CUDA backend requires an SM80-or-newer GPU for BF16 MMA"
        )

    conda_cuda_home = Path(sys.prefix)
    if (conda_cuda_home / "bin" / "nvcc").is_file():
        os.environ["CUDA_HOME"] = str(conda_cuda_home)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    from torch.utils import cpp_extension

    if (conda_cuda_home / "bin" / "nvcc").is_file():
        cpp_extension.CUDA_HOME = str(conda_cuda_home)
    source = Path(__file__).with_name("csrc") / "awq_w4a16.cu"
    _EXTENSION = cpp_extension.load(
        name="llmserve_awq_w4a16",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_ldflags=[f"-L{conda_cuda_home / 'lib'}"],
        with_cuda=True,
        verbose=os.getenv("LLMSERVE_EXT_VERBOSE") == "1",
    )
    return _EXTENSION


def repack_awq_qweight_cuda(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.device.type != "cuda":
        raise ValueError("qweight must be a CUDA tensor")
    return _load_extension().repack_qweight(qweight)


def reorder_awq_qzeros_cuda(qzeros: torch.Tensor) -> torch.Tensor:
    if qzeros.device.type != "cuda":
        raise ValueError("qzeros must be a CUDA tensor")
    return _load_extension().reorder_qzeros(qzeros)


def awq_cuda_linear(
    inputs: torch.Tensor,
    repacked_qweight: torch.Tensor,
    reordered_qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    split_k: int | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    if inputs.device.type != "cuda":
        raise ValueError("AWQ CUDA inputs must be CUDA tensors")
    if group_size != 128:
        raise ValueError("AWQ CUDA currently requires group_size=128")
    k = repacked_qweight.shape[0] * CUDA_TILE_K
    n = repacked_qweight.shape[1] * CUDA_TILE_N
    if split_k is None:
        split_k = select_awq_cuda_split_k(inputs.shape[0], k, n)
        num_groups = k // group_size
        while split_k > 1 and (
            split_k > num_groups or num_groups % split_k
        ):
            split_k //= 2
    if split_k not in {1, 2, 4, 8}:
        raise ValueError("AWQ CUDA split_k must be one of 1, 2, 4, or 8")
    required_workspace = split_k * inputs.shape[0] * n if split_k > 1 else 0
    if workspace is None:
        workspace = torch.empty(
            required_workspace,
            dtype=torch.float32,
            device=inputs.device,
        )
    if workspace.device != inputs.device or workspace.dtype != torch.float32:
        raise ValueError("AWQ CUDA workspace must be float32 on the input device")
    if not workspace.is_contiguous() or workspace.numel() < required_workspace:
        raise ValueError("AWQ CUDA workspace is too small or non-contiguous")
    return _load_extension().linear(
        inputs,
        repacked_qweight,
        reordered_qzeros,
        scales,
        workspace,
        group_size,
        split_k,
    )
