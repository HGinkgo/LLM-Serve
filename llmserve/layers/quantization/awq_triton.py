import torch

from llmserve.layers.quantization.awq import _validate_awq_tensors

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _awq_w4a16_gemv_kernel(
        input_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        output_ptr,
        stride_input_k: tl.constexpr,
        stride_qweight_k: tl.constexpr,
        stride_qweight_n: tl.constexpr,
        stride_qzeros_g: tl.constexpr,
        stride_qzeros_n: tl.constexpr,
        stride_scales_g: tl.constexpr,
        stride_scales_n: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        packed_offsets_n = (
            pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        )
        lane = tl.arange(0, 8)
        shifts = 4 * (lane // 2) + 16 * (lane % 2)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            inputs = tl.load(
                input_ptr + offsets_k * stride_input_k,
                mask=offsets_k < K,
                other=0.0,
            )
            packed_weight = tl.load(
                qweight_ptr
                + offsets_k[:, None] * stride_qweight_k
                + packed_offsets_n[None, :] * stride_qweight_n,
                mask=(offsets_k[:, None] < K)
                & (packed_offsets_n[None, :] * 8 < N),
                other=0,
            )
            quantized = (
                packed_weight[:, :, None] >> shifts[None, None, :]
            ) & 0xF
            quantized = tl.reshape(quantized, (BLOCK_K, BLOCK_N))
            group_id = k_start // GROUP_SIZE
            packed_zero = tl.load(
                qzeros_ptr
                + group_id * stride_qzeros_g
                + packed_offsets_n * stride_qzeros_n,
                mask=packed_offsets_n * 8 < N,
                other=0,
            )
            zeros = (packed_zero[:, None] >> shifts[None, :]) & 0xF
            zeros = tl.reshape(zeros, (BLOCK_N,))
            scales = tl.load(
                scales_ptr
                + group_id * stride_scales_g
                + offsets_n * stride_scales_n,
                mask=offsets_n < N,
                other=0.0,
            )
            weights = (
                (quantized.to(tl.float32) - zeros[None, :].to(tl.float32))
                * scales[None, :].to(tl.float32)
            ).to(tl.bfloat16)
            accumulator += tl.sum(
                inputs[:, None].to(tl.float32) * weights.to(tl.float32),
                axis=0,
            )

        tl.store(output_ptr + offsets_n, accumulator, mask=offsets_n < N)

    @triton.jit
    def _awq_w4a16_kernel(
        input_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        output_ptr,
        stride_input_m: tl.constexpr,
        stride_input_k: tl.constexpr,
        stride_qweight_k: tl.constexpr,
        stride_qweight_n: tl.constexpr,
        stride_qzeros_g: tl.constexpr,
        stride_qzeros_n: tl.constexpr,
        stride_scales_g: tl.constexpr,
        stride_scales_n: tl.constexpr,
        stride_output_s: tl.constexpr,
        stride_output_m: tl.constexpr,
        stride_output_n: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_split = tl.program_id(2)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        packed_offsets_n = (
            pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        )
        lane = tl.arange(0, 8)
        shifts = 4 * (lane // 2) + 16 * (lane % 2)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(
            pid_split * BLOCK_K,
            K,
            BLOCK_K * SPLIT_K,
        ):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            inputs = tl.load(
                input_ptr
                + offsets_m[:, None] * stride_input_m
                + offsets_k[None, :] * stride_input_k,
                mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
                other=0.0,
            )
            packed_weight = tl.load(
                qweight_ptr
                + offsets_k[:, None] * stride_qweight_k
                + packed_offsets_n[None, :] * stride_qweight_n,
                mask=(offsets_k[:, None] < K)
                & (packed_offsets_n[None, :] * 8 < N),
                other=0,
            )
            quantized = (
                packed_weight[:, :, None] >> shifts[None, None, :]
            ) & 0xF
            quantized = tl.reshape(quantized, (BLOCK_K, BLOCK_N))

            group_id = k_start // GROUP_SIZE
            packed_zero = tl.load(
                qzeros_ptr
                + group_id * stride_qzeros_g
                + packed_offsets_n * stride_qzeros_n,
                mask=packed_offsets_n * 8 < N,
                other=0,
            )
            zeros = (packed_zero[:, None] >> shifts[None, :]) & 0xF
            zeros = tl.reshape(zeros, (BLOCK_N,))
            scales = tl.load(
                scales_ptr
                + group_id * stride_scales_g
                + offsets_n * stride_scales_n,
                mask=offsets_n < N,
                other=0.0,
            )
            weights = (
                (quantized.to(tl.float32) - zeros[None, :].to(tl.float32))
                * scales[None, :].to(tl.float32)
            ).to(tl.bfloat16)
            accumulator += tl.dot(inputs, weights)

        tl.store(
            output_ptr
            + pid_split * stride_output_s
            + offsets_m[:, None] * stride_output_m
            + offsets_n[None, :] * stride_output_n,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )

    @triton.jit
    def _reduce_split_k_kernel(
        partial_ptr,
        output_ptr,
        stride_partial_s: tl.constexpr,
        stride_partial_m: tl.constexpr,
        stride_partial_n: tl.constexpr,
        stride_output_m: tl.constexpr,
        stride_output_n: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for split_id in tl.static_range(0, SPLIT_K):
            accumulator += tl.load(
                partial_ptr
                + split_id * stride_partial_s
                + offsets_m[:, None] * stride_partial_m
                + offsets_n[None, :] * stride_partial_n,
                mask=mask,
                other=0.0,
            )
        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_output_m
            + offsets_n[None, :] * stride_output_n,
            accumulator,
            mask=mask,
        )


def select_awq_kernel_config(k: int, n: int) -> tuple[int, int]:
    if n >= 16384:
        return 64, 1
    if k >= 8192:
        return 32, 8
    if k >= 4096 and n > 4096:
        return 32, 4
    return 32, 1


def is_awq_triton_available() -> bool:
    return triton is not None


def awq_triton_linear(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for the AWQ W4A16 kernel")
    if inputs.device.type != "cuda":
        raise ValueError("AWQ Triton inputs must be CUDA tensors")
    if inputs.ndim != 2:
        raise ValueError("AWQ Triton inputs must be two-dimensional")
    if inputs.dtype != torch.bfloat16 or scales.dtype != torch.bfloat16:
        raise ValueError("AWQ Triton currently requires bfloat16 activations/scales")
    if inputs.device != qweight.device:
        raise ValueError("AWQ Triton activations and weights must share one device")
    if group_size != 128:
        raise ValueError("AWQ Triton currently requires group_size=128")
    _validate_awq_tensors(qweight, qzeros, scales, group_size)
    if inputs.shape[1] != qweight.shape[0]:
        raise ValueError("AWQ Triton activation K must match qweight input size")

    m, k = inputs.shape
    n = scales.shape[1]
    block_m = 16
    block_n, split_k = select_awq_kernel_config(k, n)
    block_k = 32
    if k % block_k:
        raise ValueError("AWQ Triton K must be divisible by 32")
    output = torch.empty((m, n), dtype=inputs.dtype, device=inputs.device)
    if m == 1 and n >= 16384:
        gemv_grid = (triton.cdiv(n, block_n),)
        _awq_w4a16_gemv_kernel[gemv_grid](
            inputs,
            qweight,
            qzeros,
            scales,
            output,
            inputs.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            qzeros.stride(0),
            qzeros.stride(1),
            scales.stride(0),
            scales.stride(1),
            N=n,
            K=k,
            GROUP_SIZE=group_size,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
        return output
    if split_k == 1:
        partial = output
        stride_output_s = 0
    else:
        partial = torch.empty(
            (split_k, m, n),
            dtype=torch.float32,
            device=inputs.device,
        )
        stride_output_s = partial.stride(0)
    grid = (
        triton.cdiv(n, block_n),
        triton.cdiv(m, block_m),
        split_k,
    )
    _awq_w4a16_kernel[grid](
        inputs,
        qweight,
        qzeros,
        scales,
        partial,
        inputs.stride(0),
        inputs.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        scales.stride(0),
        scales.stride(1),
        stride_output_s,
        partial.stride(-2),
        partial.stride(-1),
        M=m,
        N=n,
        K=k,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=4,
        num_stages=2,
    )
    if split_k > 1:
        reduction_grid = (
            triton.cdiv(n, block_n),
            triton.cdiv(m, block_m),
        )
        _reduce_split_k_kernel[reduction_grid](
            partial,
            output,
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            output.stride(0),
            output.stride(1),
            M=m,
            N=n,
            SPLIT_K=split_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return output
