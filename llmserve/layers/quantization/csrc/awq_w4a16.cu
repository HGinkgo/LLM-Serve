#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kTileM = 16;
constexpr int kTileK = 16;
constexpr int kTileN = 64;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ uint32_t reorder_autoawq_word(uint32_t packed) {
  constexpr int shifts[8] = {0, 16, 4, 20, 8, 24, 12, 28};
  uint32_t reordered = 0;
#pragma unroll
  for (int logical = 0; logical < 8; ++logical) {
    reordered |= ((packed >> shifts[logical]) & 0xF) << (4 * logical);
  }
  return reordered;
}

__device__ __forceinline__ __nv_bfloat162 unpack_bfloat16_pair(
    uint32_t packed, int pair_index) {
  constexpr uint32_t kBfloat16Base = 0x43004300;
  int shift = pair_index * 8;
  uint32_t values = ((packed >> shift) & 0xF) |
                    (((packed >> (shift + 4)) & 0xF) << 16);
  union {
    uint32_t bits;
    __nv_bfloat162 value;
  } converted = {kBfloat16Base | values};
  union {
    uint32_t bits;
    __nv_bfloat162 value;
  } base = {kBfloat16Base};
  return __hsub2(converted.value, base.value);
}

__device__ __forceinline__ uint32_t extract_autoawq_value(
    const uint32_t* input, int row, int col, int packed_n) {
  constexpr int shifts[8] = {0, 16, 4, 20, 8, 24, 12, 28};
  uint32_t packed = input[row * packed_n + col / 8];
  return (packed >> shifts[col % 8]) & 0xF;
}

__device__ __forceinline__ uint32_t load_bfloat16_pair(
    const __nv_bfloat16* input, int row, int col, int m, int k) {
  if (row >= m) {
    return 0;
  }
  return *reinterpret_cast<const uint32_t*>(input + row * k + col);
}

__device__ __forceinline__ uint32_t dequantize_pair(
    uint32_t packed, int pair_index, __nv_bfloat162 zero,
    __nv_bfloat162 scale) {
  __nv_bfloat162 quantized = unpack_bfloat16_pair(packed, pair_index);
  __nv_bfloat162 weight = __hmul2(__hsub2(quantized, zero), scale);
  union {
    __nv_bfloat162 value;
    uint32_t bits;
  } converted = {weight};
  return converted.bits;
}

__device__ __forceinline__ void mma_bfloat16(const uint32_t* a,
                                             const uint32_t* b, float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1]));
}

__global__ void repack_qweight_kernel(const uint32_t* __restrict__ input,
                                      uint32_t* __restrict__ output, int k,
                                      int packed_n) {
  int n = packed_n * 8;
  int n_tiles = n / kTileN;
  int k_tile = blockIdx.x / n_tiles;
  int n_tile = blockIdx.x % n_tiles;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int group = lane / 4;
  int thread = lane % 4;
  int row0 = k_tile * kTileK + thread * 2;
  int row1 = row0 + 1;
  int row2 = row0 + 8;
  int row3 = row0 + 9;
  int col0 = n_tile * kTileN + warp * 16 + group;
  int col1 = col0 + 8;

  uint32_t repacked = 0;
  repacked |= extract_autoawq_value(input, row0, col0, packed_n) << 0;
  repacked |= extract_autoawq_value(input, row1, col0, packed_n) << 4;
  repacked |= extract_autoawq_value(input, row2, col0, packed_n) << 8;
  repacked |= extract_autoawq_value(input, row3, col0, packed_n) << 12;
  repacked |= extract_autoawq_value(input, row0, col1, packed_n) << 16;
  repacked |= extract_autoawq_value(input, row1, col1, packed_n) << 20;
  repacked |= extract_autoawq_value(input, row2, col1, packed_n) << 24;
  repacked |= extract_autoawq_value(input, row3, col1, packed_n) << 28;
  output[blockIdx.x * kThreads + threadIdx.x] = repacked;
}

__global__ void reorder_qzeros_kernel(const uint32_t* __restrict__ input,
                                      uint32_t* __restrict__ output,
                                      int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = reorder_autoawq_word(input[index]);
  }
}

template <bool kWriteFinal>
__global__ void awq_w4a16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16* __restrict__ output, float* __restrict__ partial, int m,
    int n, int k, int group_size, int split_k) {
  int n_tile = blockIdx.x;
  int m_tile = blockIdx.y;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  int group = lane / 4;
  int thread = lane % 4;
  int n_start = n_tile * kTileN;
  int m_start = m_tile * kTileM;
  int num_n_tiles = n / kTileN;
  int split = blockIdx.z;

  float accumulator0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float accumulator1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int col0 = n_start + warp * 16 + group;
  int col1 = col0 + 8;

  int num_quant_groups = k / group_size;
  int first_quant_group = num_quant_groups * split / split_k;
  int last_quant_group = num_quant_groups * (split + 1) / split_k;
  for (int quant_group = first_quant_group;
       quant_group < last_quant_group; ++quant_group) {
    uint32_t packed_zero0 = qzeros[quant_group * (n / 8) + col0 / 8];
    uint32_t packed_zero1 = qzeros[quant_group * (n / 8) + col1 / 8];
    int zero0 = (packed_zero0 >> ((col0 % 8) * 4)) & 0xF;
    int zero1 = (packed_zero1 >> ((col1 % 8) * 4)) & 0xF;
    __nv_bfloat162 zeros0 = __bfloat162bfloat162(
        __float2bfloat16_rn(static_cast<float>(zero0)));
    __nv_bfloat162 zeros1 = __bfloat162bfloat162(
        __float2bfloat16_rn(static_cast<float>(zero1)));
    __nv_bfloat162 scales0 =
        __bfloat162bfloat162(scales[quant_group * n + col0]);
    __nv_bfloat162 scales1 =
        __bfloat162bfloat162(scales[quant_group * n + col1]);

#pragma unroll
    for (int k_subtile = 0; k_subtile < group_size / kTileK; ++k_subtile) {
      int k_start = quant_group * group_size + k_subtile * kTileK;
      int row0 = m_start + group;
      int row1 = row0 + 8;
      int a_col0 = k_start + thread * 2;
      int a_col1 = a_col0 + 8;
      uint32_t a[4] = {
          load_bfloat16_pair(input, row0, a_col0, m, k),
          load_bfloat16_pair(input, row1, a_col0, m, k),
          load_bfloat16_pair(input, row0, a_col1, m, k),
          load_bfloat16_pair(input, row1, a_col1, m, k),
      };
      uint32_t packed =
          qweight[(((k_start / kTileK) * num_n_tiles + n_tile) * kWarps +
                   warp) *
                      32 +
                  lane];
      uint32_t b0[2] = {
          dequantize_pair(packed, 0, zeros0, scales0),
          dequantize_pair(packed, 1, zeros0, scales0),
      };
      uint32_t b1[2] = {
          dequantize_pair(packed, 2, zeros1, scales1),
          dequantize_pair(packed, 3, zeros1, scales1),
      };
      mma_bfloat16(a, b0, accumulator0);
      mma_bfloat16(a, b1, accumulator1);
    }
  }

  int output_row0 = m_start + group;
  int output_row1 = output_row0 + 8;
  int output_col0 = n_start + warp * 16 + thread * 2;
  int output_col1 = output_col0 + 8;
  if (output_row0 < m) {
    int index0 = output_row0 * n + output_col0;
    int index1 = output_row0 * n + output_col1;
    if constexpr (kWriteFinal) {
      output[index0] = __float2bfloat16_rn(accumulator0[0]);
      output[index0 + 1] = __float2bfloat16_rn(accumulator0[1]);
      output[index1] = __float2bfloat16_rn(accumulator1[0]);
      output[index1 + 1] = __float2bfloat16_rn(accumulator1[1]);
    } else {
      int split_offset = split * m * n;
      partial[split_offset + index0] = accumulator0[0];
      partial[split_offset + index0 + 1] = accumulator0[1];
      partial[split_offset + index1] = accumulator1[0];
      partial[split_offset + index1 + 1] = accumulator1[1];
    }
  }
  if (output_row1 < m) {
    int index0 = output_row1 * n + output_col0;
    int index1 = output_row1 * n + output_col1;
    if constexpr (kWriteFinal) {
      output[index0] = __float2bfloat16_rn(accumulator0[2]);
      output[index0 + 1] = __float2bfloat16_rn(accumulator0[3]);
      output[index1] = __float2bfloat16_rn(accumulator1[2]);
      output[index1 + 1] = __float2bfloat16_rn(accumulator1[3]);
    } else {
      int split_offset = split * m * n;
      partial[split_offset + index0] = accumulator0[2];
      partial[split_offset + index0 + 1] = accumulator0[3];
      partial[split_offset + index1] = accumulator1[2];
      partial[split_offset + index1 + 1] = accumulator1[3];
    }
  }
}

__global__ void reduce_split_k_kernel(const float* __restrict__ partial,
                                      __nv_bfloat16* __restrict__ output,
                                      int count, int split_k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  float value = 0.0f;
#pragma unroll
  for (int split = 0; split < 8; ++split) {
    if (split < split_k) {
      value += partial[split * count + index];
    }
  }
  output[index] = __float2bfloat16_rn(value);
}

void check_cuda_int32_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must use int32");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor repack_qweight(torch::Tensor qweight) {
  check_cuda_int32_matrix(qweight, "qweight");
  int64_t k = qweight.size(0);
  int64_t n = qweight.size(1) * 8;
  TORCH_CHECK(k % kTileK == 0, "qweight K must be divisible by 16");
  TORCH_CHECK(n % kTileN == 0, "qweight N must be divisible by 64");
  c10::cuda::CUDAGuard device_guard(qweight.device());
  auto output = torch::empty({k / kTileK, n / kTileN, kWarps, 32},
                             qweight.options());
  int blocks = (k / kTileK) * (n / kTileN);
  repack_qweight_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()), k,
      qweight.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor reorder_qzeros(torch::Tensor qzeros) {
  check_cuda_int32_matrix(qzeros, "qzeros");
  c10::cuda::CUDAGuard device_guard(qzeros.device());
  auto output = torch::empty_like(qzeros);
  int count = qzeros.numel();
  int blocks = (count + kThreads - 1) / kThreads;
  reorder_qzeros_kernel<<<blocks, kThreads, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor awq_linear(torch::Tensor input, torch::Tensor qweight,
                         torch::Tensor qzeros, torch::Tensor scales,
                         torch::Tensor workspace, int64_t group_size,
                         int64_t split_k) {
  TORCH_CHECK(input.is_cuda() && qweight.is_cuda() && qzeros.is_cuda() &&
                  scales.is_cuda(),
              "all AWQ CUDA tensors must be on CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "input must use bfloat16");
  TORCH_CHECK(scales.scalar_type() == at::kBFloat16,
              "scales must use bfloat16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt &&
                  qzeros.scalar_type() == at::kInt,
              "packed weights and zeros must use int32");
  TORCH_CHECK(input.dim() == 2 && qweight.dim() == 4 && qzeros.dim() == 2 &&
                  scales.dim() == 2,
              "invalid AWQ CUDA tensor rank");
  TORCH_CHECK(input.is_contiguous() && qweight.is_contiguous() &&
                  qzeros.is_contiguous() && scales.is_contiguous(),
              "AWQ CUDA tensors must be contiguous");
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == at::kFloat &&
                  workspace.is_contiguous(),
              "workspace must be a contiguous CUDA float32 tensor");
  TORCH_CHECK(group_size == 128, "group_size must be 128");
  TORCH_CHECK(split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8,
              "split_k must be 1, 2, 4, or 8");

  int64_t m = input.size(0);
  int64_t k = qweight.size(0) * kTileK;
  int64_t n = qweight.size(1) * kTileN;
  TORCH_CHECK(input.size(1) == k, "input K does not match qweight");
  TORCH_CHECK(qweight.size(2) == kWarps && qweight.size(3) == 32,
              "invalid repacked qweight tile shape");
  TORCH_CHECK(qzeros.sizes() == torch::IntArrayRef({k / group_size, n / 8}),
              "qzeros shape does not match qweight");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({k / group_size, n}),
              "scales shape does not match qweight");
  TORCH_CHECK((k / group_size) % split_k == 0,
              "quantization groups must be divisible by split_k");
  TORCH_CHECK(input.device() == qweight.device() &&
                  input.device() == qzeros.device() &&
                  input.device() == scales.device() &&
                  input.device() == workspace.device(),
              "all AWQ CUDA tensors must share one device");
  TORCH_CHECK(workspace.numel() >= (split_k > 1 ? split_k * m * n : 0),
              "workspace is too small for split-K output");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty({m, n}, input.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto input_ptr =
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
  auto qweight_ptr =
      reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
  auto qzeros_ptr =
      reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>());
  auto scales_ptr =
      reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  auto output_ptr =
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
  if (m > kTileM) {
    dim3 grid(n / kTileN, (m + kTileM - 1) / kTileM, 1);
    awq_w4a16_kernel<true><<<grid, kThreads, 0, stream>>>(
        input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, output_ptr, nullptr, m,
        n, k, group_size, 1);
  } else if (split_k == 1) {
    dim3 grid(n / kTileN, 1, 1);
    awq_w4a16_kernel<true><<<grid, kThreads, 0, stream>>>(
        input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, output_ptr, nullptr, m,
        n, k, group_size, split_k);
  } else {
    dim3 grid(n / kTileN, 1, split_k);
    awq_w4a16_kernel<false><<<grid, kThreads, 0, stream>>>(
        input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, nullptr,
        workspace.data_ptr<float>(), m, n, k, group_size, split_k);
    int count = m * n;
    constexpr int reduce_threads = 256;
    int reduce_blocks = (count + reduce_threads - 1) / reduce_threads;
    reduce_split_k_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        workspace.data_ptr<float>(), output_ptr, count, split_k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("repack_qweight", &repack_qweight);
  module.def("reorder_qzeros", &reorder_qzeros);
  module.def("linear", &awq_linear);
}
