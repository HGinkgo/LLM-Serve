<div align="center">

# LLM-Serve

面向单机 GPU 推理系统学习与验证的 LLM Runtime

<p>
  <a href="README.en.md">English</a> |
  简体中文
</p>

<p>
  <a href="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml"><img src="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg" alt="CPU tests"></a>
  <img src="https://img.shields.io/badge/Model-Qwen3%20%7C%20Qwen3--MoE-6f42c1" alt="Model: Qwen3 and Qwen3 MoE">
  <img src="https://img.shields.io/badge/Runtime-PyTorch%20%7C%20CUDA-76b900" alt="Runtime: PyTorch CUDA">
  <img src="https://img.shields.io/badge/Serving-Paged%20KV%20%7C%20Continuous%20Batching-0ea5e9" alt="Serving: Paged KV and continuous batching">
  <img src="https://img.shields.io/badge/Quantization-GPTQ%20%7C%20Marlin-f59e0b" alt="Quantization: GPTQ and Marlin">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</p>

</div>

LLM-Serve 是一个单机 GPU LLM 推理 Runtime，起始于 [`nano-vllm`](https://github.com/Geeeone/nano-vllm) 教学骨架。项目关注推理系统中可独立验证的核心问题：调度、KV Cache 生命周期、连续批处理、量化 MoE 执行和单模型服务治理；它不是面向多租户生产环境的通用平台。

## 已实现能力

- **Engine**：Paged KV Cache、block table、KV 分配/回收、Iteration-level Continuous Batching、Decode-first Chunked Prefill、结构化 `SchedulerOutput` 和普通 Decode CUDA Graph。
- **Serving**：OpenAI-compatible completions/chat、SSE streaming、请求取消、health/ready、Prometheus metrics、in-flight admission、429 overload response 和 step-boundary deadline。
- **Quantized MoE**：Qwen3-30B-A3B GPTQ-Int4 单卡实验路径；固定 group size 128、对称量化、关闭 act-order，支持 TinyGEMM 对照和显式加载的 vLLM Marlin CUDA backend。
- **可复现实验**：固定 trace 的服务过载 benchmark，以及 MoE Gate/Up fused-vs-unfused A/B benchmark。所有公开数据、口径和配置都放在 [benchmark 文档](benchmarks/README.md) 中。

## 执行路径

```text
HTTP / Python API
        |
  EngineServiceRuntime
        |
 Scheduler -> Paged KV / Block Table -> Model Runner
                                      |
                   Dense Qwen3 / MoE Router -> Expert GEMM
                                      |
                         PyTorch / CUDA / Marlin
```

## 快速开始

要求 Linux、Python 3.10-3.12 和 NVIDIA CUDA GPU。已验证环境为 CUDA 12.8、PyTorch 2.7.1、Triton 3.3.1、FlashAttention 2.8.3。

```bash
conda create -n LLM-Serve python=3.10 pip -y
conda activate LLM-Serve
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging wheel
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
llmserve check
```

准备本地 Qwen3-8B Hugging Face checkpoint：

```bash
llmserve generate \
  --model /path/to/Qwen3-8B \
  --prompt "用通俗的语言解释 PagedAttention" \
  --max-tokens 128
```

Python API：

```python
from llmserve import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-8B")
try:
    outputs = llm.generate(["Explain continuous batching."],
                           SamplingParams(max_tokens=128))
    print(outputs[0]["text"])
finally:
    llm.exit()
```

## HTTP 服务

```bash
pip install -e '.[serve]'
llmserve serve \
  --model /path/to/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 0.0.0.0 --port 8000
```

服务提供 `/v1/models`、`/v1/completions`、`/v1/chat/completions`、`/health/live`、`/health/ready` 和 `/metrics`。默认限制在途请求数量；超载返回 `429` 和 `Retry-After`，超时在下一个 Engine step 边界收敛。HTTPS/TLS 由反向代理或部署环境负责。

## 验证与结果

```bash
CUDA_VISIBLE_DEVICES='' python -m tests.tier_runner core
CUDA_VISIBLE_DEVICES='' python -m tests.tier_runner extended
```

- [Benchmark 使用说明](benchmarks/README.md)
- [公开结果索引](benchmarks/results/README.md)
- [MoE Gate/Up 融合证据](benchmarks/results/moe-gate-up-fusion/README.md)
- [测试分层](tests/README.md)

## 范围边界

- 主要目标是单机、单卡 TP=1；双卡 PD 路径是受限实验，不是 Tensor Parallel 实现。
- Dense Qwen3 仅支持非量化 checkpoint；MoE 仅支持上述 GPTQ-Int4 合约，当前限制为单卡 TP=1、eager 执行。
- EAGLE3 与 PD + Shared KV Transport 保留但冻结，不继续扩展动态路由、1P2D 或新的 PD 功能。
- HTTP 层是单模型、纯文本服务，不包含鉴权、工具调用、多模态或跨节点路由。

## 参考

- [nano-vllm](https://github.com/Geeeone/nano-vllm)：教学骨架。
- [vLLM](https://github.com/vllm-project/vllm)：服务接口、推理系统和 Marlin backend 的参考实现。

## License

[MIT](LICENSE)
