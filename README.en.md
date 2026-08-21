<div align="center">

# LLM-Serve

An LLM runtime for studying and validating single-host GPU inference systems

<p>
  English |
  <a href="README.md">简体中文</a>
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

LLM-Serve is a single-host GPU LLM inference runtime that started from the [`nano-vllm`](https://github.com/Geeeone/nano-vllm) teaching skeleton. It focuses on independently verifiable systems work: scheduling, KV-cache lifecycle, continuous batching, quantized MoE execution, and single-model service governance. It is not a general-purpose multi-tenant production platform.

## Implemented

- **Engine**: paged KV cache, block tables, KV allocation/reclamation, iteration-level continuous batching, decode-first chunked prefill, a structured `SchedulerOutput`, and normal decode CUDA Graphs.
- **Serving**: OpenAI-compatible completions/chat, SSE streaming, cancellation, health/readiness, Prometheus metrics, in-flight admission, 429 overload responses, and step-boundary deadlines.
- **Quantized MoE**: a single-GPU Qwen3-30B-A3B GPTQ-Int4 path with group size 128, symmetric quantization, and act-order disabled; TinyGEMM is the comparison backend and Marlin is enabled through an explicitly loaded vLLM CUDA backend.
- **Reproducible experiments**: a fixed-trace service overload benchmark and a MoE Gate/Up fused-vs-unfused A/B benchmark. Configurations, metric definitions, and public evidence live in the [benchmark documentation](benchmarks/README.md).

## Execution Path

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

## Quick Start

Requirements: Linux, Python 3.10-3.12, and an NVIDIA CUDA GPU. The verified environment uses CUDA 12.8, PyTorch 2.7.1, Triton 3.3.1, and FlashAttention 2.8.3.

```bash
conda create -n LLM-Serve python=3.10 pip -y
conda activate LLM-Serve
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging wheel
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
llmserve check
```

Prepare a local Qwen3-8B Hugging Face checkpoint:

```bash
llmserve generate \
  --model /path/to/Qwen3-8B \
  --prompt "Explain PagedAttention in plain language." \
  --max-tokens 128
```

Python API:

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

## HTTP Serving

```bash
pip install -e '.[serve]'
llmserve serve \
  --model /path/to/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 0.0.0.0 --port 8000
```

The service exposes `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/health/live`, `/health/ready`, and `/metrics`. In-flight requests are bounded by default; overload returns `429` with `Retry-After`, and expired requests converge at the next Engine step boundary. HTTPS/TLS belongs to a reverse proxy or deployment environment.

## Verification And Results

```bash
CUDA_VISIBLE_DEVICES='' python -m tests.tier_runner core
CUDA_VISIBLE_DEVICES='' python -m tests.tier_runner extended
```

- [Benchmark guide](benchmarks/README.md)
- [Public result index](benchmarks/results/README.md)
- [MoE Gate/Up fusion evidence](benchmarks/results/moe-gate-up-fusion/README.md)
- [Test tiers](tests/README.md)

## Scope

- The primary target is single-host, single-GPU TP=1. The dual-GPU PD path is a constrained experiment, not Tensor Parallelism.
- Dense Qwen3 supports non-quantized checkpoints only. The MoE path is limited to the GPTQ-Int4 contract above, single-GPU TP=1, and eager execution.
- EAGLE3 and PD with Shared KV transport are retained but frozen; dynamic routing, 1P2D, and new PD features are out of scope.
- HTTP is currently single-model and text-only; authentication, tool calling, multimodality, and cross-node routing are out of scope.

## References

- [nano-vllm](https://github.com/Geeeone/nano-vllm): teaching skeleton.
- [vLLM](https://github.com/vllm-project/vllm): reference for serving interfaces, inference-system structure, and the Marlin backend.

## License

[MIT](LICENSE)
