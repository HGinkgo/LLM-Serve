<div align="center">

# LLM-Serve

An educational runtime for studying LLM inference systems

<p>
  English |
  <a href="README.md">简体中文</a>
</p>

<p>
  <a href="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml"><img src="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg" alt="CPU tests"></a>
  <img src="https://img.shields.io/badge/Model-Qwen3--8B-6f42c1" alt="Model: Qwen3-8B">
  <img src="https://img.shields.io/badge/Runtime-PyTorch%20%7C%20Triton%20%7C%20CUDA-76b900" alt="Runtime: PyTorch Triton CUDA">
  <img src="https://img.shields.io/badge/Serving-Paged%20KV%20%7C%20Continuous%20Batching-0ea5e9" alt="Serving: Paged KV and continuous batching">
  <img src="https://img.shields.io/badge/Advanced-PD%20%7C%20EAGLE%20%7C%20AWQ-f59e0b" alt="Advanced: PD EAGLE AWQ">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</p>

</div>

LLM-Serve is an educational inference runtime centered on Qwen3-8B and single-host GPU serving. It started from the [`nano-vllm`](https://github.com/Geeeone/nano-vllm) teaching skeleton and has evolved around scheduling, KV cache management, speculative decoding, quantization, and Prefill/Decode disaggregation.

## Capabilities

- Paged KV cache, prefix cache, and iteration-level continuous batching.
- Decode-first chunked prefill with an explicit `SchedulerOutput` contract.
- EAGLE-style batched draft proposal, packed verification, and target-verify CUDA Graphs.
- Qwen3 AWQ W4A16 calibration, standard checkpoint export, and multiple Linear backends.
- Dual-GPU Prefill/Decode workers, shared-memory KV handoff, and backpressure.
- Poisson and closed-loop serving benchmarks with throughput, TTFT, TPOT, E2E, and queue metrics.

## Quick Start

LLM-Serve requires Linux, Python 3.10-3.12, and an NVIDIA CUDA GPU. The verified environment uses CUDA 12.8, PyTorch 2.7.1, Triton 3.3.1, and FlashAttention 2.8.3.

```bash
conda create -n LLM-Serve python=3.10 pip -y
conda activate LLM-Serve

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging wheel
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .

llmserve check
```

FlashAttention wheels are fetched from GitHub Releases. On restricted networks, provide a matching `cu12 / torch2.7 / Python 3.10` wheel in advance. Source builds require CUDA Toolkit 12.8 and must not use a system CUDA 13 toolchain.

Prepare a local Qwen3-8B Hugging Face checkpoint, then run:

```bash
llmserve generate \
  --model /path/to/Qwen3-8B \
  --prompt "Explain PagedAttention in plain language." \
  --max-tokens 128
```

The Python API remains available:

```python
from llmserve import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-8B")
try:
    outputs = llm.generate(
        ["Explain continuous batching in one paragraph."],
        SamplingParams(temperature=0.6, max_tokens=128),
    )
    print(outputs[0]["text"])
finally:
    llm.exit()
```

## Documentation

- [Basic Python example](example.py)
- [Prefill/Decode examples](examples/)
- [Benchmark guide and result index](benchmarks/README.md)

Performance numbers are intentionally kept out of the root README. Benchmark configurations, metric semantics, and public results live in the benchmark documentation.

## Scope

- The primary target is Qwen3-8B with single-GPU TP=1. The dual-GPU path is for Prefill/Decode disaggregation, not Tensor Parallelism.
- EAGLE, chunked prefill, KV capacity admission, and speculative CUDA Graphs are explicit opt-in features.
- AWQ is selected from checkpoint metadata. An OpenAI-compatible HTTP API is intentionally out of scope.

## License

[MIT](LICENSE)
