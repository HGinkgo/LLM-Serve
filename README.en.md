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
  <img src="https://img.shields.io/badge/Advanced-PD%20%7C%20EAGLE-f59e0b" alt="Advanced: PD EAGLE">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</p>

</div>

LLM-Serve is an educational inference runtime centered on Qwen3-8B and single-host GPU serving. It started from the [`nano-vllm`](https://github.com/Geeeone/nano-vllm) teaching skeleton. The current line of work is a checkable single-model serving loop: scheduling, KV lifecycle, overload protection, and reproducible experiments, rather than a claim to be a production multi-tenant platform.

## Current Baseline

- Paged KV cache, block tables, KV allocation/reclamation, and iteration-level continuous batching.
- Decode-first chunked prefill, a structured `SchedulerOutput` contract, and normal decode CUDA Graphs.
- A single-model OpenAI-compatible HTTP/SSE service: cancellation, health/readiness, Prometheus metrics, in-flight admission, 429 overload responses, and step-boundary deadlines.
- A fixed-Poisson-trace service benchmark comparing unbounded admission with an explicit bound, including client-observed TTFT/TPOT, output/request throughput, queues, rejection, and timeout.

## Optional And Frozen Experiments

- EAGLE3 linear speculative decoding is off by default and kept for controlled A/B measurements.
- PD with Shared KV transport is retained but frozen. Dynamic routing, 1P2D, and new PD features are out of scope.
- This branch also supports a single-GPU Qwen3-30B-A3B GPTQ-Int4 MoE
  experiment. Its checkpoint contract is fixed to group size 128, symmetric
  quantization, and no act-order. TinyGEMM is the default comparison backend;
  Marlin is enabled by explicitly loading the validated vLLM 0.9.1 CUDA
  extension.

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

For custom continuous-batching loops, use `add_request()` / `step()` and
`abort_request(request_id)`. Cancellation takes effect between `step()` calls
and does not interrupt an in-flight GPU step; Baseline, EAGLE, and PD use the
same semantics. A PD Worker or RPC failure terminates the current
`PDServingEngine` and marks unfinished requests as `failed`; create a new
Engine instance before serving more requests.

## HTTP Serving

Install the optional web-serving dependencies:

```bash
pip install -e '.[serve]'
```

The default deployment is the single-GPU Collocated Runtime with Paged KV,
continuous batching, decode-first chunked prefill, KV admission, and normal
decode CUDA Graph enabled:

```bash
llmserve serve \
  --model /path/to/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 0.0.0.0 --port 8000
```

The service exposes `/v1/models`, `/v1/completions`, `/v1/chat/completions`,
`/health/live`, `/health/ready`, and `/metrics`. `stream: true` relays real
Server-Sent Event token output. Disconnecting a client cancels its request at
the next Engine step and releases request state.

This is an HTTP service layer, not TLS termination. HTTPS certificates and TLS
belong to a reverse proxy or deployment environment; the project validates the
single-model request lifecycle and Runtime behavior itself.

The service limits total queued and active requests to the Runtime serving
capacity by default. Overload returns `429` with `Retry-After: 1` rather than
growing an unbounded queue. Use `--max-inflight-requests` for a smaller service
budget; `--request-timeout-seconds` cancels expired requests at the next Engine
step boundary.

```bash
curl http://127.0.0.1:8000/health/ready

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Explain continuous batching"}],"max_tokens":64,"temperature":0.6,"stream":true}'
```

`pd-shared` is an explicit dual-GPU deployment mode, not an automatic routing
policy. It uses separate Prefill/Decode workers, two pinned shared-memory KV
slots, descriptor-only handoff, and ACK/backpressure:

```bash
CUDA_VISIBLE_DEVICES=0,1 llmserve serve \
  --mode pd-shared \
  --model /path/to/Qwen3-8B \
  --pd-prefill-gpu 0 --pd-decode-gpus 1 \
  --pd-prefill-batch-size 4 \
  --pd-kv-slot-capacity-tokens 8192
```

## Documentation

- [Basic Python example](example.py)
- [Prefill/Decode examples](examples/)
- [Benchmark guide and result index](benchmarks/README.md)
- [Test tiers and commands](tests/README.md)

Performance numbers are intentionally kept out of the root README. Benchmark configurations, metric semantics, and public results live in the benchmark documentation.

## Scope

- The primary target is Qwen3-8B with single-GPU TP=1. The dual-GPU path is for Prefill/Decode disaggregation, not Tensor Parallelism.
- EAGLE and PD are experimental or constrained paths. Chunked prefill, KV capacity admission, and normal decode CUDA Graphs are part of the single-host serving baseline.
- The dense Qwen3 path only supports non-quantized checkpoints. The MoE path
  only supports the GPTQ-Int4 contract above and is limited to single-GPU TP=1
  eager execution. The HTTP service is currently single-model and text-only;
  it does not provide authentication, tool calling, multimodality, or
  cross-node routing.
- PD serving and EAGLE are not coupled yet. `pd-shared` rejects `--speculative-model` rather than presenting an unimplemented combination as supported.

## License

[MIT](LICENSE)
