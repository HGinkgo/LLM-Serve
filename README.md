# ThrustLM

ThrustLM is a single-GPU LLM inference engine built from the ground up for understanding how high-throughput serving works under the hood. It implements paged KV cache management, continuous batching, chunked prefill, and serving-oriented benchmarking from scratch. The initial skeleton was informed by the vLLM PagedAttention paper and the `nano-vllm` educational codebase; all subsequent scheduler, chunked prefill, benchmark, and current speculative decoding work is being independently designed and implemented in this repository.

The repository name and Python package are aligned as `ThrustLM` / `thrustlm`.

## Features

- PagedAttention-style KV cache management with block tables, reusable KV blocks, and prefix-cache-aware block allocation.
- Continuous batching with an iteration-level scheduler.
- Chunked prefill with decode-first scheduling, allowing long prompt prefill work to share iterations with active decode requests.
- Serving benchmark tooling with request-level metrics including throughput, TTFT, ITL, TPOT, request latency, wall time, and success/failure counts.
- Qwen3 model path for local single-GPU experiments.

## In Development

- Speculative decoding with draft/target model verification.

## Architecture

A request enters ThrustLM through `LLMEngine`, is represented as a `Sequence`, and is admitted into the scheduler's waiting queue. The scheduler repeatedly builds iteration-level batches from waiting prefill work and running decode work. KV memory is assigned through a block-based manager, so each sequence carries a logical block table instead of owning contiguous KV storage. `ModelRunner` then prepares prefill, decode, or mixed chunked-prefill inputs, executes the model, samples the next token, and returns results to the engine. After each iteration, the scheduler updates cached-token progress, appends generated tokens, releases finished KV blocks, and the engine records request-level serving metrics.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Set a local model path:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
```

Run a small generation example:

```bash
python example.py
```

Run a small serving benchmark:

```bash
python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 8 \
  --input-len 128 \
  --output-len 128 \
  --arrival all
```

Run the chunked prefill path:

```bash
python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 8 \
  --input-len 4096 \
  --output-len 128 \
  --arrival poisson \
  --request-rate 2 \
  --max-model-len 4352 \
  --max-num-batched-tokens 512 \
  --enable-chunked-prefill
```

Benchmark documentation:

- `docs/benchmark-v1-baseline.md`
- `docs/chunked-prefill-stage2.md`
- `docs/env-3090-ubuntu22.md`

## Notes

- Raw local experiment artifacts are kept under `experiment-data/` and are not committed.
- Agent/project memory is kept under `.agent/` and is not committed.
- Some historical commands still use the local conda environment name `nano-vllm`; that is just the environment label, not the project/package name.
- The codebase retains the original MIT license.
