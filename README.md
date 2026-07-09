# ThrustLM

English | [简体中文](README.zh-CN.md)

ThrustLM is a single-GPU LLM inference engine built from the ground up for understanding how high-throughput serving works under the hood. It implements paged KV cache management, continuous batching, chunked prefill, serving-oriented benchmarking, and an experimental EAGLE-style speculative decoding path.

The initial skeleton was informed by the vLLM PagedAttention paper and the `nano-vllm` educational codebase. The scheduler changes, chunked prefill path, benchmark tooling, and speculative decoding runtime are independently designed and implemented in this repository.

The repository name and Python package are aligned as `ThrustLM` / `thrustlm`.

## Features

- PagedAttention-style KV cache management with block tables, reusable KV blocks, and prefix-cache-aware block allocation.
- Continuous batching with an iteration-level scheduler.
- Chunked prefill with decode-first scheduling, allowing long prompt prefill work to share iterations with active decode requests.
- Serving benchmark tooling with request-level metrics including throughput, TTFT, ITL, TPOT, request latency, wall time, and success/failure counts.
- Qwen3 model support for local single-GPU experiments.

## Experimental

- EAGLE-style speculative decoding MVP with draft proposal, target verification, draft KV state, merged correction handling, and acceptance metrics.
- Speculative tracing and timing breakdown for inspecting draft tokens, target verification, acceptance length, and per-step runtime cost.
- Greedy/top-k style verification is the primary path for current hot-vocabulary EAGLE3 draft checkpoints.
- Probability rejection sampling utilities are kept for algorithm study and controlled experiments, but are not the recommended path for 32K hot-vocabulary draft heads.

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

Run the experimental EAGLE speculative path with a compatible draft checkpoint:

```bash
export SPECULATIVE_MODEL=/path/to/eagle3-draft

python bench_serving.py \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --speculative-gamma 3 \
  --speculative-accept-mode greedy \
  --speculative-trace \
  --num-requests 1 \
  --input-len 128 \
  --output-len 64 \
  --arrival all \
  --enforce-eager
```

The EAGLE benchmark summary reports acceptance rate, acceptance length, accepted tokens per step, draft tokens per step, and a speculative timing breakdown for draft proposal, target verification, accept/reject, KV update, and trace overhead.

## Notes

- The codebase retains the original MIT license.
