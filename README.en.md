# ThrustLM

[English](README.en.md) | [简体中文](README.md)

ThrustLM is a single-GPU LLM inference engine built from the ground up for understanding how high-throughput serving works under the hood. It implements paged KV cache management, continuous batching, chunked prefill, serving-oriented benchmarking, and an experimental EAGLE-style speculative decoding path.

The initial skeleton was informed by the vLLM PagedAttention paper and the `nano-vllm` educational codebase. The scheduler changes, chunked prefill path, benchmark tooling, and speculative decoding runtime are independently designed and implemented in this repository.

The repository name and Python package are aligned as `ThrustLM` / `thrustlm`.

## Features

- PagedAttention-style KV cache management with block tables, reusable KV blocks, and prefix-cache-aware block allocation.
- Continuous batching with an iteration-level scheduler.
- Chunked prefill with decode-first scheduling, allowing long prompt prefill work to share iterations with active decode requests.
- Serving benchmark tooling with throughput, TTFT, ITL, TPOT, request latency, and a closed-loop steady-state mode that separates warmup, measurement, and drain phases.
- Qwen3 model support for local single-GPU experiments.

## Code Layout

- `thrustlm/engine/`: request scheduling, KV block management, target-model execution, and speculative runtime orchestration.
- `thrustlm/models/`: Qwen3 and EAGLE3 neural-network definitions plus checkpoint loading.
- `thrustlm/speculative/`: draft generation, verification sampling, fixed-tree algorithms, Tree KV management, and shared result types.
- `thrustlm/layers/`: attention, sampling, and model building blocks.
- `bench_serving.py`: finite-workload and closed-loop serving benchmarks.

`ModelRunner` owns the target-model execution resources. `SpeculativeExecutor` composes those resources into the EAGLE decode flow, while algorithmic code remains independent under `thrustlm/speculative/`.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Set a local model path:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
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

Measure steady-state continuous batching at a fixed concurrency:

```bash
python bench_serving.py \
  --model "$MODEL_PATH" \
  --arrival closed-loop \
  --max-concurrency 8 \
  --warmup-seconds 5 \
  --measurement-seconds 15 \
  --prompt-mode natural \
  --output-len 64
```

Closed-loop mode replaces each completed request until the measurement window ends, then stops admission and drains remaining requests. Its steady-state throughput counts only tokens emitted inside the measurement window; the existing overall summary is still included for comparison.

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
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

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

The EAGLE benchmark summary reports speculative batch size, acceptance rate, acceptance length, accepted tokens per step, draft tokens per step, and a timing breakdown for draft proposal, target verification, accept/reject, KV update, and trace overhead.

The validated 24GB configuration is Qwen3-8B with the RedHatAI Qwen3-8B EAGLE3 speculator, BF16 eager mode, and fixed `gamma=3`. In three output-256 runs, throughput improved by `1.20x` at batch 1 and `1.34x` at batch 4.

The fixed-tree path is a single-request experiment. Add `--speculative-tree-nodes 6 --argmax-sampler` to compare Tree-6 with linear EAGLE. It uses an all-layer Tree KV manager and fused commit kernel, but remains disabled by default and does not support multi-request trees.

## Notes

- The codebase retains the original MIT license.
