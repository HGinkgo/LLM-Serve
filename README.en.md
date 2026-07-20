# LLM-Serve

[English](README.en.md) | [简体中文](README.md)

[![CPU tests](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml)

LLM-Serve is an educational single-GPU inference runtime extended from the `nano-vllm` skeleton. It focuses on paged KV cache management, continuous batching, chunked prefill, serving-oriented benchmarking, and EAGLE-style speculative decoding.

The initial skeleton was informed by the vLLM PagedAttention paper and the `nano-vllm` educational codebase. The scheduler changes, chunked prefill path, benchmark tooling, and speculative decoding runtime are independently designed and implemented in this repository.

The repository name and Python package are aligned as `LLM-Serve` / `llmserve`.

## Features

- PagedAttention-style KV cache management with block tables, reusable KV blocks, and prefix-cache-aware block allocation.
- Continuous batching with an iteration-level scheduler.
- Chunked prefill with decode-first scheduling, allowing long prompt prefill work to share iterations with active decode requests.
- Serving benchmark tooling with throughput, TTFT, burst ITL, output-event latency, speculative step latency, TPOT, request latency, and a closed-loop steady-state mode that separates warmup, measurement, and drain phases.
- EAGLE-style speculative decoding with batched draft proposal, packed target verification, per-request draft KV, greedy verification, and acceptance and timing metrics.
- Qwen3 model support for local single-GPU experiments.

## Code Layout

- `llmserve/engine/`: request scheduling, KV block management, target-model execution, and speculative runtime orchestration.
- `llmserve/models/`: Qwen3 and EAGLE3 neural-network definitions plus checkpoint loading.
- `llmserve/speculative/`: draft generation, verification sampling, fixed-tree algorithms, Tree KV management, and shared result types.
- `llmserve/layers/`: attention, sampling, and model building blocks.
- `bench_serving.py`: finite-workload and closed-loop serving benchmarks.
- `tests/`: CPU unit tests, optional real-checkpoint integration tests, and CUDA kernel tests.
- `scripts/summarize_benchmarks.py`: produces sanitized representative JSON and per-run CSV files.
- `benchmarks/`: public benchmark semantics and results.

`ModelRunner` owns the target-model execution resources. `SpeculativeExecutor` composes those resources into the EAGLE decode flow, while algorithmic code remains independent under `llmserve/speculative/`.

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
  --workload-name prefill-injection \
  --arrival prefill-injection \
  --num-requests 5 \
  --input-len 128 \
  --long-input-len 4096 \
  --injection-delay 1 \
  --output-len 256 \
  --max-model-len 4608 \
  --max-num-batched-tokens 512 \
  --enable-chunked-prefill
```

This workload starts four short-prompt requests, then injects one long prompt into their active decode batch. Remove `--enable-chunked-prefill` to run the matching baseline.

Run the EAGLE speculative path with a compatible draft checkpoint:

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

The Stage 4 three-workload evidence, 18 formal runs, summary CSV, and sanitized representative JSON files are in [`benchmarks/results/`](benchmarks/results/).

## Tests

CPU unit tests do not require FlashAttention or local model checkpoints:

```bash
python -m unittest discover -s tests
```

Enable the real EAGLE checkpoint integration tests explicitly:

```bash
export LLMSERVE_TEST_TARGET_MODEL=/path/to/Qwen3-target
export LLMSERVE_TEST_SPECULATIVE_MODEL=/path/to/Qwen3-EAGLE3-speculator
python -m unittest discover -s tests
```

Tree KV CUDA kernel tests run only when CUDA is available. GitHub Actions runs the remaining tests with CPU PyTorch and uploads the complete `unittest` output for the tested commit.

## Notes

- The codebase retains the original MIT license.
