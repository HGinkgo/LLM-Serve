# LLM-Serve

[English](README.en.md) | [简体中文](README.md)

[![CPU tests](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml)

LLM-Serve is an educational single-GPU inference runtime focused on paged KV cache management, continuous batching, chunked prefill, serving-oriented benchmarking, and EAGLE-style speculative decoding.

The initial skeleton was informed by the vLLM PagedAttention paper and `nano-vllm`. The scheduler changes, chunked prefill path, benchmark system, and speculative decoding runtime are independently designed and implemented in this repository.

## Features

- PagedAttention-style KV cache allocation, recycling, block tables, and prefix cache.
- Iteration-level continuous batching with explicit prefill/decode groups.
- Decode-first chunked prefill for mixed prefill/decode batches.
- EAGLE-style batched draft proposal, packed target verification, per-request draft KV, greedy verification, and timing metrics.
- Reproducible Poisson request-rate and closed-loop concurrency suites with throughput, goodput, TTFT, TPOT, burst ITL, output-event latency, E2E, queue depth, and speculative metrics.
- A validated single-GPU Qwen3-8B BF16 path and optional fixed-tree experiments.

## Layout

- `llmserve/engine/`: scheduling, KV block management, target execution, and speculative orchestration.
- `llmserve/models/`: Qwen3 and EAGLE3 definitions and checkpoint loading.
- `llmserve/speculative/`: draft, verification sampling, fixed trees, and Tree KV management.
- `llmserve/layers/`: attention, linear, sampling, and other model building blocks.
- `benchmarks/`: workloads, arrivals, metrics, point/suite runners, and public results.
- `tests/`: CPU tests plus optional checkpoint and CUDA kernel coverage.

## Quick Start

```bash
pip install -e .

export MODEL_PATH=/path/to/Qwen3-8B
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python example.py
```

Run the four-point GPU smoke suite:

```bash
python -m benchmarks.run_suite \
  --suite benchmarks/suites/smoke.json \
  --output-dir /tmp/llmserve-smoke \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --allow-dirty
```

Run the formal Poisson suite on a clean commit:

```bash
python -m benchmarks.run_suite \
  --suite benchmarks/suites/formal-poisson.json \
  --output-dir /tmp/llmserve-formal-poisson \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --resume
```

`formal-closed-loop.json` provides the fixed-concurrency complement. When running both suites concurrently on two GPUs, pass distinct distributed endpoints such as `tcp://localhost:2333` and `tcp://localhost:2334`. Each point runs in an independent subprocess. See [`benchmarks/README.md`](benchmarks/README.md) for schemas and execution details.

## Results

The public results use commit `ad35e65`, Qwen3-8B with the RedHatAI Qwen3-8B EAGLE3 speculator, BF16 eager mode, fixed `gamma=3`, argmax sampling, and one RTX 3090 24GB per suite. Every configuration has three independent runs.

### EAGLE

The decode-heavy profile is `256 input / 256 output`:

| Closed-loop concurrency | Baseline output tok/s | EAGLE output tok/s | Throughput ratio | E2E P99 ratio |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 25.08 | 41.01 | **1.635x** | 0.929x |
| 4 | 89.20 | 153.40 | **1.720x** | 1.257x |
| 8 | 172.36 | 267.39 | **1.551x** | 1.421x |

At Poisson request rates `{0.25, 0.75, 1.25}`, finite-workload output throughput improves by only `1.025x-1.042x`, while E2E P99 is `1.068x-1.220x` of baseline. The result is deliberately workload-specific: saturated capacity gains do not imply lower online request latency.

### Chunked Prefill

The mixed profile combines 80% `128 input / 128 output` requests with 20% `4096 input / 128 output` requests:

| Closed-loop concurrency | Output throughput ratio | TTFT P99 change | Short TTFT P99 change |
| :--- | ---: | ---: | ---: |
| 4 | 1.004x | **-20.1%** | **-19.7%** |
| 8 | 1.044x | **-6.3%** | **-21.0%** |
| 16 | 1.092x | **-20.4%** | **-17.2%** |

At Poisson rates `{0.5, 1.5, 2.5}`, throughput remains effectively flat (`0.992x-1.000x`), while TPOT P99 drops by about `19%-20%` and E2E P99 by about `14%-17%`. Chunked prefill is therefore presented as a scheduling and tail-latency mechanism, not an unconditional throughput optimization.

All 72 sanitized run JSON files, per-run CSVs, three-run aggregates, and manifests are published under [`benchmarks/results/`](benchmarks/results/).

## Metric Semantics

- `burst_itl`: adjacent per-token availability times; speculative bursts may include `0 ms` samples.
- `output_event_latency`: adjacent output events, recorded once per request per emitting engine step.
- `speculative_step_latency`: complete draft/verify/accept/KV step cost.
- `TPOT`: request-level average time per output token.
- Closed-loop throughput covers the measurement window. Latency and acceptance cover only requests whose arrival and finish both fall inside that window, with `latency_sample_requests` reported explicitly.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests
```

Enable real-checkpoint tests explicitly:

```bash
export LLMSERVE_TEST_TARGET_MODEL=/path/to/Qwen3-8B
export LLMSERVE_TEST_SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3
python -m unittest discover -s tests
```

CUDA-specific Tree KV tests run only when CUDA is available. GitHub Actions runs the remaining tests with CPU PyTorch.

## Scope

- The primary target is single-GPU Qwen3-8B; two non-NVLink GPUs are not presented as a tensor-parallel performance platform.
- Speculative CUDA Graph is not implemented. Fixed-tree speculation remains disabled by default.
- The project intentionally omits an OpenAI-compatible HTTP layer; benchmarks drive the in-process runtime directly.
- The codebase retains the original MIT license.
