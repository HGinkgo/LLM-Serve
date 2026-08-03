# LLM-Serve

[English](README.en.md) | [简体中文](README.md)

[![CPU tests](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml)

LLM-Serve is an educational inference runtime with a single-GPU baseline and a dual-GPU Prefill/Decode serving path. It focuses on paged KV cache management, continuous batching, chunked prefill, serving-oriented benchmarking, EAGLE-style speculative decoding, and AWQ W4A16 inference.

The initial skeleton was informed by the vLLM PagedAttention paper and `nano-vllm`. The scheduler changes, chunked prefill path, benchmark system, and speculative decoding runtime are independently designed and implemented in this repository.

## Features

- PagedAttention-style KV cache allocation, recycling, block tables, and prefix cache.
- Iteration-level continuous batching with explicit prefill/decode groups.
- Decode-first chunked prefill for mixed prefill/decode batches.
- EAGLE-style batched draft proposal, packed target verification, per-request draft KV, greedy verification, and timing metrics; an explicit opt-in CUDA Graph path for linear target verification.
- Qwen3 AWQ W4A16 calibration, standard AutoAWQ GEMM checkpoint export, reference/Triton/CUDA Linear backends, and KV capacity admission.
- Dual-GPU PD serving with independent Prefill/Decode workers, logical KV handoff, reusable pinned shared-memory slots, ACK/backpressure pipelining, and role-specific Decode CUDA Graphs.
- Reproducible Poisson request-rate and closed-loop concurrency suites with throughput, goodput, TTFT, TPOT, burst ITL, output-event latency, E2E, queue depth, and speculative metrics.
- A validated single-GPU Qwen3-8B BF16 path and optional fixed-tree experiments.

## Layout

- `llmserve/engine/`: scheduling, KV block management, target execution, and speculative orchestration.
- `llmserve/models/`: Qwen3 and EAGLE3 definitions and checkpoint loading.
- `llmserve/speculative/`: draft, verification sampling, fixed trees, and Tree KV management.
- `llmserve/quantization/`: Qwen3 AWQ calibration, layer-wise quantization, checkpoint export, and quality evaluation.
- `llmserve/pd/`: Prefill/Decode protocols, logical KV handoff, shared slots, worker lifecycle, and serving orchestration.
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

Reproduce the Stage 8 target-verify CUDA Graph comparison with `benchmarks/suites/stage8-graph-formal.json` from a clean `3bb5d21` commit; it runs three repetitions at concurrency 1/4/8.

The dual-GPU KV transport comparison uses `benchmarks/suites/pd-kv-pipeline-formal.json`. Prefill and Decode workers use GPU 0/1 by default, and the matrix isolates inline Queue Tensor transfer versus pinned shared-memory slots.

## Results

The public results use commit `ad35e65`, Qwen3-8B with the RedHatAI Qwen3-8B EAGLE3 speculator, BF16 eager mode, fixed `gamma=3`, argmax sampling, and one RTX 3090 24GB per suite. Every configuration has three independent runs.

### Dual-GPU PD KV Pipeline

The PD comparison uses Qwen3-8B BF16 on two RTX 3090 GPUs, `128 input / 64 output`, Prefill batch 4, and Decode CUDA Graphs. The only changed variable is KV transport: the baseline sends tensors through a process Queue, while the shared path sends descriptors for two reusable pinned shared-memory slots and reclaims them through acknowledgements.

| Concurrency | Inline req/s | Shared req/s | Throughput gain | TTFT P50 |
| :--- | ---: | ---: | ---: | ---: |
| 32 | 16.29 | **17.89** | **1.098x** | 207 -> 132 ms |
| 48 | 18.62 | **20.71** | **1.112x** | 227 -> 138 ms |
| 64 | 21.31 | **28.11** | **1.319x** | 639 -> 136 ms |

At concurrency 64, Prefill response-queue time falls from `47.4 ms` to `0.25 ms`, Decode admission from `14.5 ms` to `1.13 ms`, and Prefill roundtrip from `186.6 ms` to `134.2 ms`; model forward and KV export/copy are unchanged. The gain therefore comes from removing large cross-process Tensor serialization and handoff blocking, not faster model computation. All 18 formal points completed, with no inline fallback on the shared path and all slots reclaimed. See [`benchmarks/results/pd-kv-pipeline-formal/`](benchmarks/results/pd-kv-pipeline-formal/).

### EAGLE

The decode-heavy profile is `256 input / 256 output`:

| Closed-loop concurrency | Baseline output tok/s | EAGLE output tok/s | Throughput ratio | E2E P99 ratio |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 25.08 | 41.01 | **1.635x** | 0.929x |
| 4 | 89.20 | 153.40 | **1.720x** | 1.257x |
| 8 | 172.36 | 267.39 | **1.551x** | 1.421x |

At Poisson request rates `{0.25, 0.75, 1.25}`, finite-workload output throughput improves by only `1.025x-1.042x`, while E2E P99 is `1.068x-1.220x` of baseline. The result is deliberately workload-specific: saturated capacity gains do not imply lower online request latency.

### Target Verify CUDA Graph

Stage 8 isolates target-verify eager versus target-verify CUDA Graph under the same EAGLE workload. Both variants use `enforce_eager=true`, so ordinary decode CUDA Graph is excluded from the comparison. The formal suite uses `128 input / 128 output`, `gamma=3`, closed-loop concurrency `{1,4,8}`, and three repetitions per point:

| Concurrency | Eager output tok/s | Graph output tok/s | Throughput gain | Verify speedup | Graph hit rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.0 | **69.4** | **1.779x** | 2.17x | 98% |
| 4 | 139.8 | **217.9** | **1.558x** | 2.00x | 93% |
| 8 | 241.5 | **342.6** | **1.419x** | 1.79x | 84%-85% |

The backend captures six graphs for `batch={1,4,8}` and `context frontier={256,1024}`. Unsupported shapes and speculative reservation overflow fall back to eager. The gain decreases with batch size, while draft proposal and target decode remain nearly unchanged; target verification is still the dominant stage. The full manifest, CSV files, and 18 sanitized run JSON files are published in [`benchmarks/results/stage8-graph-formal/`](benchmarks/results/stage8-graph-formal/).

### Chunked Prefill

The mixed profile combines 80% `128 input / 128 output` requests with 20% `4096 input / 128 output` requests:

| Closed-loop concurrency | Output throughput ratio | TTFT P99 change | Short TTFT P99 change |
| :--- | ---: | ---: | ---: |
| 4 | 1.004x | **-20.1%** | **-19.7%** |
| 8 | 1.044x | **-6.3%** | **-21.0%** |
| 16 | 1.092x | **-20.4%** | **-17.2%** |

At Poisson rates `{0.5, 1.5, 2.5}`, throughput remains effectively flat (`0.992x-1.000x`), while TPOT P99 drops by about `19%-20%` and E2E P99 by about `14%-17%`. Chunked prefill is therefore presented as a scheduling and tail-latency mechanism, not an unconditional throughput optimization.

### AWQ W4A16

The repository implements activation-aware Qwen3-8B calibration with layer-wise error propagation, quantizes the seven QKV/O/gate/up/down Linear projections, and exports a standard AutoAWQ GEMM checkpoint. On a fixed 2,048-token evaluation drawn from 32 WikiText-2 samples:

| Model | Perplexity | Peak evaluation memory | PPL vs BF16 |
| :--- | ---: | ---: | ---: |
| BF16 | 27.149 | 15.45 GiB | - |
| Official Qwen3-8B-AWQ | 29.273 | 6.52 GiB | +7.8% |
| LLM-Serve calibrated AWQ | 30.592 | 6.52 GiB | +12.7% |

With LLM-Serve's custom CUDA backend, runtime model memory falls from `15.276 GiB` to `5.857 GiB` and KV blocks increase from `152` to `419`. In the final RTX 3090 closed-loop capacity matrix, the maximum SLO-valid concurrency rises from `64` for BF16 to `128` for AWQ. AWQ's SLO-valid peak output throughput is still only `0.871x` of BF16, so this is presented as a capacity result rather than a custom-kernel speed win.

The same checkpoint completes 24/24 control points with vLLM 0.11 AWQ-Marlin. AWQ/BF16 output-throughput ratios at concurrency 1/4/8/16 are `1.390x/1.316x/1.322x/1.316x`. This validates checkpoint compatibility with a mature W4A16 backend; the speedup belongs to vLLM Marlin, not to LLM-Serve's custom CUDA kernel.

The existing serving line contains 72 sanitized run JSON files; Stage 8 adds 18 Graph comparison runs, and the dual-GPU PD comparison adds 18 reduced sanitized runs. Per-run CSVs, three-run aggregates, manifests, and AWQ quality/capacity/Marlin summaries are published under [`benchmarks/results/`](benchmarks/results/).

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

AWQ CUDA tests, real-checkpoint generation, and capacity matrices require an RTX 3090 or another SM80+ GPU. CUDA-specific Tree KV tests run only when CUDA is available. GitHub Actions runs the remaining tests with CPU PyTorch.

## Scope

- The primary target is single-GPU Qwen3-8B. The dual-GPU path is Prefill/Decode disaggregation; two non-NVLink GPUs are not presented as a tensor-parallel performance platform.
- Speculative CUDA Graph currently supports linear EAGLE, greedy acceptance, single-GPU TP=1, and explicit opt-in only; unsupported batch/context buckets fall back to eager. Fixed-tree speculation remains disabled by default.
- The AWQ runtime is limited to Qwen3, AutoAWQ GEMM, group-128 W4A16, BF16 activations/scales, eager execution, and TP=1. The Marlin run is an external-backend control experiment.
- The project intentionally omits an OpenAI-compatible HTTP layer; benchmarks drive the in-process runtime directly.
- The codebase retains the original MIT license.
