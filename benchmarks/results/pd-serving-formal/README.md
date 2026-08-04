# Dual-GPU PD Serving Formal Matrix

This directory publishes the formal end-to-end comparison between the single-process collocated runtime and the dual-GPU Prefill/Decode (PD) serving path.

- Model: Qwen3-8B BF16, no speculative decoding.
- Hardware: two RTX 3090 GPUs, CUDA 12.8; each worker owns one GPU.
- PD configuration: Prefill batch 4, eager Prefill, Decode CUDA Graph, two reusable shared-memory KV slots.
- Workload: `128 input / 64 output`, three runs per point.
- Matrix: closed-loop concurrency `{16, 32, 48, 64}` and Poisson request rates `{8, 16, 24, 28}`.
- Source commit: `ae740739d25f7184c04208f35dcf1b720e624f2a`.

The collocated variant is the single-process BF16 Decode Graph baseline. The PD variant separates Prefill and Decode across the two GPUs and uses descriptor-only KV handoff through the shared-slot transport.

## Closed-Loop

| Concurrency | Collocated req/s | PD req/s | PD / baseline | TTFT P50 (baseline -> PD) | TPOT P50 (baseline -> PD) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 16 | 8.09 | **10.13** | **1.252x** | 379 -> 140 ms | 25.0 -> 22.9 ms |
| 32 | 12.27 | **18.27** | **1.489x** | 631 -> 132 ms | 31.0 -> 25.4 ms |
| 48 | 13.60 | **21.69** | **1.595x** | 854 -> 133 ms | 41.6 -> 33.4 ms |
| 64 | 16.00 | **28.87** | **1.804x** | 1100 -> 136 ms | 45.4 -> 33.0 ms |

The gain grows with sustained concurrency: PD reaches `+80.4%` request throughput at concurrency 64, while its TTFT remains nearly flat. The collocated baseline starts accumulating a waiting queue at concurrency 32; PD has no material waiting queue at concurrency 64.

## Poisson Arrival

| Request rate | Collocated req/s | PD req/s | PD / baseline | TTFT P50 (baseline -> PD) | TPOT P50 (baseline -> PD) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 8 | 6.47 | 6.46 | 0.999x | 65 -> 85 ms | 33.4 -> 22.2 ms |
| 16 | 11.21 | **11.73** | **1.047x** | 78 -> 121 ms | 63.3 -> 23.9 ms |
| 24 | 13.93 | **15.72** | **1.129x** | 104 -> 176 ms | 68.1 -> 28.7 ms |
| 28 | 15.02 | **17.25** | **1.148x** | 134 -> 254 ms | 67.7 -> 29.7 ms |

At low open-loop load the systems are both below saturation, so the throughput gap is small. Once Prefill/decode overlap is exercised, PD keeps TPOT around `22-30 ms` while the collocated path rises to `63-68 ms`.

## Attribution and Boundaries

The result is a serving architecture result, not a faster Qwen3 forward pass. Prefill model forward remains about `121-123 ms` and KV export/copy about `3.8 ms` in both paths. The PD handoff path is only `3-5%` of the Prefill roundtrip; the current ceiling is the batch-4 Prefill pipeline at about `29.6 req/s` (`4 / 135 ms`). The improvement comes from overlapping roles and giving Decode an independent GPU and continuous-batching loop.

All 48 points completed successfully. There were no OOMs, worker failures, CUDA Graph capture failures, or request failures. PD shared slots ended with `free_slots=2`, `pending_transfers=0`, and no inline fallback. The raw sanitized run JSON, `aggregate.csv`, `summary.csv`, and `manifest.json` are kept in this directory; the suite can be reconstructed from [`benchmarks/suites/pd-serving-formal.json`](../../suites/pd-serving-formal.json).
