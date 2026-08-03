# Dual-GPU PD KV Pipeline

This directory publishes the formal Qwen3-8B BF16 comparison of two Prefill-to-Decode KV transports on two RTX 3090 GPUs.

- Workload: closed-loop, `128 input / 64 output`, concurrency `{32, 48, 64}`.
- Runtime: Prefill batch 4, eager Prefill, Decode CUDA Graph, three runs per point.
- Baseline: inline CPU tensors sent through a multiprocessing Queue.
- Optimized path: two reusable CUDA-registered shared-memory slots, descriptor-only handoff, generation validation, ACK-based reclamation, and inline fallback for oversized batches.
- Source commit: `e779a6aa4186683327060697b8c04f3da12c0284`.

| Concurrency | Inline req/s | Shared req/s | Ratio | Inline TTFT P50 | Shared TTFT P50 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 32 | 16.29 | 17.89 | 1.098x | 207 ms | 132 ms |
| 48 | 18.62 | 20.71 | 1.112x | 227 ms | 138 ms |
| 64 | 21.31 | 28.11 | 1.319x | 639 ms | 136 ms |

At concurrency 64, response-queue time falls from `47.4 ms` to `0.25 ms`, Decode admission from `14.5 ms` to `1.13 ms`, and total Prefill roundtrip from `186.6 ms` to `134.2 ms`. Model forward and KV export/copy remain unchanged. The result isolates an IPC and pipeline improvement rather than a model-compute speedup.

All 18 points completed successfully. Every shared run finished with two free slots and zero pending transfers; no shared batch fell back to inline transport. The reduced run JSON files omit request-level records and repetitive per-batch timing details while retaining aggregate metrics, queue/slot samples, CUDA Graph counters, and worker health.
