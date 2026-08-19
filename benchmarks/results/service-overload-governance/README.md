# Service Overload Governance

验证日期：2026-08-19。该结果使用 Qwen3-8B、单张 RTX 3090、CUDA 12.8，直接驱动
`EngineServiceRuntime`，不包含 HTTP 网络开销。`unbounded` 与 `limit-64` 使用同一份
固定 Poisson trace、请求内容、seed 和 arrival schedule；trace seed 为 `20260818`，
trace hash 为 `ad9598cb2100149a164484be4ec4dfed9c493c063fec4c06fc59e14c298ba972`。

参数为 request rate 24 req/s、input 128、output 64、warmup 10 s、measurement 20 s、
drain 15 s、`max_model_len=512`、`max_num_batched_tokens=1024`、`max_num_seqs=64`、
`gpu_memory_utilization=0.8`。完整脱敏汇总见 [`summary.csv`](summary.csv)，实验元数据见
[`metadata.json`](metadata.json)；带本地绝对路径的 raw JSON 保留在未纳入公开证据的
`experiment-data/2026-08-19_service-overload-streaming-metrics-gpu1/`。

## Summary

| variant | accepted / rejected | completed in window / cohort | unfinished after drain | request/s | output tok/s | TTFT P50/P99 ms | TPOT P50/P99 ms | waiting P99 / max | inflight P99 / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unbounded | 487 / 0 | 290 / 341 | 146 | 14.5 | 975.55 | 10391.72 / 16861.23 | 67.21 / 72.87 | 288.88 / 292 | 352.88 / 356 |
| limit-64 | 271 / 216 | 268 / 271 | 0 | 13.4 | 821.15 | 184.85 / 372.80 | 70.85 / 76.30 | 0 / 0 | 64 / 64 |

Timed-out requests were `0` in both measurement windows. The service benchmark intentionally
does not export E2E or goodput: TTFT is submit-boundary to the first token event, and TPOT is
computed from actual token-event intervals (with a single-token request represented as no TPOT
sample). `completed in window` is the measurement-window outcome; `completed cohort` is the
latency cohort used for TTFT/TPOT, so the two counts are not interchangeable.

At this arrival rate, the 64-request admission bound reduced TTFT P99 from 16861.23 ms to
372.80 ms and removed the unbounded waiting buildup, at the cost of 216/487 measurement-window
requests rejected, 7.6% lower request throughput, and 15.8% lower output throughput. TPOT P99
increased from 72.87 ms to 76.30 ms.
