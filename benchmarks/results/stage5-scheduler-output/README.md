# Stage 5 SchedulerOutput Results

These results were generated from clean commit `f2d3843` after replacing the scheduler's negative token sentinel and `last_num_prefill_seqs` side channel with explicit `SchedulerOutput` groups.

- Hardware: one NVIDIA GeForce RTX 3090 24GB
- Runtime: Python 3.10.20, PyTorch 2.7.1+cu128, CUDA 12.8, BF16 eager
- Target revision: `b968826d9c46dd6066d109eabc6255188de91218`
- EAGLE3 draft revision: `08610ffa01dd9f16731fe8f627b85905b6aa51c4`
- Three runs per configuration with fixed seeds `0`, `1`, and `2`

| Workload | Baseline throughput | Comparison throughput | Ratio | Baseline TPOT P50/P99 | Comparison TPOT P50/P99 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Prefill injection | 91.33 +/- 1.68 tok/s | 84.49 +/- 2.13 tok/s (chunked) | 0.925x | 50.44/50.44 ms | 50.09/50.09 ms |
| Decode-heavy | 80.45 +/- 0.43 tok/s | 106.73 +/- 1.77 tok/s (EAGLE) | **1.327x** | 45.35/45.35 ms | 29.37/33.16 ms |
| Mixed closed-loop steady state | 179.73 +/- 4.65 tok/s | 198.64 +/- 8.68 tok/s (EAGLE) | **1.105x** | 44.46/44.54 ms | 41.63/51.86 ms |

`summary.csv` contains all 18 runs. `representative/` contains the fixed-seed-0 JSON for each of the six configurations, sanitized by `scripts/summarize_benchmarks.py`. Complete raw JSON remains local.

Speculative `burst_itl` measures per-token availability and therefore includes `0 ms` values when several tokens are emitted in one engine step. Use `output_event_latency` for adjacent output events and `speculative_step_latency` for full speculative-step cost.
