# Benchmark Results

This directory preserves two comparable Qwen3-8B benchmark datasets generated on one RTX 3090 24GB with BF16 eager execution:

- `summary.csv` and `representative/`: Stage 4 results from commit `89a1403`, before the SchedulerOutput refactor.
- `stage5-scheduler-output/`: Stage 5 results from commit `f2d3843`, after the SchedulerOutput refactor.

Both datasets use the same model revisions, workloads, seeds, and metric definitions. Each summary contains all 18 runs; each representative directory contains one sanitized JSON per configuration.

## Stage 5 SchedulerOutput Rerun

| Workload | Baseline mean throughput | Comparison mean throughput | Ratio | Baseline output-event P99 | Comparison output-event P99 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Prefill injection | 91.33 +/- 1.68 tok/s | 84.49 +/- 2.13 tok/s (chunked) | 0.925x | 46.74 ms | 135.55 ms |
| Decode-heavy | 80.45 +/- 0.43 tok/s | 106.73 +/- 1.77 tok/s (EAGLE) | **1.327x** | 47.87 ms | 61.51 ms |
| Mixed closed-loop steady state | 179.73 +/- 4.65 tok/s | 198.64 +/- 8.68 tok/s (EAGLE) | **1.105x** | 46.52 ms | 166.47 ms |

The values are mean +/- sample standard deviation across three runs. Relative to Stage 4, the six configuration means changed by `-2.04%` to `+4.28%`; there is no systematic performance-regression signal from replacing hidden scheduler state with `SchedulerOutput`. Chunked prefill still does not establish a throughput win in this workload.

All 18 Stage 5 JSON files record `git_dirty=false`, the same target/draft revisions, zero failures, and 401 successful request records. EAGLE burst ITL retains its `0 ms` samples; output-event latency and speculative-step latency remain separate metrics.

## Stage 4 Baseline

These results were generated from commit `89a1403`.

- Target model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- EAGLE3 draft revision: `08610ffa01dd9f16731fe8f627b85905b6aa51c4`
- Three runs per configuration, fixed seeds `0`, `1`, and `2`
- Decode-heavy and mixed workloads use natural prompts; prefill-injection uses random-token prompts
- Baselines explicitly disable the speculative model

The public `summary.csv` contains all 18 runs. `representative/` contains one sanitized JSON for each published configuration. Complete raw JSON remains local.

## Summary

| Workload | Baseline mean throughput | Comparison mean throughput | Ratio | Baseline output-event P99 | Comparison output-event P99 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Prefill injection | 89.25 tok/s | 85.52 tok/s (chunked) | 0.958x | 48.03 ms | 136.23 ms |
| Decode-heavy | 82.12 tok/s | 102.80 tok/s (EAGLE) | **1.252x** | 47.28 ms | 66.99 ms |
| Mixed closed-loop steady state | 177.60 tok/s | 190.49 tok/s (EAGLE) | **1.073x** | 47.27 ms | 176.00 ms |

The prefill-injection result does not establish a throughput win for chunked prefill. Its value must be evaluated through the specific long-prompt interruption and tail-latency behavior, not throughput alone. EAGLE burst ITL retains its `0 ms` samples; output-event latency and speculative-step latency are reported separately in the CSV.

## Verification

- GitHub CPU test run: [29747762901](https://github.com/HGinkgo/LLM-Serve/actions/runs/29747762901)
- Local Qwen3-8B + EAGLE3 + CUDA test suite: `104/104` passed
- Every published JSON records `git_dirty=false` and the same benchmark commit.
