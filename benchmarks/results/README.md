# Stage 4 v1 Results

These results were generated from commit `89a1403` on one RTX 3090 24GB with BF16 eager execution.

- Target model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- EAGLE3 draft revision: `08610ffa01dd9f16731fe8f627b85905b6aa51c4`
- Three runs per configuration, fixed seeds `0`, `1`, and `2`
- Decode-heavy and mixed workloads use natural prompts; prefill-injection uses random-token prompts
- Baselines explicitly disable the speculative model

The public `summary.csv` contains all 18 runs. `representative/` contains one sanitized JSON for each published configuration. Complete raw JSON remains local until a release archive is prepared.

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
