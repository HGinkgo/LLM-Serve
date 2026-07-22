# Benchmark Results

This directory contains the public, sanitized evidence for LLM-Serve benchmark claims.

## Data Boundary

- `results/summary.csv` and `results/representative/`: Stage 4 pre-refactor dataset.
- `results/stage5-scheduler-output/`: Stage 5 post-refactor summary and sanitized representative JSON.
- Full raw runs remain local unless they are explicitly curated for publication.
- Public files must not contain absolute model paths, prompt paths, credentials, or host-specific workspace paths.

Historical JSON generated before Stage 4 does not include output-event or speculative-step latency. The summary script preserves its old `itl` field as `burst_itl` and leaves the newer metrics empty. Historical data is not used as a substitute for the clean Stage 4 rerun.

## Metric Semantics

- `burst_itl`: adjacent per-token availability timestamps; speculative bursts may contain `0 ms` samples.
- `output_event_latency`: adjacent engine output events, recorded once per request per emitting step.
- `speculative_step_latency`: full draft/verify/accept/KV step time.
- `tpot`: request-level average time per output token.

Throughput, TPOT, and request latency can be compared directly between baseline and speculative runs. Burst ITL must retain its burst-emission label.

## Generate Public Artifacts

Run benchmarks with `--output-json`, `--workload-name`, and explicit model revision metadata. Then generate the public CSV and sanitized representative JSON:

```bash
python scripts/summarize_benchmarks.py \
  /path/to/run1.json /path/to/run2.json \
  --csv benchmarks/results/<dataset>/summary.csv \
  --representative-dir benchmarks/results/<dataset>/representative
```

The script copies measured values without modification. It only normalizes local path fields in the representative JSON.
