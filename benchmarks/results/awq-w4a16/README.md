# AWQ W4A16 Evidence

This directory publishes sanitized aggregates for the Qwen3-8B AWQ work. Model checkpoints, calibration text, layer caches, absolute paths, and raw logs remain local.

## Scope

- `quality.csv`: BF16, the official Qwen3-8B-AWQ checkpoint, and the LLM-Serve-calibrated checkpoint on the same fixed 2,048-token evaluation.
- `capacity.csv`: LLM-Serve closed-loop capacity comparison, three runs per concurrency, 45-second warmup and 120-second measurement.
- `marlin-control.csv`: the LLM-Serve-calibrated checkpoint deployed through vLLM 0.11 AWQ-Marlin, three runs per concurrency.
- `metadata.json`: workload, SLO, environment, checkpoint, and result provenance.
- `representative/`: compact BF16/AWQ `C=128, run=1` JSON records generated from the suite output; request-level records are omitted.

The custom LLM-Serve CUDA backend reduces runtime model memory from `15.276 GiB` to `5.857 GiB` and increases KV blocks from `152` to `419`. It doubles the maximum SLO-valid concurrency from `64` to `128`, but its SLO-valid peak output throughput is `0.871x` of BF16.

The Marlin control reaches `1.316x-1.390x` AWQ/BF16 throughput. This validates the exported checkpoint against a mature W4A16 backend; it is not a performance result for LLM-Serve's custom CUDA kernel.

SLO-valid means `TTFT P99 <= 10,000 ms`, `TPOT P99 <= 250 ms`, and `E2E P99 <= 120,000 ms`. All 24 LLM-Serve capacity points and all 24 Marlin control points completed successfully.
