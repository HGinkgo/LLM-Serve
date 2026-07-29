# Stage 8 Target Verify CUDA Graph

This directory publishes the formal Stage 8 comparison for linear EAGLE target verification on one RTX 3090.

- Workload: closed-loop, `128 input / 128 output`, `gamma=3`, greedy acceptance.
- Variants: eager target verification versus target-verify CUDA Graph.
- Both variants use `enforce_eager=true`; ordinary decode CUDA Graph is excluded.
- Matrix: concurrency `{1, 4, 8}`, three runs per variant and concurrency, 18 points total.
- Source commit: `3bb5d21ad5fd9ae0044943d93255a4542cc5ca75`.

Graph captures six fixed shapes: batch buckets `{1, 4, 8}` and context frontiers `{256, 1024}`. Unsupported shapes and block-table reservation overflow fall back to eager. All 18 points completed successfully. The aggregate output throughput gains over eager were `1.779x`, `1.558x`, and `1.419x` at concurrency 1, 4, and 8.

The `runs/` JSON files are sanitized benchmark outputs. They contain no absolute model paths, prompt token IDs, or absolute request timestamps.
