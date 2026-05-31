# nano-vllm-runtime

`nano-vllm-runtime` is a learning-oriented LLM inference runtime project based on `nano-vllm`.

The current focus is single-GPU serving observability and scheduler-oriented benchmarking. It is not a production vLLM replacement.

## Current Status

- Baseline runtime imported from `nano-vllm`.
- Request-level Serving Benchmark v1 implemented.
- `LLMEngine` records request arrival, first-token, decode-token, and finished timestamps.
- `bench_serving.py` reports throughput, TTFT, ITL/TPOT, request latency, wall time, and success/failure.
- RTX 3090 single-GPU baseline results are documented in `docs/benchmark-v1-baseline.md`.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Set a local model path:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
```

Run the example:

```bash
python example.py
```

Run a small serving benchmark:

```bash
python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 8 \
  --input-len 128 \
  --output-len 128 \
  --arrival all
```

## Project Direction

The next runtime work is planned around scheduler behavior, especially chunked prefill A/B experiments using the Serving Benchmark v1 metrics.

Out of scope for the current mainline:

- speculative decoding
- multi-node distributed serving
- KV cache eviction policy
- CUDA kernel optimization
- tensor parallel as the primary direction

## Notes

- Raw local experiment artifacts are kept under `experiment-data/` and are not committed.
- Agent/project memory is kept under `.agent/` and is not committed.
- This project keeps the original `nano-vllm` license.
