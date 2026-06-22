# Serving Benchmark v1 Baseline

Date: 2026-05-31

## Goal

Add online serving observability to `ThrustLM` without changing scheduler, model runner, attention, or kernel logic.

## Scope

Implemented:

- `LLMEngine.add_request()` records arrival time and returns `seq_id`.
- `LLMEngine.step()` records first-token, decode-token, and finished timestamps.
- `LLMEngine.get_metrics()` returns request-level TTFT, ITL, TPOT, request latency, wall time, throughput, and success/failure counts.
- `bench_serving.py` supports all-at-once and Poisson request arrivals.

Not changed:

- `scheduler.py`
- `model_runner.py`
- attention / kernel files
- `bench.py`

## Metric Definitions

- TTFT: first generated token time minus request arrival time.
- ITL: interval between consecutive generated token timestamps for the same request.
- TPOT: finished time minus first token time, divided by generated tokens after the first token.
- Request latency: finished time minus request arrival time.
- Throughput: total generated output tokens divided by serving wall time.
- Success: request reached finished state.
- Failure: request did not finish during the run or was interrupted.

## Commands

Model path:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
```

All-at-once arrival:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 32 \
  --input-len 256 \
  --output-len 256 \
  --arrival all \
  --output-json /path/to/local-results/all_32x256x256_graph.json
```

Poisson arrival:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 32 \
  --input-len 256 \
  --output-len 256 \
  --arrival poisson \
  --request-rate 4 \
  --output-json /path/to/local-results/poisson_32x256x256_graph.json
```

Eager mode:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 32 \
  --input-len 256 \
  --output-len 256 \
  --arrival all \
  --enforce-eager \
  --output-json /path/to/local-results/all_32x256x256_eager.json
```

Long input short output:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 16 \
  --input-len 1024 \
  --output-len 128 \
  --arrival all \
  --output-json /path/to/local-results/all_16x1024x128_graph.json
```

Short input long output:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 16 \
  --input-len 128 \
  --output-len 1024 \
  --arrival all \
  --output-json /path/to/local-results/all_16x128x1024_graph.json
```

## Run Metadata

- Run date: 2026-05-31
- GPU: NVIDIA GeForce RTX 3090, GPU 0, 24576 MiB
- Driver: 580.95.05
- OS: Ubuntu 22.04.5 LTS
- glibc: 2.35
- Python: 3.10.20
- PyTorch: 2.7.1+cu128
- torch CUDA: 12.8
- transformers: 4.57.6
- triton: 3.3.1
- flash-attn: 2.8.3
- Model: Qwen3-0.6B
- Model path: local Qwen3-0.6B path supplied with `MODEL_PATH`
- Git commit: `27ba4a6d12deb518ce049e86d62df4ad8d6f3bbc`
- Worktree status during run: dirty. Relevant Serving Benchmark v1 changes were uncommitted.
- Raw result archive: kept locally under `experiment-data/` and not committed

## 3090 Baseline Results

All runs below use one RTX 3090 via `CUDA_VISIBLE_DEVICES=0`.

Times are reported in milliseconds except wall time. Throughput is output tokens per serving wall time.

| Case | Arrival | Requests | Input Len | Output Len | enforce_eager | Throughput | TTFT mean/P99 | ITL mean/P99 | Latency mean/P99 | Success/Failure |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| eager vs cuda graph: graph | all | 32 | 256 | 256 | False | 2980.61 tok/s | 1193.55 / 1193.64 | 6.10 / 6.57 | 2748.34 / 2748.43 | 32 / 0 |
| eager vs cuda graph: eager | all | 32 | 256 | 256 | True | 829.18 tok/s | 1316.94 / 1317.03 | 33.58 / 36.13 | 9879.54 / 9879.62 | 32 / 0 |
| long input short output | all | 16 | 1024 | 128 | False | 1023.47 tok/s | 1248.29 / 1248.37 | 5.93 / 6.12 | 2000.96 / 2001.04 | 16 / 0 |
| short input long output | all | 16 | 128 | 1024 | False | 2696.19 tok/s | 1166.00 / 1166.04 | 4.80 / 6.02 | 6076.68 / 6076.72 | 16 / 0 |
| poisson arrival | poisson | 32 | 256 | 256 | False | 768.44 tok/s | 103.12 / 859.11 | 3.82 / 18.72 | 1076.19 / 2112.90 | 32 / 0 |

## Observations

- CUDA graph mode is much faster than eager mode for the `32 x 256 x 256` all-at-once workload: `2980.61 tok/s` vs `829.18 tok/s`.
- Eager mode has much worse ITL in this workload: mean ITL `33.58 ms` vs graph mean ITL `6.10 ms`.
- Long input / short output is prefill-heavy and has lower output-token throughput than short input / long output: `1023.47 tok/s` vs `2696.19 tok/s`.
- Poisson arrival throughput is lower because wall time includes request inter-arrival time. It should not be interpreted as saturated offline throughput.
- All five baseline runs completed without failed requests.

## Notes

- `bench_serving.py` uses synthetic token ids, matching the current `bench.py` benchmark style.
- `ignore_eos=True` is used so configured `output_len` controls generated length.
- The script introduces no new Python dependencies.
