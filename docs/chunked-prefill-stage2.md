# Chunked Prefill Stage 2

Date: 2026-06-07

## Goal

Add an experimental chunked prefill scheduler path for online serving workloads.

The target behavior is decode-first scheduling: existing running requests receive one decode token first, then the remaining token budget is used for prefill chunks. This is intended to reduce decode stalls when new prompt prefill work arrives during serving.

## Scope

Implemented:

- `Config.enable_chunked_prefill`, default `False`.
- `Sequence.num_computed_tokens`, `Sequence.is_prefilling`, and `Sequence.num_uncomputed_tokens` compatibility properties.
- `Scheduler.schedule()` dispatches to an experimental chunked prefill path when enabled.
- Chunked path schedules running decode first, then waiting prefill chunks with the remaining token budget.
- `ModelRunner.prepare_prefill()` can build a mixed batch containing prefill chunks and one-token decode work.
- Mixed batches run as `is_prefill=True`, so they use `flash_attn_varlen_func` and do not use decode CUDA graph replay.
- `bench_serving.py` accepts `--enable-chunked-prefill`.

Not changed:

- `block_manager.py`
- attention kernels
- sampler
- `bench.py`

## Implementation Notes

The implementation is intentionally smaller than `nano-vllm-plus`:

- The public `Scheduler.schedule()` return shape is preserved as `(seqs, is_prefill)`.
- Mixed batches are represented as prefill sequences followed by decode sequences.
- `Scheduler.last_num_prefill_seqs` records the split point for `LLMEngine.step()`.
- In mixed mode, decode sequences are marked with `num_scheduled_tokens = -1` before entering `ModelRunner.prepare_prefill()`.
- `LLMEngine.step()` postprocesses mixed batches as two slices: prefill first, decode second.

Known tradeoff:

- Partial prefill chunks still produce sampled logits that are ignored by scheduler postprocess. This keeps the change small and functional, but adds overhead versus a fuller implementation that samples only completed-prefill and decode sequences.

## Validation

Syntax check:

```bash
conda run -n nano-vllm python -m py_compile \
  nanovllm/config.py \
  nanovllm/engine/sequence.py \
  nanovllm/engine/scheduler.py \
  nanovllm/engine/model_runner.py \
  nanovllm/engine/llm_engine.py \
  bench_serving.py
```

Non-chunked smoke:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --num-requests 2 \
  --input-len 16 \
  --output-len 4 \
  --arrival all \
  --enforce-eager \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --output-json /tmp/nano_vllm_no_chunk_smoke_stage2_final.json
```

Result:

- finished/failed: 2 / 0
- output tokens: 8
- throughput: 5.40 tok/s

Chunked smoke:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --num-requests 4 \
  --input-len 128 \
  --output-len 8 \
  --arrival poisson \
  --request-rate 16 \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-model-len 512 \
  --max-num-batched-tokens 64 \
  --output-json /tmp/nano_vllm_chunk_smoke_stage2_final.json
```

Result:

- finished/failed: 4 / 0
- output tokens: 32
- throughput: 9.82 tok/s

## Poisson A/B

Environment:

- GPU: one RTX 3090 via `CUDA_VISIBLE_DEVICES=0`
- Model: Qwen3-0.6B
- Run date: 2026-06-07
- Conda env: `nano-vllm`
- `enforce_eager=False`
- `max_num_batched_tokens=16384`

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --num-requests 32 \
  --input-len 256 \
  --output-len 256 \
  --arrival poisson \
  --request-rate 4 \
  --output-json experiment-data/2026-06-07_chunked-prefill/poisson_32x256x256_no_chunk_final.json
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python bench_serving.py \
  --num-requests 32 \
  --input-len 256 \
  --output-len 256 \
  --arrival poisson \
  --request-rate 4 \
  --enable-chunked-prefill \
  --output-json experiment-data/2026-06-07_chunked-prefill/poisson_32x256x256_chunked_final.json
```

Times are milliseconds except wall time. Throughput is output tokens per serving wall time.

| Mode | Finished/Failed | Throughput | Wall Time | TTFT mean/P99/max | ITL mean/P99/max | TPOT mean/P99 | Latency mean/P99/max |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| chunked off | 32 / 0 | 768.44 tok/s | 10.661 s | 96.22 / 829.14 / 1112.55 | 3.85 / 39.25 / 279.94 | 3.85 / 4.96 | 1078.03 / 2093.67 / 2438.57 |
| chunked on | 32 / 0 | 768.50 tok/s | 10.660 s | 95.31 / 815.16 / 1093.01 | 3.79 / 36.85 / 196.73 | 3.79 / 4.83 | 1062.43 / 2045.91 / 2383.59 |

## Observations

- Both modes completed all 32 requests.
- Throughput was effectively unchanged on this workload.
- ITL P99 improved slightly: `39.25 ms` to `36.85 ms`.
- ITL max improved: `279.94 ms` to `196.73 ms`.
- TTFT P99 improved slightly: `829.14 ms` to `815.16 ms`.
- The `32 x 256 x 256` Poisson workload with default `max_num_batched_tokens=16384` is not a strong stress test for long-prompt blocking. It should be treated as a correctness and light A/B check, not as proof of large chunked prefill benefit.

## Follow-Up

- Add a stronger stress workload with longer prompts or smaller `max_num_batched_tokens`.
- Consider exposing chunk size / token budget explicitly in the benchmark matrix.
- Optimize sampling so partial prefill chunks do not produce unused sampled tokens.
- If the scheduler interface is allowed to change later, consider the clearer `prefill_seqs, chunk_sizes, decode_seqs` shape used by `nano-vllm-plus`.
