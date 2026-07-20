#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the Qwen3-8B checkpoint directory}"
: "${SPECULATIVE_MODEL:?set SPECULATIVE_MODEL to the EAGLE3 checkpoint directory}"

RUNS="${RUNS:-3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment-data/stage4-v1}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUTPUT_DIR"

common=(
  --model "$MODEL_PATH"
  --temperature 0.01
  --enforce-eager
  --argmax-sampler
)

baseline=(
  --speculative-model ""
)

eagle=(
  --speculative-model "$SPECULATIVE_MODEL"
  --speculative-gamma 3
  --speculative-accept-mode greedy
)

for ((run = 1; run <= RUNS; run++)); do
  seed=$((run - 1))

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" bench_serving.py \
    "${common[@]}" \
    "${baseline[@]}" \
    --workload-name prefill-injection-baseline \
    --arrival prefill-injection \
    --num-requests 5 \
    --input-len 128 \
    --long-input-len 4096 \
    --injection-delay 1 \
    --output-len 256 \
    --max-model-len 4608 \
    --max-num-batched-tokens 512 \
    --seed "$seed" \
    --output-json "$OUTPUT_DIR/prefill_baseline_r${run}.json"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" bench_serving.py \
    "${common[@]}" \
    "${baseline[@]}" \
    --workload-name prefill-injection-chunked \
    --arrival prefill-injection \
    --num-requests 5 \
    --input-len 128 \
    --long-input-len 4096 \
    --injection-delay 1 \
    --output-len 256 \
    --max-model-len 4608 \
    --max-num-batched-tokens 512 \
    --enable-chunked-prefill \
    --seed "$seed" \
    --output-json "$OUTPUT_DIR/prefill_chunked_r${run}.json"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" bench_serving.py \
    "${common[@]}" \
    "${baseline[@]}" \
    --workload-name decode-heavy-baseline \
    --arrival all \
    --num-requests 4 \
    --prompt-mode natural \
    --output-len 256 \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --seed "$seed" \
    --output-json "$OUTPUT_DIR/decode_baseline_r${run}.json"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" bench_serving.py \
    "${common[@]}" \
    "${eagle[@]}" \
    --workload-name decode-heavy-eagle \
    --arrival all \
    --num-requests 4 \
    --prompt-mode natural \
    --output-len 256 \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --seed "$seed" \
    --output-json "$OUTPUT_DIR/decode_eagle_r${run}.json"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" bench_serving.py \
    "${common[@]}" \
    "${baseline[@]}" \
    --workload-name mixed-closed-loop-baseline \
    --arrival closed-loop \
    --max-concurrency 8 \
    --warmup-seconds 5 \
    --measurement-seconds 15 \
    --prompt-mode natural \
    --output-len 64 \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --seed "$seed" \
    --output-json "$OUTPUT_DIR/mixed_baseline_r${run}.json"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" bench_serving.py \
    "${common[@]}" \
    "${eagle[@]}" \
    --workload-name mixed-closed-loop-eagle \
    --arrival closed-loop \
    --max-concurrency 8 \
    --warmup-seconds 5 \
    --measurement-seconds 15 \
    --prompt-mode natural \
    --output-len 64 \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --seed "$seed" \
    --output-json "$OUTPUT_DIR/mixed_eagle_r${run}.json"
done

"$PYTHON" scripts/summarize_benchmarks.py \
  "$OUTPUT_DIR"/*.json \
  --csv "$OUTPUT_DIR/summary.csv"
