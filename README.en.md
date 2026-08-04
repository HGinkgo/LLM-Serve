<div align="center">

# LLM-Serve

An educational LLM inference runtime for single- and dual-GPU serving

<p>
  English |
  <a href="README.md">简体中文</a>
</p>

<p>
  <a href="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml"><img src="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg" alt="CPU tests"></a>
  <img src="https://img.shields.io/badge/Model-Qwen3--8B-6f42c1" alt="Model: Qwen3-8B">
  <img src="https://img.shields.io/badge/Runtime-PyTorch%20%7C%20Triton%20%7C%20CUDA-76b900" alt="Runtime: PyTorch Triton CUDA">
  <img src="https://img.shields.io/badge/Serving-Paged%20KV%20%7C%20Continuous%20Batching-0ea5e9" alt="Serving: Paged KV and continuous batching">
  <img src="https://img.shields.io/badge/Advanced-PD%20%7C%20EAGLE%20%7C%20AWQ-f59e0b" alt="Advanced: PD EAGLE AWQ">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</p>

</div>

LLM-Serve is an educational inference runtime centered on Qwen3-8B. It starts from a single-GPU engine and develops the core mechanisms behind LLM serving: paged KV cache management, continuous batching, chunked prefill, EAGLE-style speculative decoding, AWQ W4A16, and dual-GPU Prefill/Decode disaggregation.

The early project was informed by the [PagedAttention paper](https://arxiv.org/abs/2309.06180) and the [`nano-vllm`](https://github.com/Geeeone/nano-vllm) teaching skeleton. The scheduler, serving benchmark system, speculative runtime, quantization calibration path, and PD serving path have been developed independently in this repository.

## Capabilities

- **Runtime**: paged KV cache, block tables, prefix cache, iteration-level continuous batching, and an explicit `SchedulerOutput` contract.
- **Scheduling**: decode-first chunked prefill for mixed prefill/decode batches and long-prompt isolation.
- **Speculative decoding**: EAGLE-style batched draft proposal, packed target verification, per-request draft KV, greedy accept/reject, and target-verify CUDA Graphs.
- **Quantization**: Qwen3 activation-aware AWQ W4A16 calibration, standard AutoAWQ GEMM checkpoint export, reference/Triton/CUDA Linear backends, and KV capacity admission.
- **Dual-GPU serving**: independent Prefill/Decode workers, logical KV handoff, pinned shared-memory slots, ACK/backpressure, and Decode CUDA Graphs.
- **Reproducible experiments**: Poisson request-rate and closed-loop concurrency runners with throughput, goodput, TTFT, TPOT, E2E, queue, and stage-timing metrics.

## Repository Layout

```text
llmserve/
├── engine/        scheduling, KV blocks, target execution, speculative orchestration
├── models/        Qwen3 and EAGLE3 model definitions
├── speculative/   draft, verification, fixed trees, and Tree KV management
├── quantization/  AWQ calibration, checkpoint export, and quality evaluation
├── pd/            Prefill/Decode protocols, KV handoff, slots, and workers
└── layers/        attention, linear, sampling, and model building blocks
benchmarks/        workloads, arrivals, metrics, suite runners, and public data
tests/             CPU, checkpoint integration, and CUDA tests
```

## Quick Start

```bash
pip install -e .

export MODEL_PATH=/path/to/Qwen3-8B
python example.py
```

Run the CPU regression suite:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests
```

Run the GPU smoke suite:

```bash
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python -m benchmarks.run_suite \
  --suite benchmarks/suites/smoke.json \
  --output-dir /tmp/llmserve-smoke \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --allow-dirty
```

## Experiments and Documentation

- [Benchmark guide](benchmarks/README.md): suites, workloads, metric semantics, and reproduction commands.
- [Public benchmark data](benchmarks/results/): manifests, CSV files, sanitized run JSON, and stage reports.
- [PD end-to-end results](benchmarks/results/pd-serving-formal/): collocated versus dual-GPU Prefill/Decode serving.
- [AWQ results](benchmarks/results/awq-w4a16/): quality, capacity, and vLLM Marlin control experiments.
- [Verification evidence](benchmarks/results/verification.md): CPU regression, CUDA smoke, and public-data checks.

Benchmark numbers are intentionally not duplicated in this README. Use the corresponding directory under `benchmarks/results/` as the source of truth for detailed measurements and sanitized raw data.

## Scope and Limitations

- The primary target is Qwen3-8B, single-GPU TP=1, and an RTX 3090 24GB. The dual-GPU path is Prefill/Decode disaggregation, not a Tensor Parallel performance platform for non-NVLink GPUs.
- Speculative CUDA Graphs support linear EAGLE, greedy acceptance, and explicit opt-in only. Unsupported shapes fall back to eager; fixed-tree speculation is disabled by default.
- The AWQ runtime is limited to Qwen3, AutoAWQ GEMM, group-128 W4A16, BF16 activations/scales, and TP=1. vLLM Marlin measurements are external-backend control experiments.
- The project focuses on runtime, scheduling, and serving mechanisms and intentionally does not provide an OpenAI-compatible HTTP API layer.

## License

[MIT](LICENSE)
