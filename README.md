# LLM-Serve

[English](README.en.md) | [简体中文](README.md)

[![CPU tests](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml)

LLM-Serve 是一个面向单机单卡场景的 LLM 推理引擎项目，重点关注高吞吐 serving 背后的核心系统机制：Paged KV Cache、continuous batching、chunked prefill、serving benchmark，以及 EAGLE 风格投机解码。

项目早期骨架参考了 vLLM PagedAttention 论文和 `nano-vllm` 教学实现。后续的 scheduler 改造、chunked prefill、benchmark 工具和 speculative decoding runtime 都是在本仓库中独立设计和实现的。

仓库名和 Python 包名分别为 `LLM-Serve` / `llmserve`。

## 功能

- PagedAttention 风格的 KV cache 管理：block table、可复用 KV block、prefix-cache-aware block 分配。
- Continuous batching：以 iteration-level scheduler 组织 prefill 和 decode。
- Chunked prefill：decode 优先，把剩余 token budget 分给长 prompt 的 prefill chunk，避免长 prompt 长时间阻塞 decode。
- Serving benchmark：记录 throughput、TTFT、burst ITL、output-event latency、speculative step latency、TPOT 和 request latency，并提供区分 warmup、measurement 和 drain 的 closed-loop 稳态测量模式。
- EAGLE 风格投机解码：支持 batched draft proposal、packed target verify、per-request draft KV、greedy verify，以及 acceptance 和 timing 指标。
- Qwen3 模型支持，用于本地单卡实验。

## 代码结构

- `llmserve/engine/`：请求调度、KV block 管理、target 模型执行和投机解码流程编排。
- `llmserve/models/`：Qwen3 与 EAGLE3 网络定义及 checkpoint 加载。
- `llmserve/speculative/`：draft 生成、验证采样、固定候选树算法、Tree KV 管理和共享结果类型。
- `llmserve/layers/`：attention、sampling 和模型基础组件。
- `bench_serving.py`：有限 workload 与 closed-loop serving benchmark。
- `tests/`：CPU 单元测试、可选真实 checkpoint 集成测试和 CUDA kernel 测试。
- `scripts/summarize_benchmarks.py`：生成脱敏的代表性 JSON 与逐运行 CSV。
- `benchmarks/`：公开 benchmark 口径和结果。

`ModelRunner` 持有 target 模型执行资源，`SpeculativeExecutor` 组合这些资源完成 EAGLE decode；与 runtime 无关的算法集中在 `llmserve/speculative/`。

## 快速开始

以 editable 模式安装：

```bash
pip install -e .
```

设置本地模型路径：

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

运行一个简单生成示例：

```bash
python example.py
```

运行一个小型 serving benchmark：

```bash
python bench_serving.py \
  --model "$MODEL_PATH" \
  --num-requests 8 \
  --input-len 128 \
  --output-len 128 \
  --arrival all
```

以固定并发测量 continuous batching 的稳态性能：

```bash
python bench_serving.py \
  --model "$MODEL_PATH" \
  --arrival closed-loop \
  --max-concurrency 8 \
  --warmup-seconds 5 \
  --measurement-seconds 15 \
  --prompt-mode natural \
  --output-len 64
```

closed-loop 模式会在请求完成后立即补位，直到 measurement window 结束才停止接收新请求并 drain。稳态吞吐只统计 measurement window 内产生的 token，同时仍保留整体 summary 供对照。

运行 chunked prefill 路径：

```bash
python bench_serving.py \
  --model "$MODEL_PATH" \
  --workload-name prefill-injection \
  --arrival prefill-injection \
  --num-requests 5 \
  --input-len 128 \
  --long-input-len 4096 \
  --injection-delay 1 \
  --output-len 256 \
  --max-model-len 4608 \
  --max-num-batched-tokens 512 \
  --enable-chunked-prefill
```

该 workload 会先启动 4 个短 prompt 请求，再向正在 decode 的 batch 注入一个长 prompt；去掉 `--enable-chunked-prefill` 即得到同一 workload 的 baseline。

运行 EAGLE speculative decoding 路径：

```bash
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python bench_serving.py \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --speculative-gamma 3 \
  --speculative-accept-mode greedy \
  --speculative-trace \
  --num-requests 1 \
  --input-len 128 \
  --output-len 64 \
  --arrival all \
  --enforce-eager
```

EAGLE benchmark summary 会输出 speculative batch size、acceptance rate、acceptance length、accepted tokens per step、draft tokens per step，以及 draft proposal、target verification、accept/reject、KV update 和 trace overhead 的 timing breakdown。

已在 24GB 显存上验证的配置是 Qwen3-8B、RedHatAI Qwen3-8B EAGLE3 speculator、BF16 eager 和固定 `gamma=3`。output length 256 的三轮实验中，batch 1 吞吐提升 `1.20x`，batch 4 提升 `1.34x`。

## 测试

CPU 单元测试不需要安装 FlashAttention，也不需要本地模型：

```bash
python -m unittest discover -s tests
```

真实 EAGLE checkpoint 集成测试通过环境变量显式启用：

```bash
export LLMSERVE_TEST_TARGET_MODEL=/path/to/Qwen3-target
export LLMSERVE_TEST_SPECULATIVE_MODEL=/path/to/Qwen3-EAGLE3-speculator
python -m unittest discover -s tests
```

Tree KV 的 CUDA kernel 测试只在 CUDA 可用时运行。GitHub Actions 使用 CPU PyTorch 执行其余测试，并上传完整的 `unittest` 输出作为对应 commit 的测试结果。

## 说明

- 代码保留原始 MIT License。
