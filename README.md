# LLM-Serve

[English](README.en.md) | [简体中文](README.md)

[![CPU tests](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml)

LLM-Serve 是一个面向单机单卡场景的教学型 LLM 推理引擎，重点实现并验证高吞吐 serving 背后的核心机制：Paged KV Cache、continuous batching、chunked prefill、serving benchmark，以及 EAGLE 风格投机解码。

项目早期骨架参考了 vLLM PagedAttention 论文和 `nano-vllm` 教学实现。后续 scheduler 改造、chunked prefill、benchmark 系统和 speculative decoding runtime 均在本仓库中独立设计与实现。

## 功能

- PagedAttention 风格 KV Cache：block table、KV block 分配回收和 prefix cache。
- Continuous batching：iteration-level scheduler，显式区分 prefill/decode 序列。
- Chunked prefill：decode-first，把剩余 token budget 分给长 prompt prefill。
- EAGLE 风格投机解码：batched draft、packed target verification、per-request draft KV、greedy verification，以及 acceptance/timing 指标。
- Serving benchmark：Poisson request-rate 扫描与 closed-loop 固定并发，覆盖吞吐、goodput、TTFT、TPOT、burst ITL、output-event latency、E2E、queue depth 和 speculative timing。
- Qwen3-8B 单卡 BF16 路径，以及可选的固定候选树实验实现。

## 代码结构

- `llmserve/engine/`：scheduler、KV block 管理、target 执行与投机解码编排。
- `llmserve/models/`：Qwen3 与 EAGLE3 网络定义和 checkpoint 加载。
- `llmserve/speculative/`：draft、verification sampling、固定树与 Tree KV 管理。
- `llmserve/layers/`：attention、linear、sampling 等基础组件。
- `benchmarks/`：workload、arrival、指标、单点 runner、suite runner、公开结果。
- `tests/`：CPU 单元测试、可选真实 checkpoint 集成测试和 CUDA kernel 测试。

`ModelRunner` 持有 target 模型执行资源，`SpeculativeExecutor` 组合这些资源完成 EAGLE decode；与 runtime 无关的算法集中在 `llmserve/speculative/`。

## 快速开始

```bash
pip install -e .

export MODEL_PATH=/path/to/Qwen3-8B
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python example.py
```

运行四点 GPU smoke，验证 baseline、EAGLE 和 chunked prefill 路径：

```bash
python -m benchmarks.run_suite \
  --suite benchmarks/suites/smoke.json \
  --output-dir /tmp/llmserve-smoke \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --allow-dirty
```

在干净 commit 上运行正式 Poisson 主实验：

```bash
python -m benchmarks.run_suite \
  --suite benchmarks/suites/formal-poisson.json \
  --output-dir /tmp/llmserve-formal-poisson \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --resume
```

`formal-closed-loop.json` 是固定并发补充实验。两个 suite 并行使用两张卡时，需要分别指定不同的 `--distributed-init-method`，例如 `tcp://localhost:2333` 与 `tcp://localhost:2334`。每个点都在独立子进程中运行，失败会记录 JSON 并继续，其余细节见 [`benchmarks/README.md`](benchmarks/README.md)。

## Benchmark 结果

正式结果基于 commit `ad35e65`，Qwen3-8B + RedHatAI Qwen3-8B EAGLE3 speculator，BF16 eager，固定 `gamma=3`，argmax，单张 RTX 3090 24GB。每个配置重复三次；Poisson 与 closed-loop 回答不同问题，不能混成一个 speedup。

### EAGLE

decode-heavy workload 为 `256 input / 256 output`。closed-loop 中 EAGLE 明显提高饱和吞吐，但并发 4/8 的 request E2E P99 同时变差：

| 并发 | Baseline output tok/s | EAGLE output tok/s | 吞吐比 | E2E P99 比 |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 25.08 | 41.01 | **1.635x** | 0.929x |
| 4 | 89.20 | 153.40 | **1.720x** | 1.257x |
| 8 | 172.36 | 267.39 | **1.551x** | 1.421x |

Poisson request-rate `{0.25, 0.75, 1.25}` 下，有限 workload 的 output throughput 只提高 `1.025x-1.042x`，E2E P99 为 baseline 的 `1.068x-1.220x`。这说明 EAGLE 的收益高度依赖持续饱和与 batch 形态，不能用 closed-loop 峰值代替在线到达场景结论。三档 acceptance rate 均值约 `45.5%-46.8%`，acceptance length 约 `2.37-2.40 tokens/step`。

### Chunked Prefill

mixed-serving workload 为 80% `128 input / 128 output` 与 20% `4096 input / 128 output`。closed-loop 结果：

| 并发 | Output throughput 比 | TTFT P99 变化 | Short TTFT P99 变化 |
| :--- | ---: | ---: | ---: |
| 4 | 1.004x | **-20.1%** | **-19.7%** |
| 8 | 1.044x | **-6.3%** | **-21.0%** |
| 16 | 1.092x | **-20.4%** | **-17.2%** |

Poisson request-rate `{0.5, 1.5, 2.5}` 下，吞吐基本持平（`0.992x-1.000x`），但 TPOT P99 降低约 `19%-20%`，E2E P99 降低约 `14%-17%`。因此本项目把 chunked prefill 定位为调度与尾延迟优化，而不是无条件的吞吐优化。

完整的 72 份脱敏 run JSON、逐运行 CSV、三轮均值/标准差和 manifest 位于 [`benchmarks/results/`](benchmarks/results/)。

## 指标口径

- `burst_itl`：逐 token 可用时间间隔；speculative burst 会自然产生 `0 ms` 样本。
- `output_event_latency`：同一请求相邻输出事件的时间间隔，每个 emitting engine step 只记录一次。
- `speculative_step_latency`：完整一次 draft/verify/accept/KV step 成本。
- `TPOT`：请求级平均每输出 token 时间。
- closed-loop 吞吐只统计 measurement window；延迟与 acceptance 只统计 arrival/finish 均落在窗口内的请求，并显式报告 `latency_sample_requests`。

## 测试

CPU 单元测试不需要本地模型：

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests
```

真实 EAGLE checkpoint 集成测试通过环境变量显式启用：

```bash
export LLMSERVE_TEST_TARGET_MODEL=/path/to/Qwen3-8B
export LLMSERVE_TEST_SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3
python -m unittest discover -s tests
```

Tree KV CUDA kernel 测试只在 CUDA 可用时运行。GitHub Actions 使用 CPU PyTorch 执行其余测试。

## 边界

- 当前主配置是单卡 Qwen3-8B，不把双卡无 NVLink 环境包装成 tensor-parallel 性能平台。
- speculative CUDA Graph 尚未实现；固定候选树默认关闭，不作为主性能结论。
- 项目不提供 OpenAI HTTP API 层，benchmark 直接驱动 in-process engine，聚焦 runtime 与 scheduler。
- 代码保留原始 MIT License。
