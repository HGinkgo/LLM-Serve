# ThrustLM

[English](README.en.md) | [简体中文](README.md)

ThrustLM 是一个面向单机单卡场景的 LLM 推理引擎项目，重点关注高吞吐 serving 背后的核心系统机制：Paged KV Cache、continuous batching、chunked prefill、serving benchmark，以及实验性的 EAGLE 风格投机解码路径。

项目早期骨架参考了 vLLM PagedAttention 论文和 `nano-vllm` 教学实现。后续的 scheduler 改造、chunked prefill、benchmark 工具和 speculative decoding runtime 都是在本仓库中独立设计和实现的。

仓库名和 Python 包名分别为 `ThrustLM` / `thrustlm`。

## 功能

- PagedAttention 风格的 KV cache 管理：block table、可复用 KV block、prefix-cache-aware block 分配。
- Continuous batching：以 iteration-level scheduler 组织 prefill 和 decode。
- Chunked prefill：decode 优先，把剩余 token budget 分给长 prompt 的 prefill chunk，避免长 prompt 长时间阻塞 decode。
- Serving benchmark：记录 throughput、TTFT、ITL、TPOT、request latency，并提供区分 warmup、measurement 和 drain 的 closed-loop 稳态测量模式。
- Qwen3 模型支持，用于本地单卡实验。

## 代码结构

- `thrustlm/engine/`：请求调度、KV block 管理、target 模型执行和投机解码流程编排。
- `thrustlm/models/`：Qwen3 与 EAGLE3 网络定义及 checkpoint 加载。
- `thrustlm/speculative/`：draft 生成、验证采样、固定候选树算法、Tree KV 管理和共享结果类型。
- `thrustlm/layers/`：attention、sampling 和模型基础组件。
- `bench_serving.py`：有限 workload 与 closed-loop serving benchmark。

`ModelRunner` 持有 target 模型执行资源，`SpeculativeExecutor` 组合这些资源完成 EAGLE decode；与 runtime 无关的算法集中在 `thrustlm/speculative/`。

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
  --num-requests 8 \
  --input-len 4096 \
  --output-len 128 \
  --arrival poisson \
  --request-rate 2 \
  --max-model-len 4352 \
  --max-num-batched-tokens 512 \
  --enable-chunked-prefill
```

运行实验性的 EAGLE speculative decoding 路径：

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

固定候选树是单请求实验能力。在 linear EAGLE 命令上增加 `--speculative-tree-nodes 6 --argmax-sampler` 可运行 Tree-6。该路径使用跨层 Tree KV Manager 和融合提交 kernel，但仍默认关闭，且不支持多请求树。

## 说明

- 代码保留原始 MIT License。
