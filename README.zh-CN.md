# ThrustLM

[English](README.md) | 简体中文

ThrustLM 是一个面向单机单卡场景的 LLM 推理引擎项目，重点关注高吞吐 serving 背后的核心系统机制：Paged KV Cache、continuous batching、chunked prefill、serving benchmark，以及实验性的 EAGLE 风格投机解码路径。

项目早期骨架参考了 vLLM PagedAttention 论文和 `nano-vllm` 教学实现。后续的 scheduler 改造、chunked prefill、benchmark 工具和 speculative decoding runtime 都是在本仓库中独立设计和实现的。

仓库名和 Python 包名分别为 `ThrustLM` / `thrustlm`。

## 功能

- PagedAttention 风格的 KV cache 管理：block table、可复用 KV block、prefix-cache-aware block 分配。
- Continuous batching：以 iteration-level scheduler 组织 prefill 和 decode。
- Chunked prefill：decode 优先，把剩余 token budget 分给长 prompt 的 prefill chunk，避免长 prompt 长时间阻塞 decode。
- Serving benchmark：记录 throughput、TTFT、ITL、TPOT、request latency、wall time、成功/失败请求数。
- Qwen3 模型支持，用于本地单卡实验。

## 实验性功能

- EAGLE 风格 speculative decoding MVP：包含 draft proposal、target verification、draft KV 状态维护、merged correction 处理和 acceptance metrics。
- Speculative trace 和 timing breakdown：用于观察 draft token、target verification、acceptance length，以及每个 speculative step 的耗时构成。
- 当前推荐路径是 greedy/top-k 风格 verify，更适合 hot-vocabulary EAGLE3 draft checkpoint。
- 概率拒绝采样工具保留用于算法学习和受控实验，但对于 32K hot-vocabulary draft head，它不是当前推荐路径。

## 快速开始

以 editable 模式安装：

```bash
pip install -e .
```

设置本地模型路径：

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
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
export SPECULATIVE_MODEL=/path/to/eagle3-draft

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

EAGLE benchmark summary 会输出 acceptance rate、acceptance length、accepted tokens per step、draft tokens per step，以及 draft proposal、target verification、accept/reject、KV update 和 trace overhead 的 timing breakdown。

## 说明

- 代码保留原始 MIT License。
