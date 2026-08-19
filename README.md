<div align="center">

# LLM-Serve

面向学习与系统研究的 LLM 推理 Runtime

<p>
  <a href="README.en.md">English</a> |
  简体中文
</p>

<p>
  <a href="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml"><img src="https://github.com/HGinkgo/LLM-Serve/actions/workflows/cpu-tests.yml/badge.svg" alt="CPU tests"></a>
  <img src="https://img.shields.io/badge/Model-Qwen3--8B-6f42c1" alt="Model: Qwen3-8B">
  <img src="https://img.shields.io/badge/Runtime-PyTorch%20%7C%20Triton%20%7C%20CUDA-76b900" alt="Runtime: PyTorch Triton CUDA">
  <img src="https://img.shields.io/badge/Serving-Paged%20KV%20%7C%20Continuous%20Batching-0ea5e9" alt="Serving: Paged KV and continuous batching">
  <img src="https://img.shields.io/badge/Advanced-PD%20%7C%20EAGLE-f59e0b" alt="Advanced: PD EAGLE">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</p>

</div>

LLM-Serve 是一个以 Qwen3-8B 为主要目标、面向单机 GPU 推理系统学习的 Runtime。项目参考 [`nano-vllm`](https://github.com/Geeeone/nano-vllm) 的教学骨架，当前主线是把单模型服务的调度、KV 生命周期、过载治理与可复现实验做成可检查的系统闭环，而不是尝试复刻企业级多租户平台。

## 当前主线

- Paged KV Cache、block table、KV 分配/回收，以及 iteration-level Continuous Batching。
- Decode-first Chunked Prefill、结构化 `SchedulerOutput` 与普通 Decode CUDA Graph。
- 单模型 OpenAI-compatible HTTP/SSE：请求取消、health/ready、Prometheus 指标、
  in-flight 准入、429 过载响应和 step-boundary deadline。
- 固定 Poisson trace 的 service benchmark：公平比较无界入队与受限入队，记录
  client-observed TTFT/TPOT、输出/请求吞吐、队列、拒绝与超时。

## 可选与冻结实验

- EAGLE3 线性投机解码默认关闭，只用于可控的 A/B 实验。
- PD + Shared KV Transport 已保留，但当前不扩展动态路由、1P2D 或新的 PD 功能。
- 本分支额外支持 Qwen3-30B-A3B GPTQ-Int4 的单卡 MoE 实验路径。checkpoint
  契约固定为 group size 128、对称量化且关闭 act-order；TinyGEMM 是默认对照后端，
  Marlin 通过显式加载已验证的 vLLM 0.9.1 CUDA 扩展启用。

## 快速开始

要求 Linux、Python 3.10-3.12 和 NVIDIA CUDA GPU。项目验证环境为 CUDA 12.8、PyTorch 2.7.1、Triton 3.3.1 与 FlashAttention 2.8.3。

```bash
conda create -n LLM-Serve python=3.10 pip -y
conda activate LLM-Serve

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging wheel
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .

llmserve check
```

FlashAttention 的预编译 wheel 来自 GitHub Releases；受限网络需要提前准备匹配 `cu12 / torch2.7 / Python 3.10` 的 wheel。源码编译必须使用 CUDA Toolkit 12.8，不能混用系统 CUDA 13。

准备一个本地 Qwen3-8B Hugging Face checkpoint，然后运行：

```bash
llmserve generate \
  --model /path/to/Qwen3-8B \
  --prompt "用通俗的语言解释 PagedAttention" \
  --max-tokens 128
```

也可以直接使用 Python API：

```python
from llmserve import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-8B")
try:
    outputs = llm.generate(
        ["Explain continuous batching in one paragraph."],
        SamplingParams(temperature=0.6, max_tokens=128),
    )
    print(outputs[0]["text"])
finally:
    llm.exit()
```

需要自行驱动连续批处理时，可使用 `add_request()` / `step()`，并通过
`abort_request(request_id)` 取消请求。取消仅在两次 `step()` 之间生效，
不会中断正在执行的 GPU step；Baseline、EAGLE 与 PD 路径采用相同语义。
PD Worker 或 RPC 故障会终止当前 `PDServingEngine`，并将未完成请求标记为
`failed`；继续服务需要创建新的 Engine 实例。

## HTTP 服务

安装 Web 服务依赖：

```bash
pip install -e '.[serve]'
```

默认以强单卡 Collocated Runtime 启动，保留 Paged KV、Continuous Batching、
decode-first Chunked Prefill、KV 准入与普通 Decode CUDA Graph：

```bash
llmserve serve \
  --model /path/to/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 0.0.0.0 --port 8000
```

服务提供 `/v1/models`、`/v1/completions`、`/v1/chat/completions`、
`/health/live`、`/health/ready` 和 `/metrics`。`stream: true` 使用真实的
Server-Sent Events token 流；客户端断开会在下一个 Engine step 取消请求并回收
请求状态。

这里提供的是 HTTP 服务层而非 TLS 终止；需要 HTTPS 时，应由反向代理或部署环境
负责证书与 TLS。项目自身的可验证边界是单模型请求生命周期与 Runtime 行为。

服务入口默认将总在途请求限制为 Runtime 的可服务序列数；超限请求返回 `429` 和
`Retry-After: 1`，不会进入无界等待队列。可用 `--max-inflight-requests` 设定更小的
服务配额；`--request-timeout-seconds` 会在下一个 Engine step 边界取消超时请求。

```bash
curl http://127.0.0.1:8000/health/ready

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"解释连续批处理"}],"max_tokens":64,"temperature":0.6}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"解释连续批处理"}],"max_tokens":64,"temperature":0.6,"stream":true}'
```

PD+Shared 是显式双 GPU 部署模式，不会自动路由请求。它使用独立 Prefill/Decode
Worker、双槽位 pinned shared-memory KV、descriptor-only handoff 与 ACK/backpressure：

```bash
CUDA_VISIBLE_DEVICES=0,1 llmserve serve \
  --mode pd-shared \
  --model /path/to/Qwen3-8B \
  --pd-prefill-gpu 0 --pd-decode-gpus 1 \
  --pd-prefill-batch-size 4 \
  --pd-kv-slot-capacity-tokens 8192
```

## 文档

- [基础 Python 示例](example.py)
- [Prefill/Decode 示例](examples/)
- [Benchmark 使用与结果索引](benchmarks/README.md)
- [测试分层与命令](tests/README.md)

根 README 不重复维护性能数字；实验配置、指标口径和公开结果以 Benchmark 文档为准。

## 当前边界

- 主要支持 Qwen3-8B、单卡 TP=1；双卡用于 Prefill/Decode 分离，不用于 Tensor Parallel。
- EAGLE 和 PD 是实验性/受限能力；Chunked Prefill、KV 容量准入和普通 Decode CUDA Graph 是单机服务基线的一部分。
- Dense Qwen3 路径仅支持非量化 checkpoint；MoE 路径仅支持上述 GPTQ-Int4
  checkpoint，限制为单卡 TP=1 与 eager 执行。HTTP 服务当前为单模型、纯文本接口，
  不提供鉴权、工具调用、多模态或跨节点路由。
- PD 服务与 EAGLE 尚未耦合；PD 模式会拒绝 `--speculative-model`，避免把未实现组合包装成可用能力。

## License

[MIT](LICENSE)
