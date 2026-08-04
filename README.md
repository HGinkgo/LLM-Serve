<div align="center">

# LLM-Serve

面向学习与系统研究的单机 LLM 推理 Runtime

<p>
  <a href="README.en.md">English</a> |
  简体中文
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

LLM-Serve 是一个以 Qwen3-8B 为主要目标、面向单机 GPU 推理系统学习的 Runtime。项目从单卡推理出发，逐步实现 Paged KV Cache、continuous batching、chunked prefill、EAGLE 风格投机解码、AWQ W4A16 和双卡 Prefill/Decode 分离。

项目早期参考了 [PagedAttention 论文](https://arxiv.org/abs/2309.06180) 与 [`nano-vllm`](https://github.com/Geeeone/nano-vllm) 的教学骨架；scheduler、serving benchmark、投机解码、量化校准和 PD serving 由本仓库独立演进。

## 核心能力

- **Runtime**：Paged KV Cache、block table、prefix cache、iteration-level continuous batching 和显式 `SchedulerOutput`。
- **调度**：decode-first chunked prefill，支持 mixed prefill/decode batch 和长 prompt 隔离。
- **投机解码**：EAGLE 风格 batched draft、packed target verification、per-request draft KV、greedy accept/reject，以及 target verify CUDA Graph。
- **量化**：Qwen3 activation-aware AWQ W4A16 校准、标准 AutoAWQ GEMM checkpoint 导出、reference/Triton/CUDA Linear backend 和 KV capacity admission。
- **双卡 Serving**：独立 Prefill/Decode Worker、logical KV handoff、pinned shared-memory slots、ACK/backpressure 和 Decode CUDA Graph。
- **可复现实验**：Poisson request-rate 与 closed-loop concurrency runner，记录 throughput、goodput、TTFT、TPOT、E2E、队列和阶段耗时。

## 项目结构

```text
llmserve/
├── engine/        scheduler、KV block 管理、target 执行与 speculative 编排
├── models/        Qwen3 与 EAGLE3 模型定义
├── speculative/   draft、verification、sampling 与 target CUDA Graph
├── quantization/  AWQ 校准、checkpoint 导出与质量评估
├── pd/            Prefill/Decode 协议、KV handoff、共享槽位与 worker 生命周期
└── layers/        attention、linear、sampling 等基础组件
benchmarks/        workload、arrival、指标、suite runner 与公开结果
tests/             CPU 单元测试、checkpoint 集成测试与 CUDA 测试
```

## 快速开始

```bash
pip install -e .

export MODEL_PATH=/path/to/Qwen3-8B
python example.py
```

运行 CPU 回归：

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests
```

运行 GPU smoke：

```bash
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python -m benchmarks.run_suite \
  --suite benchmarks/suites/smoke.json \
  --output-dir /tmp/llmserve-smoke \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --allow-dirty
```

## 实验与文档

- [Benchmark 使用说明](benchmarks/README.md)：suite、workload、指标口径和复现实验命令。
- [公开 Benchmark 数据](benchmarks/results/)：manifest、CSV、脱敏 run JSON 和各阶段说明。
- [PD 端到端结果](benchmarks/results/pd-serving-formal/)：单进程 collocated 与双卡 Prefill/Decode 对照。
- [AWQ 结果](benchmarks/results/awq-w4a16/)：质量、容量和 vLLM Marlin 控制实验。
- [完整测试证据](benchmarks/results/verification.md)：CPU 回归、CUDA smoke 和公开数据校验记录。

实验结果不在根 README 中重复维护，具体数字和原始脱敏数据以 `benchmarks/results/` 下对应目录为准。

## 当前边界

- 主目标是 Qwen3-8B、单卡 TP=1 和 RTX 3090 24GB；双卡路径用于 Prefill/Decode 分离，不作为无 NVLink Tensor Parallel 平台。
- speculative CUDA Graph 只支持线性 EAGLE、greedy acceptance 和显式 opt-in；不支持的 shape 会回退 eager。
- AWQ Runtime 固定为 Qwen3、AutoAWQ GEMM、group-128 W4A16、BF16 activation/scales 和 TP=1；vLLM Marlin 结果属于外部执行后端控制实验。
- 项目聚焦 Runtime、调度和 serving 机制，不包含 OpenAI-compatible HTTP API 层。

## License

[MIT](LICENSE)
