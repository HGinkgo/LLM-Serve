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
  <img src="https://img.shields.io/badge/Advanced-PD%20%7C%20EAGLE%20%7C%20AWQ-f59e0b" alt="Advanced: PD EAGLE AWQ">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</p>

</div>

LLM-Serve 是一个以 Qwen3-8B 为主要目标、面向单机 GPU 推理系统学习的 Runtime。项目参考 [`nano-vllm`](https://github.com/Geeeone/nano-vllm) 的教学骨架，并围绕调度、KV Cache、投机解码、量化和 Prefill/Decode 分离持续演进。

## 核心能力

- Paged KV Cache、Prefix Cache 与 iteration-level Continuous Batching。
- Decode-first Chunked Prefill 与结构化 `SchedulerOutput`。
- EAGLE 风格批量草稿生成、打包验证和 Target Verify CUDA Graph。
- Qwen3 AWQ W4A16 校准、标准 checkpoint 导出和多种 Linear backend。
- 双卡 Prefill/Decode Worker、共享内存 KV handoff 与背压控制。
- Poisson 和 closed-loop Serving Benchmark，覆盖吞吐、TTFT、TPOT、E2E 与队列指标。

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

## 文档

- [基础 Python 示例](example.py)
- [Prefill/Decode 示例](examples/)
- [Benchmark 使用与结果索引](benchmarks/README.md)

根 README 不重复维护性能数字；实验配置、指标口径和公开结果以 Benchmark 文档为准。

## 当前边界

- 主要支持 Qwen3-8B、单卡 TP=1；双卡用于 Prefill/Decode 分离，不用于 Tensor Parallel。
- EAGLE、Chunked Prefill、KV 容量准入和 Speculative CUDA Graph 均为显式可选能力。
- AWQ 路径由 checkpoint 元数据启用；当前不提供 OpenAI-compatible HTTP API。

## License

[MIT](LICENSE)
