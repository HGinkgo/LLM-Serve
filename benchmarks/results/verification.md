# Verification Evidence

验证日期：2026-07-22。正式 GPU benchmark 对应代码 commit `ad35e65cacdcb306362268c3a60923abd199b431`。

## CPU 回归

```bash
CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n nano-vllm \
  python -m unittest discover -s tests
```

结果：

```text
Ran 127 tests in 8.788s
OK (skipped=8)
```

8 个 skip 是需要 CUDA、真实 target checkpoint 或 EAGLE3 checkpoint 的可选测试；CPU 测试进程显式隐藏 GPU。

## 静态与入口检查

```bash
conda run --no-capture-output -n nano-vllm \
  python -m compileall -q llmserve benchmarks example.py \
  check_speculative_correctness.py

conda run --no-capture-output -n nano-vllm \
  python -m benchmarks.run_suite --help

git diff --check
```

三条命令均以状态码 0 结束。

## GPU 结果完整性

- `formal-poisson/manifest.json`：`36/36` points 完成，零失败。
- `formal-closed-loop/manifest.json`：`36/36` points 完成，零失败。
- 72 个 run JSON 与两个 manifest 均可由 Python 标准 JSON 解析器读取，且全部为 `complete=true`。
- 结果扫描未发现本地绝对路径、workspace 路径、traceback 或端口冲突错误。

Poisson 与 closed-loop suite 分别使用一张 RTX 3090；双卡只用于并行执行独立 suite，不是 tensor parallel。

## AWQ 收口验证

验证日期：2026-07-24。AWQ 容量结果对应 dirty source commit `b32ba391160fb21e020bbaa7df5f287f38705460`，公开 metadata 保留该事实。

```bash
CUDA_VISIBLE_DEVICES="" conda run -n nano-vllm \
  python -m unittest discover -s tests

CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python -m unittest \
  tests.test_awq_reference tests.test_awq_linear_backend \
  tests.test_awq_linear_profile tests.test_awq_triton tests.test_awq_cuda \
  tests.test_awq_quality tests.test_awq_quantizer \
  tests.test_qwen3_awq_calibration tests.test_quantize_qwen3_awq_cli \
  tests.test_linear_profile -v
```

结果：CPU 回归 `198 tests, skipped=15`；AWQ/CUDA/量化器定向回归 `61 tests`，零失败。`compileall`、`git diff --check` 和 AWQ 公开 CSV/JSON 自校验均以状态码 0 结束。

- LLM-Serve 容量矩阵 BF16/AWQ 共 `24/24` points，零失败，每个 point 都有非零 latency cohort。
- vLLM Marlin 控制实验共 `24/24` points，12 个 AWQ log 均确认 `awq_marlin`。
- `awq-w4a16/` 未包含绝对模型路径、校准文本、checkpoint、逐层 cache 或原始日志。
