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
