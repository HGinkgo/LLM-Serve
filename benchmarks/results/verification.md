# Verification Evidence

验证日期：2026-07-22。正式 GPU benchmark 对应代码 commit `ad35e65cacdcb306362268c3a60923abd199b431`。

## CPU 回归

```bash
CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n LLM-Serve \
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
conda run --no-capture-output -n LLM-Serve \
  python -m compileall -q llmserve benchmarks example.py \
  check_speculative_correctness.py

conda run --no-capture-output -n LLM-Serve \
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

## Stage 8 Target Verify CUDA Graph

验证日期：2026-07-29。正式 Graph 对照对应 commit `3bb5d21ad5fd9ae0044943d93255a4542cc5ca75`，使用 Qwen3-8B、EAGLE3 draft、RTX 3090 和 CUDA 12.8。

- `stage8-graph-formal/manifest.json`：18/18 points 完成，零失败。
- 两个 variant 均为 `enforce_eager=true`；Graph variant 额外启用 target verify CUDA Graph，排除普通 decode CUDA Graph 干扰。
- Graph 捕获 6 个固定 shape，batch `1/4/8`、context frontier `256/1024`；正式点均完成，无 OOM 或 capture failure。
- Graph 对 eager 的 output throughput 提升为：concurrency 1/4/8 分别 `1.779x/1.558x/1.419x`。
- 18 个公开 run JSON 已扫描，无绝对路径、prompt token、绝对时间戳、traceback 或凭据。
- 当前 worktree CPU 回归：`217 tests, skipped=5, OK`；4096 max-model-len 的固定 prompt token consistency check 通过，实际命中 Graph 且无 eager fallback。

## Dual-GPU PD KV Pipeline

验证日期：2026-08-03。正式 PD KV transport 对照对应 commit `e779a6aa4186683327060697b8c04f3da12c0284`，使用 Qwen3-8B BF16、双 RTX 3090 和 CUDA 12.8。

- `pd-kv-pipeline-formal/manifest.json`：18/18 points 完成，零失败。
- 两个 variant 均使用 Prefill batch 4、eager Prefill 和 Decode CUDA Graph，只改变 inline Queue Tensor / pinned shared-memory slot transport。
- concurrency 32/48/64 的 shared/inline request throughput 为 `1.098x/1.112x/1.319x`；并发 64 的三轮标准差为 `0.08 req/s`。
- 9 个 shared points 均未触发 inline fallback；每个 run 最终为 `free_slots=2`、`pending_transfers=0`，无 OOM、worker timeout 或 Graph capture failure。
- 18 个公开 run 删除重复的 request records 和 per-batch timing detail，保留 suite 聚合指标、Queue/slot samples、Graph counters 与 worker health；扫描未发现绝对路径、prompt token IDs、traceback 或凭据。
- 合并前 CPU 回归：`274 tests, skipped=5, OK`；真实双卡 4-request smoke 的 token IDs 与 inline 基线完全一致。

## Service Overload Governance

验证日期：2026-08-19。正式结果对应 dirty source commit
`f64e8f0b1e1c2bee6520888abc2a01976286eb2e`，使用 Qwen3-8B、单张 RTX 3090、CUDA 12.8。

- `service-overload-governance/summary.csv`：固定 trace 的 `unbounded` 与 `limit-64` 两个
  points，`2/2` 完成，trace seed `20260818`，trace hash
  `ad9598cb2100149a164484be4ec4dfed9c493c063fec4c06fc59e14c298ba972`。
- 服务基准直接驱动 `EngineServiceRuntime`；指标契约只包含输出/请求吞吐、TTFT、TPOT、
  队列分位数和 admission outcomes，E2E、goodput 与 HTTP 网络开销不在该结果中。
- 本次运行后 GPU1 回到 3 MiB、0% 利用率；GPU0 的外部 PID 770954 仍占用约 350 MiB，
  与本次 benchmark 无关。
