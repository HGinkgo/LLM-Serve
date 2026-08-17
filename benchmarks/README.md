# Benchmark Harness

`benchmarks` 直接驱动进程内 Engine 验证 Runtime 行为，不是 HTTP 压测客户端。
每个 point 在独立进程中运行，输出 manifest、逐运行 JSON、`summary.csv` 和
`aggregate.csv`。

## 保留套件

- `smoke.json`：短时单卡回归。
- `awq-capacity-confirm.json`：BF16/AWQ 质量与容量边界。
- `stage8-graph-formal.json`：EAGLE Target Verify CUDA Graph 对照。
- `pd-resource-equivalent-formal.json`：单卡、双 Collocated 副本与 PD+Shared 的资源对照。
- `pd-phase-map-smoke.json`：全短、decode-heavy、平衡混合与 Prefill-heavy 下的 PD 判别。

PD 标准部署使用 Shared KV。Inline Queue、旧的非 Chunked 基线与已完成传输专项
均已移除，不能再作为性能基线。

## 运行

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python -m benchmarks.run_suite \
  --suite benchmarks/suites/pd-resource-equivalent-formal.json \
  --output-dir /tmp/llmserve-pd-resource \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL"
```

正式套件默认拒绝 dirty worktree。AWQ、PD 和 EAGLE 长测不是日常回归；改动
Runtime 后至少运行 `smoke.json` 或对应 CPU 单元测试。

## 指标

- `output_tokens_per_second`：measurement window 内生成的输出 token 数除以窗口时长。
- `requests_per_second`：measurement window 内完成请求数除以窗口时长。
- `TTFT`：Engine 接收请求到模型采样首 token 的时间，不是客户端收到网络响应的时间。
- `TPOT`：首 token 到完成之间的平均每 token 时间。
- `E2E`：Engine 接收请求到完成的时间。
- `goodput`：满足 suite SLO 的请求吞吐。

EAGLE 的 `burst_itl` 可包含一次验证 burst 内的 `0 ms`；比较时应同时查看
`output_event_latency`、TPOT 和 E2E。
