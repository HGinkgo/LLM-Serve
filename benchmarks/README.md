# Benchmark Harness

`benchmarks` 直接驱动进程内 Engine 验证 Runtime 行为，不是 HTTP 压测客户端。
常规套件和 `service_overload` 都让每个 point 在独立进程中运行，避免前一个点的
CUDA allocator 或 driver 状态影响下一个点。输出 manifest、逐运行 JSON、
`summary.csv` 和 `aggregate.csv`。

`service_overload` 是例外：它直接驱动 `EngineServiceRuntime`，验证服务层的
in-flight 准入和流式事件时序；HTTP 的 429 映射由 API 单元测试覆盖，不把网络
客户端开销混入该实验。

## 保留套件

- `smoke.json`：短时单卡回归。
- `stage8-graph-formal.json`：EAGLE Target Verify CUDA Graph 对照。

PD 标准部署使用 Shared KV。Inline Queue、旧的非 Chunked 基线、双 Collocated
实验运行时与已完成传输专项均已移除，不能再作为性能基线。

## 运行

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python -m benchmarks.run_suite \
  --suite benchmarks/suites/smoke.json \
  --output-dir /tmp/llmserve-smoke \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL"
```

正式套件默认拒绝 dirty worktree。PD 和 EAGLE 长测不是日常回归；改动
Runtime 后至少运行 `smoke.json` 或对应 CPU 单元测试。

## 服务过载扫描

同一固定 Poisson request trace 下，对照无服务准入上限与有限 in-flight 上限：

```bash
python -m benchmarks.service_overload \
  --model "$MODEL_PATH" \
  --output-dir /tmp/llmserve-service-overload \
  --request-rates 24 \
  --inflight-limits unbounded,64
```

输出的 `traces/` 保存 token trace 和 arrival trace，`runs/` 保存逐策略原始 JSON，
`summary.csv` 汇总 accepted/rejected、window 内 output/request throughput、
客户端可见 TTFT/TPOT P50/P99，以及每 250 ms 采样的服务队列深度。`unbounded`
明确关闭服务层上限，不会隐式回退到 `max_num_seqs`。同一 request rate 下的策略使用
同一个 trace 文件、prompt token、seed 和 arrival schedule；worker 在启动前复算
trace hash，发生不一致时该点失败而不执行。

在解释过载结果前，先用单请求诊断排除模型加载与 service driver 差异。两条命令必须
使用同一个模型和 Engine 参数，只改变 `--mode`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.service_startup \
  --mode direct --model "$MODEL_PATH" \
  --output /tmp/llmserve-direct-startup.json \
  --max-model-len 512 --max-num-batched-tokens 1024 --max-num-seqs 64

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.service_startup \
  --mode service --model "$MODEL_PATH" \
  --output /tmp/llmserve-service-startup.json \
  --max-model-len 512 --max-num-batched-tokens 1024 --max-num-seqs 64
```

诊断 JSON 包含 git 状态、模型 revision、完整配置、终态、原始异常和退出后的 CUDA
显存快照。它不是性能测试；`request_ok: false` 时不应把后续的吞吐或延迟数字当作
有效结论。

## 指标

- `output_tokens_per_second`：measurement window 内生成的输出 token 数除以窗口时长。
- `requests_per_second`：measurement window 内完成请求数除以窗口时长。
- Engine benchmark 的 `TTFT`：Engine 接收请求到模型采样首 token 的时间，不是客户端收到网络响应的时间。
- `service_overload` 的 `TTFT`：调用 `EngineServiceRuntime.submit` 前一刻到 benchmark 客户端观察到首个 token event；该工具不包含 HTTP 网络栈。
- `service_overload` 的 `TPOT`：首、末 token event 的时间差除以 `token_count - 1`；只有一个输出 token 时显式记为不可计算，不以完成事件替代末 token 时间。

`throughput` 和 `measurement_window` outcomes 按事件落在 measurement window 内归属；
延迟使用固定 trace 中计划 arrival 落在 measurement interval 的 cohort，而每个请求的
TTFT/TPOT 仍从实际调用 `submit()` 前一刻开始。每个计划 arrival 都有独立
client submitter，避免一个阻塞 admission 改写后续 arrival schedule。`queue_depth` 的
分位数只使用带有 `phase: measurement` 的样本，完整 warmup/measurement/drain 采样仍
保存在原始 JSON 以便排查。

EAGLE 的 `burst_itl` 可包含一次验证 burst 内的 `0 ms`；比较时应同时查看
`output_event_latency` 和 TPOT。
