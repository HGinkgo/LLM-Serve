# Serving Benchmark

`benchmarks` 直接驱动 in-process `LLMEngine`，用于研究 runtime 与 scheduler，不包含 HTTP/OpenAI API 胶水层。

## 目录

- `workloads.py`：按权重生成确定性 token workload。
- `arrivals.py`：可复现的 Poisson 到达时间。
- `runtime.py`：有限 Poisson 与 closed-loop warmup/measurement/drain 循环。
- `metrics.py`：吞吐、延迟、goodput 和 EAGLE 聚合。
- `serve.py`：执行单个 benchmark point。
- `run_suite.py`：展开 suite，每点启动独立子进程，处理恢复、失败和 CSV 汇总。
- `linear_profile.py`：按 Qwen3 实际 QKV、O、gate/up、down shape 测量 dense Linear CUDA latency。
- `awq_linear_profile.py`：从真实 AWQ checkpoint 加载 layer 权重，对比反量化、matmul、reference、Triton 与自研 CUDA forward。
- `suites/`：smoke、serving 正式矩阵和 AWQ 容量确认矩阵。
- `results/`：公开 manifest、逐运行 CSV、aggregate CSV 与完整脱敏 run JSON。

## Workload

正式矩阵包含两个 profile：

| Profile | 请求组成 | 用途 |
| :--- | :--- | :--- |
| decode-heavy | 100% `256 input / 256 output` | EAGLE 与持续 decode 容量 |
| mixed-serving | 80% `128 input / 128 output` + 20% `4096 input / 128 output` | chunked prefill 与尾延迟 |

Poisson 主实验扫描 decode `{0.25, 0.75, 1.25} req/s`、mixed `{0.5, 1.5, 2.5} req/s`。closed-loop 补充实验扫描 decode concurrency `{1, 4, 8}`、mixed concurrency `{4, 8, 16}`，使用 5 秒 warmup 与 60 秒 measurement。

所有 A/B variant 在相同 run 内共享 workload seed 和 arrival seed。Poisson 正式点在 measurement 前对每个请求类别做短 warmup，然后清空 engine metrics；模型加载与首次 CUDA 执行不计入正式请求。

## 运行

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export SPECULATIVE_MODEL=/path/to/Qwen3-8B-speculator.eagle3

python -m benchmarks.run_suite \
  --suite benchmarks/suites/formal-poisson.json \
  --output-dir /tmp/llmserve-formal-poisson \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --resume
```

正式运行默认拒绝 dirty worktree；`--allow-dirty` 只用于 smoke/pilot。`--resume` 仅接受 schema、point ID 和 commit SHA 均匹配且 `complete=true` 的结果。

每个 point 在独立子进程中加载/释放模型，避免 CUDA 和 KV 状态跨点污染。worker 失败时仍写入 `complete=false` JSON，suite 继续执行其他点并最终返回非零。并行运行两个 suite 时必须使用不同 endpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.run_suite \
  --suite benchmarks/suites/formal-poisson.json \
  --output-dir /tmp/poisson \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --distributed-init-method tcp://localhost:2333

CUDA_VISIBLE_DEVICES=1 python -m benchmarks.run_suite \
  --suite benchmarks/suites/formal-closed-loop.json \
  --output-dir /tmp/closed-loop \
  --model "$MODEL_PATH" \
  --speculative-model "$SPECULATIVE_MODEL" \
  --distributed-init-method tcp://localhost:2334
```

短 Linear profiling 用于定位 AWQ kernel 优化优先级：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.linear_profile \
  --model "$MODEL_PATH" \
  --dtype bfloat16 \
  --num-tokens 1,4,8,16 \
  --output /tmp/qwen3-dense-linear.json
```

这里的 `num_tokens` 是 GEMM 的 M，不是 request concurrency。输出是隔离的 Linear microbenchmark，只用于比较投影 shape 和执行后端；端到端结论仍以 Poisson/closed-loop suite 为准。

AWQ Linear backend profiling：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.awq_linear_profile \
  --model /path/to/Qwen3-8B-AWQ \
  --num-tokens 1,4,8,16 \
  --warmup 10 \
  --repeats 50 \
  --output /tmp/qwen3-awq-linear.json
```

`dequantize` 与 `matmul` 分别隔离计时，`reference`、`triton` 与 `cuda` 统计三条完整 Linear backend。自研 CUDA 路径在模型加载时完成权重 repack，并复用预分配的 split-K workspace；这些 microbenchmark 用于定位 kernel 工作，不替代端到端延迟。

自制 Qwen3 AWQ checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.run_suite \
  --suite benchmarks/suites/awq-capacity-confirm.json \
  --output-dir /tmp/llmserve-awq-capacity \
  --model /path/to/Qwen3-8B-LLMServe-AWQ \
  --distributed-init-method tcp://localhost:2335 \
  --resume
```

同一 suite 还要用 BF16 checkpoint 在同型号 GPU 上顺序运行，并使用不同输出目录和 distributed endpoint。矩阵覆盖 concurrency `{48, 64, 96, 128}`，每点 warmup `45s`、measurement `120s`、重复三次；它是长测试，不属于日常回归。

量化与导出命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m llmserve.quantization.quantize_qwen3 \
  --model /path/to/Qwen3-8B \
  --output /path/to/Qwen3-8B-LLMServe-AWQ \
  --calib-file /path/to/calibration.txt \
  --cache-dir /tmp/qwen3-awq-calibration-cache
```

质量验证使用 `python -m llmserve.quantization.awq_quality`，分别以 `--mode bf16` 和 `--mode awq-reference` 运行。当前量化器固定为 Qwen3、W4A16、group size 128、非对称 zero，并输出标准 AutoAWQ GEMM checkpoint。CUDA backend 只支持该格式、BF16 activation/scales、SM80+、eager 和单卡 TP=1；首次加载会通过 PyTorch JIT 编译扩展。

## 输出

每个 suite 输出：

- `manifest.json`：commit、dirty 状态、环境、GPU、模型名/revision、完成/失败点数。
- `runs/*.json`：脱敏配置、指标和请求级 compact records；不保存绝对模型路径、prompt token、绝对时间戳或 speculative trace。
- `summary.csv`：每个 run 一行，延迟统一为毫秒。
- `aggregate.csv`：三次重复的均值、样本标准差和 ratio-to-baseline，显式记录单位。
- `points/*.json`：展开后的 point 配置；公开结果不重复提交这些文件，因为 `suites/*.json` 已可重建。

## 指标语义

- request/input/output/total throughput：完整请求或 measurement window 的容量指标。
- `TTFT`：计划到达时刻到首 token；Poisson step 中途到达但稍后才入队的延迟也会计入。
- `TPOT`：请求首 token 到完成之间的平均每输出 token 时间。
- `burst_itl`：逐 token 可用时间间隔；EAGLE 同一 burst 的 token 会包含 `0 ms`。
- `output_event_latency`：相邻输出事件间隔，每个 request/emitting step 只记录一次。
- `speculative_step_latency`：完整 speculative step 成本。
- `E2E`：计划到达到请求完成。
- `goodput`：满足 suite 中 TTFT/TPOT/E2E SLO 的请求吞吐。
- queue/batch metrics：scheduled batch、speculative batch、waiting/running queue 的分布。
- EAGLE metrics：steps、draft/accepted/emitted token、acceptance rate/length、gamma histogram 与 timing。

closed-loop output throughput 统计 measurement window 内产生的 token；request throughput 统计窗口内完成的请求。延迟和 speculative 指标只聚合 arrival 与 finish 都落在窗口内的请求，`latency_sample_requests` 用于判断 P50/P99 是否有有效样本。

## 公开结果边界

公开数据基于 commit `ad35e65`，Qwen3-8B、RedHatAI Qwen3-8B EAGLE3 speculator、BF16 eager、argmax、固定 `gamma=3` 和 RTX 3090 24GB。Poisson 与 closed-loop suite 同时在两张独立 3090 上运行，但每个 suite/engine 始终只使用单卡，不是 tensor parallel。

核心结论见仓库根目录 README；serving 原始数值以 [`results/formal-poisson/aggregate.csv`](results/formal-poisson/aggregate.csv) 和 [`results/formal-closed-loop/aggregate.csv`](results/formal-closed-loop/aggregate.csv) 为准，AWQ 脱敏汇总与实验边界见 [`results/awq-w4a16/`](results/awq-w4a16/)。
