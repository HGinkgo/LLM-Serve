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

## Qwen3 MoE GPTQ 外部对照

`moe_reference` 用外部 vLLM 的 `gptq_marlin` 后端对照 LLM-Serve 对同一 GPTQ
checkpoint 的贪婪生成结果。它只验证模型加载、量化线性层和 MoE 路径的端到端 token
一致性，不报告或比较吞吐、延迟和显存。

参考运行时必须安装在与 LLM-Serve 环境隔离的目录中，并在该机器上完成一次导入验证。
工具仅将该目录注入 vLLM 子进程的 `PYTHONPATH`；候选子进程不会看到它。两个子进程按
顺序运行，避免在一张 24 GiB 卡上同时加载模型。示例中的 vLLM `0.9.1` 是本项目已
验证的参考版本；更换版本或 PyTorch 组合后，必须重新完成对照，不应假定兼容。

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.moe_reference \
  --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --reference-package-dir /path/to/isolate/vllm-0.9.1 \
  --output /tmp/llmserve-moe-gptq-parity.json \
  --prompts-file benchmarks/fixtures/qwen3_moe_gptq_parity.json \
  --max-new-tokens 16 \
  --max-model-len 512 \
  --max-num-batched-tokens 128
```

默认候选实现使用 LLM-Serve 的 `tinygemm` 后端。验证可选 Marlin provider 时，显式
传入 vLLM `0.9.1` 已安装包中的 `_C.abi3.so`，候选子进程只通过 `torch.ops.load_library`
加载该算子库，不导入完整 vLLM Python runtime：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.moe_reference \
  --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --reference-package-dir /path/to/isolate/vllm-0.9.1 \
  --candidate-gptq-backend marlin \
  --candidate-marlin-library /path/to/isolate/vllm-0.9.1/vllm/_C.abi3.so \
  --output /tmp/llmserve-moe-gptq-marlin-parity.json \
  --prompts-file benchmarks/fixtures/qwen3_moe_gptq_parity.json \
  --max-new-tokens 16 \
  --max-model-len 512 \
  --max-num-batched-tokens 128
```

该 provider ABI 固定为已验证的 vLLM `0.9.1`；切换 vLLM 版本需要重新完成数值对照，
并在结果 JSON 中保留 `runtime_config` 的候选后端与动态库路径。

fixture 包含英文、中文、Python、算术和较长上下文，顺序与 case ID 是验收契约。工具先由
checkpoint tokenizer 编码每个 prompt，并把同一份 token IDs 传给两个运行时。两侧都固定
greedy、`ignore_eos` 和相同的生成长度；每个运行时只加载一次模型，随后按 case 顺序以
单请求生成。输出 JSON 记录每个 prompt token hash、参考版本、两侧 token IDs 和首个
不匹配 case/位置。退出码 `0` 表示完整 token trace 一致，退出码 `2` 表示两侧成功运行
但结果不一致。临时诊断可用 `--prompt` 替代 `--prompts-file`；候选侧可加
`--candidate-enable-gate-up-fusion` 验证融合加载与生成路径，默认仍为未融合。

## MoE 后端性能基线

本项目固定使用 vLLM `0.9.1` 的 `gptq_marlin` 作为外部参考基线。每份 MoE
benchmark/profile/parity JSON 都写入 `baseline` 和 `optimization` 元数据，记录参考
运行时、后端、版本/动态库来源以及本次优化是否启用。`moe_backend` 中的吞吐、TTFT、
TPOT 是 LLM-Serve 候选实现的测量值；它们与默认未融合候选的 A/B 差异不能直接称为
“vLLM 吞吐”，除非另行运行 vLLM 自己的测量程序。vLLM parity 报告则作为 token
正确性和 Marlin provider ABI 的外部契约。

`moe_backend` 在独立子进程中顺序运行 TinyGEMM 和 Marlin，保证两次运行不共享
CUDA allocator 或模型进程。父进程先用同一个 checkpoint tokenizer 编码 prompt，
再把固定 token trace 传给两个 worker；每个 worker 只加载一次模型，按相同 batch
size 和输出长度运行。结果记录 output/request throughput、TTFT、TPOT、完成/失败数、
请求级摘要和 CUDA allocated/reserved 峰值，不报告 E2E 或 HTTP 指标。

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n LLM-Serve \
  python -m benchmarks.moe_backend \
  --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --prompts-file benchmarks/fixtures/qwen3_moe_gptq_parity.json \
  --backends tinygemm,marlin \
  --marlin-library /path/to/isolate/vllm-0.9.1/vllm/_C.abi3.so \
  --batch-sizes 1,4 --repeats 1 \
  --max-new-tokens 16 --max-model-len 512 \
  --max-num-batched-tokens 128 \
  --output experiment-data/YYYY-MM-DD_moe-backend-baseline/baseline.json
```

该基线固定 `enforce_eager=True`，不启用 CUDA Graph；默认保持未融合路径。使用
`--enable-gate-up-fusion` 可运行实验性的 expert 内 Gate/Up 投影融合（一次 GPTQ
GEMM 后拆分 Gate/Up），并与默认路径做同一 workload 的 A/B 对照。拿到数据后再用
profiler 判断是否值得改 dispatch 或 kernel；融合加载期的显存峰值也必须单独记录。

完成同一 workload 的未融合/融合 A/B 后，用汇总工具从两份原始 JSON 生成一个不覆盖
输入的 `summary.json`。它按 `(backend, batch_size)` 聚合所有 repeat 的 request 样本，
吞吐按总 token/总 wall time 计算，TTFT/TPOT 重新取全局分位数，并记录两份输入的 SHA-256。

```bash
python -m benchmarks.moe_gate_up_summary \
  --control experiment-data/YYYY-MM-DD_moe-gate-up-fusion-rN/unfused.json \
  --optimized experiment-data/YYYY-MM-DD_moe-gate-up-fusion-rN/fused.json \
  --output experiment-data/YYYY-MM-DD_moe-gate-up-fusion-rN/summary.json
```

汇总中的 `external_baseline` 固定标识 vLLM `gptq_marlin`，而 `candidate_ab_control`
与 `candidate_ab_optimized` 是 LLM-Serve 内部实现对照。两者角色不同，禁止把该 A/B
吞吐差异表述为相对 vLLM 的性能提升。

## GPTQ GEMM microbenchmark

`moe_gptq_gemm` 读取真实 GPTQ checkpoint 的一个 MoE expert projection，固定
`K/N/group_size`，只改变输入 token 数 `M`，对生产 `GPTQLinear.forward()` 做 CUDA
event 计时。默认 workload 是 `model.layers.0.mlp.experts.0.gate_proj`，测试
`M=1,4,8,16,32`；它包含该层的输入类型转换和量化 GEMM 调用，不包含模型加载、
权重预处理和 MoE router/dispatch。结果同时记录每个点的有效 TFLOP/s 与 generation
阶段显存峰值，适合定位小 `M` 下 Marlin 是否占优，不应替代端到端 serving benchmark。

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n LLM-Serve \
  python -m benchmarks.moe_gptq_gemm \
  --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --backends tinygemm,marlin \
  --marlin-library /path/to/isolate/vllm-0.9.1/vllm/_C.abi3.so \
  --m-values 1,4,8,16,32 \
  --warmup 20 --iterations 100 \
  --output experiment-data/YYYY-MM-DD_moe-gptq-gemm/microbenchmark.json
```

## MoE runtime profiler

`moe_profile` 使用固定 prompt token trace，在独立子进程中顺序运行一个短 batch，
通过 `torch.profiler` 聚合量化 GEMM、router、dispatch、expert loop、combine 和
attention 相关事件。默认只记录 eager runtime；它用于定位热点，不替代吞吐基线。
可选 `--trace-dir` 会额外写出 Chrome trace。两个 backend 的模型、prompt、batch、
seed 和 runtime 限制相同。

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n LLM-Serve \
  python -m benchmarks.moe_profile \
  --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --prompts-file benchmarks/fixtures/qwen3_moe_gptq_parity.json \
  --backends tinygemm,marlin \
  --marlin-library /path/to/isolate/vllm-0.9.1/vllm/_C.abi3.so \
  --batch-size 4 --max-new-tokens 4 --warmup-tokens 2 \
  --max-model-len 512 --max-num-batched-tokens 128 \
  --trace-dir experiment-data/YYYY-MM-DD_moe-profile/traces \
  --output experiment-data/YYYY-MM-DD_moe-profile/profile.json
```

报告中的 `events` 是筛选后的聚合表，时间单位为毫秒；`llmserve.*` 是本项目标记，
`aten::index_select`、`aten::index_add`、`_weight_int4pack_mm` 和
`gptq_marlin_gemm` 等是保留的关键原生事件。profiler 结果用于决定是否值得优化
Python dispatch 或 Marlin 调用路径，不能直接解释为端到端加速百分比。

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
