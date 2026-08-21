# MoE Gate/Up Fusion A/B

验证日期：2026-08-20。模型为 Qwen3-30B-A3B-GPTQ-Int4，单张 RTX 3090（GPU 1），
LLM-Serve 使用 eager runtime，固定 5 个 prompt、seed `20260819`、batch size `1,4`、
每点 3 次、每次生成 16 tokens。未融合和融合运行使用同一份 prompt token trace，
trace SHA-256 为
`f89b7f00654ccab42d3be2c1f1eb486df8343ae6724f91a5e8139064d2a337ae`。

外部参考基线固定为 vLLM `0.9.1` 的 `gptq_marlin`；本表的 control/optimized 数字是
LLM-Serve 内部 Marlin 候选的未融合/融合 A/B，不是 vLLM Python runtime 的吞吐数字。
两侧均通过 vLLM `0.9.1` 的 Marlin provider，并已完成 vLLM token parity（5/5 cases）。

## Marlin A/B

| batch | control output tok/s | optimized output tok/s | throughput delta | TTFT P50 ms | TTFT delta | TPOT P50 ms | TPOT delta | peak allocated delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.8267 | 3.3780 | +19.50% | 999.46 -> 818.91 | -18.06% | 309.27 -> 261.23 | -15.53% | +19.25% |
| 4 | 4.2335 | 5.2442 | +23.87% | 2443.07 -> 1916.55 | -21.55% | 846.21 -> 687.18 | -18.79% | +19.25% |

Reserved memory变化约为 `+0.04%`；allocated 峰值增加主要对应融合路径获得了更大的
KV cache 配额，而不是模型常驻权重膨胀。原始请求级数据中 completed/failed 均为
batch 1 的 `3/0`、batch 4 的 `12/0`（每个 backend、batch 组合均运行 3 次）。

TinyGEMM 也记录在同一份 raw/summary JSON 中，吞吐增益为 batch 1 `+18.53%`、batch 4
`+24.68%`，但 Marlin 才是本项目固定的 serving candidate 和外部基线对照路径。

## Controlled Marlin Profile

同一 prompt trace、batch 4、生成 4 tokens、warmup 2 tokens 的未融合/融合 profile
分别运行一次。profile 只用于解释热点，不替代上面的吞吐测量。

| event | control | optimized | delta |
| --- | ---: | ---: | ---: |
| Marlin calls | 24,489 | 16,580 | -32.3% |
| `gptq_marlin_gemm` CUDA ms | 247.31 | 149.34 | -39.6% |
| `llmserve.moe.expert_loop` CUDA ms | 5,862.38 | 4,629.17 | -21.0% |
| `llmserve.moe.dispatch` CUDA ms | 70.39 | 69.82 | -0.8% |
| `llmserve.moe.combine` CUDA ms | 196.68 | 189.61 | -3.6% |

profile 表明 Gate/Up 融合已经减少了量化 GEMM 调用；dispatch/combine 仍基本未优化，
因此下一轮应围绕 token gather、Python expert loop 和 `index_add_` 设计候选，不能把
下一轮收益继续归因于 Marlin Gate/Up。

## Reproduction and provenance

原始数据保留在未纳入版本控制的
`experiment-data/2026-08-20_moe-gate-up-fusion-r1/`，其中 `summary.json` 由
`python -m benchmarks.moe_gate_up_summary` 从两份 raw JSON 生成；本目录不复制含本地
绝对路径的原始 request 记录。

- source commit: `578feb19a506aa3850227bd81b579104e3915d80`
- source worktree: dirty（实验时存在本地未提交修改）
- control SHA-256: `5df6fbd80d0d9b84da66c2fbeb0713e1e3dbbb3a52709a464e232e0c8f412f20`
- optimized SHA-256: `c21896e4db999e23a9b73134fea8d07852224c9c3be6d8f534f77f97eb1f9c36`
- summary SHA-256: `cbcb1a4e1b26c13bbe0d6137e09cd7c9b81e2fe44073e4e576490e8c47e5fc66`
- vLLM Marlin provider: `/root/hpf/workspace/LLM-Serving/.reference-vllm-0.9.1/vllm/_C.abi3.so`
- profile raw directory: `experiment-data/2026-08-20_moe-gate-up-profile-r1/`
- profile control JSON SHA-256: `c7faa108167f373e62b6d73d0aaa7b497ab20623b783ce71155b49bf646a4da2`
- profile optimized JSON SHA-256: `dfcbf13027976569c405cb3bdc5431788a6431e7545e3bf572832bd8b6dfb03b`
- profile control Chrome trace SHA-256: `53970842f8c3abccd9375ae2ddfe9c41806a6b0daafa1190e5b7e3055d89454b`
- profile optimized Chrome trace SHA-256: `37659b0f1e97b7e7b424aff4f412e560ec11e0721a99991d5b5443f04491806f`
