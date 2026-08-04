# Public Benchmark Evidence

本目录只包含新 serving benchmark 系统生成的正式数据，旧 Stage 4/5 脚本与结果已删除。

## 数据集

- `formal-poisson/`：36 个 run，Poisson request-rate 主实验。
- `formal-closed-loop/`：36 个 run，固定并发稳态补充实验。
- `stage8-graph-formal/`：18 个 run，线性 EAGLE target verify eager/ CUDA Graph 对照。
- `pd-kv-pipeline-formal/`：18 个精简脱敏 run，双卡 PD inline Queue / pinned shared-memory KV transport 对照。
- `pd-serving-formal/`：48 个脱敏 run，单进程 collocated 与双卡 PD Prefill/Decode 端到端对照。
- `awq-w4a16/`：AWQ 质量、LLM-Serve 容量矩阵和 vLLM Marlin 控制实验的脱敏汇总。

每个 serving 目录包含 `manifest.json`、`summary.csv`、`aggregate.csv` 和 `runs/*.json`。原有 72 个 run 对应 commit `ad35e65cacdcb306362268c3a60923abd199b431`；Stage 8 的 18 个 run 对应 commit `3bb5d21ad5fd9ae0044943d93255a4542cc5ca75`；PD KV Pipeline 的 18 个 run 对应 commit `e779a6aa4186683327060697b8c04f3da12c0284`；PD 端到端矩阵的 48 个 run 对应 commit `ae740739d25f7184c04208f35dcf1b720e624f2a`。模型 revision 和软硬件环境见各自 manifest。

公开文件已经扫描，不包含本地绝对路径、prompt token IDs、凭据、traceback 或 host-specific workspace 信息。PD 精简 run 删除重复的逐请求记录和逐 Prefill batch 明细，保留聚合指标、Queue/slot 样本、Graph 和 worker health；`summary.csv` 与 `aggregate.csv` 仍为 suite runner 的原始输出。

AWQ 目录只公开汇总 CSV 和 metadata；checkpoint、校准文本、逐层 cache 与原始日志保留在本地。LLM-Serve 自研 CUDA 与 vLLM Marlin 的归因边界见该目录 README。

CPU 回归、编译检查和结果完整性校验见 [`verification.md`](verification.md)。
