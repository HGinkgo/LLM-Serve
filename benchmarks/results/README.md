# Public Benchmark Evidence

本目录只包含新 serving benchmark 系统生成的正式数据，旧 Stage 4/5 脚本与结果已删除。

## 数据集

- `formal-poisson/`：36 个 run，Poisson request-rate 主实验。
- `formal-closed-loop/`：36 个 run，固定并发稳态补充实验。

每个目录包含 `manifest.json`、`summary.csv`、`aggregate.csv` 和 `runs/*.json`。72 个 run 均为 `complete=true`，对应 commit `ad35e65cacdcb306362268c3a60923abd199b431`，模型 revision 和软硬件环境见各自 manifest。

公开文件已经扫描，不包含本地绝对路径、prompt token、凭据、traceback 或 host-specific workspace 信息。`summary.csv` 与 `aggregate.csv` 由 suite runner 直接从 run JSON 生成，不经过手工改写。

CPU 回归、编译检查和结果完整性校验见 [`verification.md`](verification.md)。
