# Public Benchmark Evidence

本目录仅保留仍与当前 Runtime 边界一致的脱敏证据：

- `awq-w4a16/`：AWQ checkpoint 的质量、显存容量与外部 Marlin 对照。
- `stage8-graph-formal/`：EAGLE Target Verify CUDA Graph 对照。

早期非 Chunked 单卡基线、Inline Queue 对照和旧的有限请求 Poisson 数据均已移除。
它们不能代表当前强 Collocated 基线或 Shared KV 默认路径。

后续 PD 结论必须使用 `pd-resource-equivalent-formal.json` 或
`pd-phase-map-smoke.json` 在报告硬件上重跑。manifest、展开 point 配置、逐运行
JSON 和 CSV 共同构成可追溯证据。PD+Shared 相对单卡的结果属于完整部署收益，不能
归因为纯 PD 或纯 KV 传输收益。
