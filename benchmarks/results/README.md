# Public Benchmark Evidence

本目录仅保留仍与当前 Runtime 边界一致的脱敏证据：

- `stage8-graph-formal/`：EAGLE Target Verify CUDA Graph 对照。
- `service-overload-governance/`：固定 Poisson trace 下无界入队与 64-request admission
  bound 的单机服务治理对照，只报告输出/请求吞吐、TTFT、TPOT、队列和 admission outcomes。

早期非 Chunked 单卡基线、Inline Queue 对照、双 Collocated 实验运行时和旧的有限
请求 Poisson 数据均已移除。它们不能代表当前强 Collocated 基线或 Shared KV 默认路径。

后续 PD 结论需要单独评审资源等价 suite；PD+Shared 相对单卡的结果属于完整部署收益，
不能归因为纯 PD 或纯 KV 传输收益。
