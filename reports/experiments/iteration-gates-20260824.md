# GuardedFusion 前置迭代门控与负结果

本文件固化 v2-safe 之前的真实淘汰证据；这些结果用于方法开发，不计入最终 seeds 133–137。

| 路线 | 范围 | overall ΔF1 | unseen ΔF1 | 决策 |
|---|---|---:|---:|---|
| HeteroMemory v1 | 8 类 × seed 130 | -0.0079 | -0.0216 | 淘汰 |
| HeteroMemory v2 guarded | 8 类 × seed 130 | +0.0151 | -0.0308 | 淘汰 |
| GuardedFusion v1 heldout | 7 类 × seeds 130–132 | -0.0046 | -0.0053 | 淘汰 |
| layer2 PatchCore-lite | 8 类 × seed 130 | -0.0616 | -0.1077 | 淘汰 |

HeteroMemory v2 虽提高固定阈值总体 F1，但总体 AUROC 从 0.8658 降到 0.8284，且 unseen
F1 从 0.7533 降到 0.7225，不能保留。GuardedFusion v1 在 21 个留出任务中 20 个不变，
但 transistor seed 130 被仅 4 个开发异常误导而退化，促成 v2-safe 的
`min_development_anomalies: 6`。layer2 只有 screw 改善，cable 和 pill 的开发选择方向与
final-test 相反，因此不建立 ScaleRouter。

证据目录：

- `reports/experiments/heteromemory-v1-pilot-8gpu-20260823T154056Z-8800/`
- `reports/experiments/heteromemory-v2-dev-8gpu-20260823T154634Z-14240/`
- `reports/experiments/guarded-fusion-v1-heldout7-3seed-8gpu-20260823T155420Z-22970/`
- `reports/experiments/pc-lite-layer2-dev-8gpu-20260823T160049Z-575/`

