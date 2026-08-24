# PatchCore-lite MVTec AD 15 类三 seed 基线

> 初步全类别基线；不是正式上游 PatchCore，不是创新模型，也不是 GTX 2060 / 2500×2500
> 达标证据。

## 实验范围

- 数据：MVTec AD 直接归档，15 类。
- 协议：每类 3 seeds；目标支持集 100 normal + 30 seen anomaly，开发集选择固定阈值，
  final-test 图像与真值进程隔离。
- toothbrush 因数据不足使用 48 normal + 22 anomaly 支持，且只有一种缺陷类型；transistor
  使用 22 anomaly 支持。所有缩减均在训练前固定并记录。
- 任务：45/45 完成，失败 0。

## 宏平均真实结果

| 方法/切片 | AUROC | AP | 固定阈值 F1 | Accuracy |
|---|---:|---:|---:|---:|
| PatchCore-lite overall | 0.9284 | 0.9409 | 0.8691 | 0.8701 |
| PatchCore-lite seen | 0.9304 | 0.9322 | 0.8688 | 0.8880 |
| PatchCore-lite unseen | 0.9372 | 0.9123 | 0.8250 | 0.8796 |
| Linear head overall | 0.8810 | 0.9029 | 0.7856 | 0.7986 |
| Linear head unseen | 0.8426 | 0.7976 | 0.6934 | 0.8038 |

最弱类别为 screw（AUROC 0.6670，F1 0.6286，unseen F1 0.4177），随后是 capsule
（AUROC 0.7840）与 toothbrush（AUROC 0.8403）。leather、carpet 等纹理类明显更强。

## 证据

- 汇总：`reports/experiments/pc-lite-mvtec15-3seed-8gpu-20260823T152434Z-18420/aggregate.json`
- 逐任务：同目录 `runs/`，包含 45 份 metrics、模型、预测、时延和 registry。

