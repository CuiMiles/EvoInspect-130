# PatchCore-lite MVTec AD bottle 直接归档三 seed 报告

> 单类别初步基线；不是完整 MVTec AD 均值，不是官方 PatchCore 复现，也不是部署达标证据。

## 数据验收

- 直接归档：5,264,982,680 bytes，SHA-256
  `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`。
- 归档安全检查：6,880 个成员，无路径穿越，无软/硬链接；解包后所有原始文件只读。
- 全量 manifest：15 类、5,354 张输入图像，4,096 normal、1,258 anomaly，内容重复 0；
  manifest SHA-256 `6069c761f5ad36c25f9cbd461d11eb1805686a3020a8bd9f494e25570b9cbcd9`。
- Bottle manifest：292 张，SHA-256
  `a9c44956f7aa9645904c2246410f5df18f8c0edc37fdabdc8445585b502b1571`。
- Bottle 与固定 Voxel51 mirror commit 的内容哈希集合 292/292 完全一致。该交叉检查增强了
  数据身份可信度，但直接归档来源与许可证资格仍保留人工复核门禁。

## 协议

每个 seed 使用 100 normal + 30 seen anomaly 支持集；20 normal + 6 seen anomaly 仅用于
development 阈值选择；最终测试包含 20 normal、6 seen anomaly、21 unseen contamination，
共 47 张。支持集与测试 ID 不交叉，推理输入无标签，评测真值无图像路径。

配置：`configs/baselines/patchcore_lite_bottle.yaml`；配置文件 SHA-256
`3bef46cd58355150aba7baf5ceb9e0dee0bd81b5ba7b3cb1e71c15d753654dd4`，规范化配置哈希
`416de329c2f738f070eea47f48b524592dd1e4c1e2abbc1280fb7ec87e0769bd`。

## 真实指标

| 方法/切片 | AUROC mean±sd | AP mean±sd | 固定阈值 F1 mean±sd | Accuracy mean±sd |
|---|---:|---:|---:|---:|
| PatchCore-lite overall | 0.9988±0.0021 | 0.9991±0.0016 | 0.9197±0.0216 | 0.9149±0.0213 |
| PatchCore-lite seen | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 |
| PatchCore-lite unseen | 0.9984±0.0027 | 0.9985±0.0026 | 0.8942±0.0291 | 0.9024±0.0244 |
| Linear head overall | 0.8605±0.0222 | 0.9352±0.0112 | 0.8170±0.0295 | 0.8227±0.0246 |
| Linear head unseen | 0.8206±0.0286 | 0.9026±0.0172 | 0.7640±0.0364 | 0.8049±0.0244 |

PatchCore-lite 三 seed overall 固定阈值 F1 为 0.9200、0.9412、0.8980；分别漏检 4、3、5
张，假阳性均为 0。高 AUROC 仍不能替代固定开发阈值下的漏检分析。

## 资源与时延

- 仅使用物理 GPU 1（RTX 3090）；三个 seed 顺序运行，每个阶段前检查无其他计算进程。
- 适配耗时 `7.93±0.37 s`；峰值 PyTorch 显存 477,419,008 bytes（约 455.3 MiB）。
- RTX 3090、FP32、batch=1、224×224、预热 10 次、不含模型加载和输出文件 I/O：端到端
  p50 三 seed 均值 `30.51 ms`，p95 均值 `67.36 ms`，最大观测 `225.26 ms`。
- 单 seed 适配产物 8,452,626 bytes，另依赖 275,905,729-byte backbone 权重。

这不是 2500×2500 或 GTX 2060 实测，不能据此声称满足官方 200 ms 参考目标。

## 产物

- 汇总：`reports/experiments/pc-lite-bottle-official-3seed-20260823T144354Z.json`
- 分 seed：`reports/experiments/pc-lite-bottle-official-s{130,131,132}-20260823T144354Z/`
- 数据核验：`evidence/mvtec_ad_archive_verification_20260823.json`
- Git commit 仍为 `UNAVAILABLE`，由配置、数据、划分、权重和模型哈希补充审计链。
