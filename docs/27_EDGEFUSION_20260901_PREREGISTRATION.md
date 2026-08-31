# EfficientAD-S+M EdgeFusion 预注册协议

冻结时间：2026-09-01（运行融合结果前）

## 目的与边界

本实验只检验已经完成的 EfficientAD-S seed 143 与 EfficientAD-M seed 143 的固定权重融合，
不重训主干、不读取测试真值选择权重、不选择 checkpoint，也不修改 `submission/final/`。
它是一次有边界的探索筛选；只有同时通过质量门和独立 GTX2060 速度复测，才允许作为候选
Edge Engine。否则只记录负结果。

## 固定方法

对每个类别，将两模型的测试异常图分别除以各自由支持集确定的冻结阈值，固定使用
`alpha_M=0.5`、`alpha_S=0.5`，逐像素加权后以融合图 `amax` 作为图像分数，固定融合阈值为
`1.0`。所有权重和规则在打开 `test_truth.csv` 前写入配置并保持不变。

## 范围与门槛

使用已完成的 15 类、seed 143 strict-v2.1 输出，共 15 次；要求 15/15 完整、零测试标签
泄漏、Overall F1≥0.905、eligible Unseen F1≥0.84、Image AUROC≥0.97。若通过，才允许在
已固定的 GTX2060、2500×2500、batch=1 测试中检查端到端 p95<200ms；失败不扩展 seed、不扫参。

机器入口：`scripts/evaluate_efficientad_edgefusion.py`。
