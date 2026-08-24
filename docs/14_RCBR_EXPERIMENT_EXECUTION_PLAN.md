# RCBR 统一实验执行与判定说明

更新时间：2026-08-24

## 1. 已冻结的方法决定

导师回复解除实验暂停，停止 HeteroMemory、GuardedFusion 和 MaskedPrototype 的继续修补。
当前唯一算法主线为 **Risk-Calibrated Budgeted Refinement（RCBR）**：

1. EfficientAD-S 以 80 张支持正常样本训练；其余最多 20 张支持正常样本只用于空间风险校准；
2. 30 张支持缺陷不微调主干，只通过五折 cross-fitting 学习 ROI 复检收益；
3. 候选同时来自校准风险、多尺度不一致、高频和位置稀有度，避免只依赖粗热图；
4. 使用实测 ROI 模型时延、最多 4 个 ROI、时延预算和面积上限进行确定性路由；
5. 全图与 ROI 共享同一个 EfficientAD-S；局部证据不足或输出异常时显式回退；
6. PatchCore commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` 是固定强基线，
   不进入最终实时图。

Anomalib 固定为官方仓库 tag `v2.3.0`、commit
`091ca6aca92c8d0e416394f79e52f5a3cea3db73`、Apache-2.0。运行器拒绝改变六种
受控对照的集合或顺序。

## 2. 一次训练覆盖六种对照

每个“类别 × seed”只训练一个 EfficientAD-S，随后复用同一权重、同一全局热图和已计算的
局部热图，评估：

1. `uniform_downsample`：统一低分辨率；
2. `full_grid`：4×4 全网格高像素密度复检；
3. `fixed_topk`：固定 Top-4 原始热图 ROI；
4. `uncertainty_only`：只按多尺度不一致选择；
5. `risk_calibrated`：正常风险校准路由；
6. `full_rcbr`：风险、收益、成本、硬预算和安全融合的完整方法。

这样开发阶段为 15 类 × 3 seeds = 45 次训练，不会把六种对照扩大成 270 次训练。

## 3. 数据隔离和阶段门

- 开发：seeds 130–132；
- 历史 seeds 133–137：只保留为既有诊断，不参与选择；PatchCore 匹配参考来自既有
  seeds 130–132 保存结果的 CPU 重评；
- 确认：seeds 138–142，仅在开发通过、人工复核并生成 freeze manifest 后显式解锁一次；
- 每个任务先生成进程隔离的 `adaptation.csv`、`test_inputs.csv` 和 `test_truth.csv`；
- 运行器在所有预测、ROI 和阈值固定后才读取 test truth；
- 阈值只由 development 正常/异常样本选择；
- 测试标签不用于训练、路由收益、阈值、早停或回退选择。

开发按不重复训练的三级执行：四类 seed-130 功能门；补齐四类 seeds 131–132 后 smoke
门；通过后只补其余 11 类 × 3 seeds。任何门失败均非零退出，不自动扫参。

## 4. 评测与门槛

固定面积定义：Tiny ≤0.1%，Small 为 0.1%–1%，Large >1%；原 q25/q75 只作补充。
主要比较为同类别、同 seeds 的 RCBR 对固定 PatchCore：

- 四类 smoke：平均 ΔAUPRO@0.05 ≥0.025；至少 3/4 类不下降；最差类 ≥-0.015；
  ΔOverall F1 ≥-0.005；ΔUnseen F1 ≥-0.010；
- 全 15 类：ΔAUPRO@0.05 ≥0.015；ΔPRO@1% FPR ≥0.020；
  Δ固定 Small AUPRO@0.05 ≥0.010；ΔOverall F1 ≥-0.003；
  ΔUnseen F1 ≥-0.005；ΔImage AUROC ≥-0.002；平均 ROI 面积 ≤15%，p95 ≤25%；
- 输出类别配对 bootstrap 95% CI；CI 跨 0 时不得写“显著提升”。

## 5. GPU 安全

启动器不会要求八卡必须全空闲。它逐卡检查 compute process、显存和利用率，只使用当前
安全空闲卡；繁忙卡会显示 `SKIP BUSY GPU`，不会终止、暂停或修改任何其他进程。每张卡还
使用 `/tmp/evoinspect-130-gpu-N.lock` 协作锁，并在每个任务前重新检查。小模型以独立任务
并行，不使用无意义 DDP。

## 6. 用户执行命令

首次安装隔离环境（不修改现有项目或 PatchCore 环境）：

```bash
cd /home/CuiMinghao/projects/AOI2026/EvoInspect_130_Research_Kit
bash scripts/setup_efficientad_env.sh
```

先检查将运行什么，不训练：

```bash
EVOINSPECT_DRY_RUN=1 bash scripts/run_rcbr_experiment_suite.sh development
```

完整开发实验：

```bash
bash scripts/run_rcbr_experiment_suite.sh development 2>&1 | tee logs/rcbr-development.log
```

开发结果应先交回分析。不得在同一次命令中自动运行封存确认。开发通过并人工冻结后，确认
命令才是：

```bash
EVOINSPECT_ALLOW_CONFIRMATION=1 \
EVOINSPECT_FROZEN_MANIFEST=/absolute/path/to/frozen-rcbr.json \
bash scripts/run_rcbr_experiment_suite.sh confirmation 2>&1 | tee logs/rcbr-confirmation.log
```

## 7. 预期输出

每个任务保存 checkpoint、六种方法 mask、结构化指标、阈值、拆分/配置/模型哈希、五折
收益诊断和逐图 ROI 审计。批次保存 smoke/full gate。全开发门通过后，还在一张空闲 3090
上进行 batch=1、100 warmup、1000 次的 2500×2500 合成分辨率端到端时延测试，分开报告
解码、预处理/传输、全局模型、路由、局部模型、后处理和序列化。该结果只允许表述为
“3090 合成分辨率实测”，不得替代原生高分辨率精度或 GTX 2060 实测。

## 8. 尚未由本代码包解决的任务

本轮没有伪装以下未完成项：AHL/DRA 少监督开放集基线、MVTec AD 2、MVTec LOCO 视频逻辑、
反馈/影子发布/回滚产品闭环、GTX 2060 与 CPU 实机性能、最终说明书/视频/压缩包。RCBR 开发
结果返回后，下一轮应根据门控结果决定冻结或执行唯一一次机制级修订，同时并行补这些系统
交付，不得把它们写成已经完成。
