# EvoInspect-130 最终路线紧急决策记录

更新时间：2026-08-29 17:27（北京时间）  
提交截止：2026-09-01 23:59（仓库本地官方文件记录）  
剩余时间：约 78.5 小时  
决策状态：立即执行，不等待新增研究方向

## 一、结论先行

当前作品已经具备强基线、真实视频功能链路、GuardedAdapt 系统创新、四件提交草稿和清洁
环境复现，但**尚未达到可以诚实宣称“已具备获奖水平”或“可以直接提交”的状态**。最大
风险不是材料数量，而是 EfficientAD-M 正式质量证据尚未成立、当前 evaluator 与冻结的
100+30 协议及上游图像评分不一致，以及真实 GTX 2060 时延仍为空。

时间已经不允许恢复 RCBR、增加新模型族、扩展数据集或无边界扫参。最终路线固定为：

> **保留并完成当前 EfficientAD-M checkpoint；立即冻结 strict-100+30 evaluator v2；按上游
> `amax` 图像分数和真实 30 张缺陷支持集重新评测已有权重。M 通过则直接部署和实测；M
> 失败则正式冻结为负结果，只允许一次 EfficientAD-S 单 seed、15 类的有界 Pareto 筛查。
> 最终主叙事始终是 PatchCore Accuracy Engine + 合格的 Edge Engine（若有）+
> GuardedAdapt + 真实视频功能验证。**

这是当前预期价值最高、最符合导师冻结决定且最可能在截止前闭环的唯一路线。

## 二、截至本文件时间的机器真实状态

批次：
`reports/experiments/efficientad-m-frozen-20260828T095200Z-shared23`

- 计划范围：MVTec AD 15 类 × seeds 143--145，共 45 个 EfficientAD-M 任务；
- 已产生 `metrics.json`：26/45；
- 当前 `failure.json`：0；
- 当前覆盖：9/15 类；
- 活跃 CUDA worker：10；
- 当前使用 GPU 0--3；四卡利用率 97%--100%；
- 当前空闲显存约 8.7--11.3 GiB；
- GPU 温度约 71--84 摄氏度；GPU 2 最高，需持续监控；
- 其他用户进程未被终止或修改；共享 GPU 状态可能改变剩余墙钟时间。

26 个阶段性文件的简单宏平均如下。它们是**诊断值，不是正式 100+30 结果**：

| 指标 | 当前样本数 | 阶段均值 | 冻结最低线 | 当前解释 |
|---|---:|---:|---:|---|
| Overall F1 | 26 | 0.872878 | 0.89 | 低于门槛，但剩余任务仍可改变均值 |
| Unseen F1 | 24 | 0.779026 | 0.83 | 低于门槛；2 个 toothbrush run 为 N/A |
| Image AUROC | 26 | 0.945544 | 0.97 | 当前评分口径下已经不可能通过 45-run 门槛 |

在当前评分口径下，即使剩余 19 个 run 的 Image AUROC 全部为理论最高值 1.0，最终 45-run
均值最高也只有 0.968537，仍低于 0.97。因此，**当前 scorer 对应的 M 路线已在数学上无法
通过冻结质量门**。这不等同于上游一致、严格 100+30 的 M 已经失败，因为当前 evaluator
存在下述已定位偏差。

当前最弱的已完成任务集中在 cable 和 capsule。最低 Image AUROC 为 cable seed143 的
0.824200；cable 三 seed 的 Unseen F1 约为 0、0.095238 和 0.086957。这说明即使修复 scorer，
类别鲁棒性仍是高风险项，不能预设修复后一定通过。

## 三、为什么当前结果不能进入正式 claim ledger

### 3.1 30 张缺陷支持样本未被 baseline runner 使用

每个 `adaptation.csv` 已包含 `support_anomaly`，例如 bottle seed143 的清单包含 30 张缺陷
支持图。但 `scripts/efficientad_baseline_100_30.py` 当前只读取 `support_normal` 和
`development`，没有读取 `support_anomaly`。因此：

- 80 张正常支持图进入训练；
- 20 张正常支持图进入正常校准；
- 30 张正式缺陷支持图没有用于阈值适配；
- 阈值反而由额外的 development 正常/异常切片确定。

这不满足导师冻结的“完全相同 MVTec AD 数据隔离与 100+30 evaluator”。当前 F1 只能作为
工程诊断，不能作为正式质量门或摘要数字。

### 3.2 图像级分数偏离固定上游 EfficientAD

固定 Anomalib v2.3.0 上游实现为：

```python
pred_score = torch.amax(anomaly_map, dim=(-2, -1))
```

当前项目 runner 则使用：

```python
np.quantile(anomaly_map, 0.995)
```

99.5% 分位数可能弱化面积很小但峰值明显的缺陷，尤其影响 cable 等类别的图像排序。恢复
上游 `amax` 是实现一致性修复，不是根据测试标签扫参。修复必须在查看重评结果前写入配置、
测试和 Git commit，并对所有类别/seed 一致应用。

### 3.3 单缺陷类型类别的 Unseen F1 未定义

toothbrush 只有一个缺陷类型，当前 seed143、seed144 的 `result.unseen` 为 `null`。现有
聚合器直接索引 `run["result"]["unseen"]["f1_fixed_threshold"]`，完整批次结束后会报错，
而不是生成合法 gate。

v2 必须预先固定：

- 15 类全部参加 Overall F1 和 Image AUROC；
- 只有至少两种缺陷类型且可形成严格类型外推的类别参加 Unseen F1；
- 单类型类别记录 `unseen_not_applicable`，不得填 0、不得伪造；
- gate 同时报告 eligible 类别/run 数和覆盖率；
- 45/45、15 类完整、零 failure、零 test-label leakage 仍是硬条件。

## 四、为什么优先重评已有权重，而不是立刻重训或换模型

代码审计显示 EfficientAD 优化损失只使用正常训练图；map normalization quantile 也在上游
代码中显式过滤 `label == 0`。当前训练没有早停，并在训练结束后显式保存最终 checkpoint。
因此现有权重**有较大概率可以在完成影响审计后直接用于 strict-100+30 v2 重评**：

1. 载入固定 checkpoint；
2. 用 20 张正常校准图和 30 张缺陷支持图计算固定阈值；
3. 使用上游 `amax` 图像分数；
4. 在打开测试真值前保存所有测试预测；
5. 再读取隔离真值并计算指标；
6. 生成新 run id、配置哈希、模型哈希和独立 evidence。

在完成代码级影响审计前，不把“无需重训”写成既成事实。如果验证发现 development 异常标签
影响了优化、模型选择或最终模型状态，则必须重训；旧权重只保留为诊断证据。

## 五、时间预算与硬停止点

当前估计不是承诺，按共享 GPU 波动保留余量：

| 工作 | 预计墙钟 | 是否并行 |
|---|---:|---|
| 当前 M checkpoint 补齐到 45/45 | 6--9 小时 | GPU 持续运行 |
| strict-100+30 evaluator v2、单测和冻结清单 | 2--4 小时 | 与训练并行 |
| 45 checkpoint 修正重评 | 1--3 小时 | checkpoint 齐全后运行 |
| 聚合、质量门和 evidence 更新 | 1 小时 | 重评后 |
| M 通过后的 2060 环境/2500 benchmark | 2--4 小时 | 需远端连接 |
| 最终 PDF/MP4/ZIP 替换、审校和复现 | 3--6 小时 | 结果确定后 |

预期 8--14 小时内得到修正后的 M 决策，保守不超过 18 小时。M 通过且 2060 立即可用时，
目标是在 24 小时内完成技术闭环，至少保留约两天用于材料审校和上传。

硬停止规则：

- evaluator v2 不允许任何测试标签调阈值、选 scorer、选 checkpoint 或决定早停；
- M 修正重评失败后，不修改 M 学习率、分辨率、训练步数、seed 或阈值策略；
- 不恢复 RCBR、HeteroMemory、GuardedFusion、MaskedPrototype、TriSynth；
- 不启动 DAGM、AD2、LOCO、AnomalyDINO、AHL/DRA、RealNet、GLASS 或 CPU 挑战；
- 提交闭环与新增实验冲突时，提交闭环优先。

## 六、最终自动决策树

### A. 修正后的 EfficientAD-M 通过全部冻结门槛

要求同时满足：Overall F1 >= 0.89、eligible Unseen F1 >= 0.83、Image AUROC >= 0.97、
15 类与 45 run 完整、零 failure、零 test-label leakage。

动作：立即冻结 M 为 Edge Engine；使用预声明的 bottle seed143 checkpoint 构建 bundle；连接
真实 GTX 2060；只运行固定 2500×2500、batch=1、warmup=100、repeats=1000 的 model-only
和 end-to-end p50/p95/p99 benchmark；结果不得用于再调模型。

### B. M 质量门失败

动作：保存完整负结果并停止 M。只允许一次 EfficientAD-S 的 15 类、seed143、相同 evaluator
v2 筛查，不扫参。只有当该单 seed 筛查通过相同质量门，且固定代表样本显示明确速度优势时，
才扩展 seeds144--145；否则立即停止 S，不消耗剩余提交时间。

### C. M/S 都不能形成合格 Edge Engine

动作：不伪造 Edge 结论。最终系统保留 PatchCore Accuracy Engine，EfficientAD 作为受控部署
候选负结果，主创新固定为 GuardedAdapt，视频明确为桌面功能验证。材料中删除“满足 GTX2060
实时目标”和“Edge Engine 已冻结”，诚实报告未闭环项。该路线可以形成有效提交，但获奖上限
和专家说服力会明显降低。

## 七、当前获奖竞争力判断

### 已经形成的优势

- 固定 upstream PatchCore 15 类五 seed：Overall F1 0.9224、Image AUROC 0.9817；
- GuardedAdapt 75-run 离线反馈 replay：有害更新率 0.0267，低于 NaiveUpdate 的 0.0933，
  11/11 拒绝更新回滚成功；
- OpenCV 实拍视频 5/5、2944/2944 帧完成，FSM 输出事件区间；
- 四件提交草稿已通过格式/时长/大小机器约束；
- 清洁目录和新环境复现通过；
- 负结果、哈希、日志和 claim 边界相对完整，研究真实性较强。

### 当前不容乐观之处

- EfficientAD-M 正式 100+30 质量结果不存在；当前阶段 scorer 下 AUROC gate 已不可能通过；
- cable 等类别表现很弱，上游 scorer 修复不保证把均值提升到门槛；
- 尚无真实 GTX 2060 结果，官方 200ms 目标不能声明；
- 主要视觉算法研发没有稳定正向创新，GuardedAdapt 是唯一主要创新；
- 视频没有逐帧人工真值，只能作为功能验证；
- 参赛组别、正式文件名、许可证和若干官方接口口径仍未闭环；
- 截止前约 78.5 小时，已经没有重新探索多个模型族的时间。

综合判断：**当前尚未达到高把握获奖状态；完成 evaluator v2、合格 Edge/2060 证据和最终材料
闭环后，可以达到“具备获奖竞争力”的诚实表述，但任何奖项都不能保证。** 如果 Edge 和
2060 均失败，作品仍可提交，但必须依靠 PatchCore、GuardedAdapt、系统完整度和表达质量，
获奖风险显著增加。

## 八、从现在起的唯一主动作

**在不中断当前 M checkpoint 生产的同时，立即实现并冻结 strict-100+30 evaluator v2；
checkpoint 齐全后只重评一次并按上述决策树自动收敛。**

可并行但不得阻塞主动作：准备 2060 SSH 信息；确认参赛组别；人工审阅 PDF/MP4；完成许可证
签核和正式文件名。除此之外不再启动新研究任务。

## 九、冻结执行结果更新（2026-08-30）

EfficientAD-M已完成45/45 strict-v2.1重评：Overall F1=0.903604通过0.89门槛，eligible
Unseen F1=0.820986低于0.83，Image AUROC=0.956915低于0.97；15类完整、0 failure、0测试
标签泄漏，总门`passed=false`。因此进入本文件决策树B：M正式冻结为负结果，不再调参；只运行
一次EfficientAD-S的15类seed143相同协议筛查。

机器证据：
`reports/experiments/efficientad-m-frozen-20260828T095200Z-shared23/strict-quality-gate-v2.json`。

## 十、EfficientAD-S单seed停止结论（2026-08-30）

EfficientAD-S已按冻结决策完成15类seed143唯一一次筛查和15/15 strict-v2.1重评：Overall
F1=0.890785通过0.89门槛，eligible Unseen F1=0.798847低于0.83，Image AUROC=0.963851
低于0.97；15类完整、0 failure、0测试标签泄漏，总门`passed=false`。S在RTX 3090、
256x256模型段诊断中比M更快，但该诊断不是GTX2060或2500x2500端到端证据，不能抵消质量
失败。按预注册决策，S停止，不扩展seed、不调参，也不尝试第三种检测模型。

因此，2060与视频GT前的冻结算法任务已经全部结束，但结论不容乐观：PatchCore仍是唯一通过
现有精度证据的Accuracy Engine；没有通过冻结质量门的Edge Engine；GuardedAdapt-Risk也未
通过正向创新门。项目可继续完成2060、视频GT和提交闭环，但材料必须把M、S和Risk写为负结果，
不得宣称已有合格实时引擎或国奖保障。

机器证据：
`reports/experiments/efficientad-s-frozen-20260830T004009Z-seed143-gpu0-3/strict-quality-gate-v2.json`。
