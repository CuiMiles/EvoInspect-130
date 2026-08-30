# EvoInspect-130 PRELIMINARY-SUBMISSION FREEZE

冻结时间：2026-08-27（北京时间）  
优先级：本文件记录的导师最终决定高于此前未冻结研究计划；官方附件和组织方书面答复仍为
最高权威。

## 1. 冻结结论

- AOI 项目继续，不换题。
- RCBR 正式失败，停止修改和扫参，不进入摘要或主创新；仅保留为研究决策负结果。
- HeteroMemory、GuardedFusion、MaskedPrototype、TriSynth 停止新增研发。
- PatchCore 固定为 Accuracy Engine 和正式强基线，不承担 GTX 2060 实时目标。
- EfficientAD-M 是当前唯一 Edge Engine 主候选；EfficientAD-S 只在 M 失败且具有速度/质量
  Pareto 价值时评估。
- GuardedAdapt 是当前唯一主要系统创新。
- DAGM、MVTec AD 2、LOCO、AnomalyDINO、AHL/DRA、RealNet、GLASS 和 CPU<2s 暂停，
  直到 EfficientAD、GTX2060、视频、GuardedAdapt 和提交闭环完成。

## 2. 当前机器真实状态

2026-08-28 00:25复核时，8 张 RTX 3090 均有其他用户计算进程和高显存占用。本轮没有启动
任何训练、没有终止或修改其他用户进程。`scripts/run_efficientad_frozen_8gpu.sh` 只选择无计算
进程、显存≤256 MiB、利用率≤5%的卡，并为每张卡取得 `/tmp/evoinspect-130-gpu-N.lock`；
任务启动前再次检查，失败即退出。

2026-08-28 17:51调度附记：用户明确授权在不OOM的前提下与其他GPU进程共存，并要求将45项
M实验墙钟压缩到12小时以内。启动器因此增加显式共享模式；默认独占行为不变。共享模式按
实时空闲显存门禁、限制单进程CUDA比例和CPU线程，绝不终止或修改其他用户进程。当前规划
使用GPU2--7共23个slot，排除显存余量不足的GPU0/1。12小时是资源调度目标，最终耗时仍须
由机器日志确认。

远程 GTX 2060 应立即并行接入，但仓库和当前 SSH 配置中没有主机、端口、用户名或认证方式，
因此尚不能安全连接。需要的最小信息为：主机/IP、SSH 端口、用户名、认证方式和远程工作目录。
取得后只对冻结 EfficientAD checkpoint 运行独立 2500×2500 基准，不使用 2060 结果调参。

## 3. EfficientAD 冻结协议

### 3.1 实现状态

- 新增 `configs/baselines/efficientad_m_100_30.yaml`；M/S 都固定 Anomalib v2.3.0、commit
  `091ca6aca92c8d0e416394f79e52f5a3cea3db73`、70,000 steps、256×256 和同一教师/Imagenette。
- 修复原训练函数硬编码 `model_size="small"`：现由冻结配置显式选择 medium/small。
- 新增独立 `efficientad_baseline_100_30.py`，不再经过 RCBR 路由。测试输入全部预测并写入
  `predictions.jsonl` 后才读取独立 `test_truth.csv`。
- 正式 M 批次固定 15 类×seeds 143–145=45 任务；聚合器要求 45/45、15 类完整且无失败。
- 质量门：Overall F1≥0.89、Unseen F1≥0.83、Image AUROC≥0.97、test-label leakage=0。

### 3.2 启动命令和时间估计

```bash
EVOINSPECT_DRY_RUN=1 scripts/run_efficientad_frozen_8gpu.sh m
scripts/run_efficientad_frozen_8gpu.sh m
```

既有 EfficientAD-S 70k 正式 smoke 的 12 个任务训练耗时为 17,546–18,318 秒/任务，中位数
18,169 秒（约5.05小时）。M 尚未实测，不把估算写成结果；若8卡全程独占且 M 为 S 的
1.2–2.0 倍，45任务训练墙钟保守约35–60小时，另加评估和等待空闲 GPU 的时间。当前 8 卡
均被占用，实际完成时间取决于安全空闲窗口。

M 通过质量门后直接进入部署，不要求超过 PatchCore；M 失败后保存完整证据，不无边界调参。

## 4. 统一推理与2500基准

`src/evoinspect/inference.py` 已实现统一 `InferenceEngine` 协议、Callable adapter 和显式
`SwitchableInferenceEngine`。Accuracy/Edge 输出统一包含异常分数、二值判定、置信度、缺陷
标签、区域/掩码、最近正常证据、模型版本和预处理/模型/后处理/序列化时延。不允许静默
fallback。

`scripts/benchmark_efficientad_latency.py` 固定 batch=1、输入2500×2500、warmup=100、
repeats=1000，分别报告 model-only 与 end-to-end p50/p95/p99/max、吞吐、CUDA/Torch/GPU、
checkpoint和输入哈希。当前没有冻结 EfficientAD checkpoint，也没有 GTX2060 实机报告，
因此 claim ledger 中不存在“满足2060 200ms”的声明。

连接前交接见 `docs/19_REMOTE_2060_HANDOFF.md`。硬件测量 checkpoint 已在训练前固定为
EfficientAD-M bottle/seed143；质量门不通过时打包器拒绝生成部署包。远端脚本拒绝繁忙 GPU、
默认拒绝非2060并保存设备、环境、模型、阈值和输入哈希。

## 5. 真实视频功能验证

原始素材为 `data/video/video_5/1.mp4` 至 `5.mp4`，均由 OpenCV 成功打开；五段均为
1080×1920、约30fps，总计2944帧、98.131秒。输入原件未修改。

实现采用 ArUco 优先、固定机位简单组件检测 fallback。经素材提供者复核，真实预期装配顺序
冻结为`cup -> bottle -> mouse`，再由 FSM 输出公开词汇
step_completed/skip/repeat/reorder/missing/unknown 和时间区间。原始文件和哈希未修改，派生视频
使用可读标题。基于v1.1冻结人工GT和±0.5秒、一对一最大二分匹配的最终结果：

| 视频 | 检测到的步骤/逻辑结果 |
|---|---|
| 1.mp4 | 正常：cup、bottle、mouse；事件F1=1.0 |
| 2.mp4 | 错序：bottle与cup顺序反转；事件F1=1.0 |
| 3.mp4 | cup、bottle后结尾缺mouse；事件F1=1.0 |
| 4.mp4 | cup后放bottle，移走并再次放入bottle；事件F1=1.0 |
| 5.mp4 | 先放并移走bottle，再从cup开始返工；事件F1=0.8 |

Micro Precision=Recall=F1=0.947368（18/19事件匹配）。唯一剩余错误在5.mp4：当前前端不输出
REMOVE动作，因而将返工后的cup记为reorder而非GT的step_completed。机器报告：
`evidence/video_event_evaluation_20260830.json`；GT：
`data/derived/video/desktop_assembly_gt_v1.1_frozen.json`。该结果是五段固定机位桌面功能验证，
GT在系统审查后由用户确认，不是盲法工业benchmark，不得声称跨场景泛化。unknown由单元测试
支持，但本次五段实拍没有unknown真值样例。

## 6. GuardedAdapt 真实分数反馈回放

配置冻结为 `configs/innovations/guarded_adapt_replay.yaml`。数据来自固定 PatchCore 15类、
seeds 133–137 的75个真实 MVTec 预测分数与真值，按标签和 sample-id 哈希确定性划分为反馈、
target evaluation、gate anchor、audit anchor。标签仅模拟操作员按顺序揭示；这是离线 replay，
不是生产用户研究。

为支持清洁目录复现，逐样本图像不进入仓库；仅导出 sample-id、冻结分数、标签、初始阈值和
原始文件哈希到 `evidence/guarded_adapt_replay_input.jsonl.gz`。该82,754字节 score pack 不含
图像，可在原75个run目录不存在时重放同一策略结果。

| 策略 | target gain均值 | anchor regression均值 | harmful-update rate | accepted-update rate |
|---|---:|---:|---:|---:|
| NoUpdate | 0.000000 | 0.000000 | 0.0000 | 0.0000 |
| NaiveUpdate | -0.013662 | -0.023685 | 0.0933 | 1.0000 |
| ThresholdUpdate | +0.013814 | -0.020042 | 0.0667 | 1.0000 |
| GuardedAdapt | +0.016299 | -0.021522 | 0.0267 | 0.8533 |

GuardedAdapt 对11个拒绝更新执行精确恢复，rollback success=11/11=1.0；五个在独立 audit
anchor 上表现为有害的 bounded candidate 中，门禁阻断3个（0.60）。适应逻辑 CPU p50/p95/
p99=0.477/0.931/0.937ms。小切片离散性较强，仍有2/75有害更新未被 gate anchor 预见，
因此不能写成“完全阻断”或“准确率提高”；核心证据只是相同 replay 下有害更新率下降、历史
能力门禁和回滚链路可执行。

机器报告：`reports/experiments/guarded-adapt-replay-20260827T194500-cpu/report.json`。

## 7. 四件提交物

`submission/drafts/` 已生成四件可审查草稿，机器约束检查全部通过：

| 提交物 | 当前草稿 | 机器约束 |
|---|---|---|
| 作品简介PDF | `works_intro.pdf` | 1页；中文169字、非空白268字符，均≤300 |
| 官方结构项目PDF | `project_document.pdf` | 5页、A4、包含官方模板各章节 |
| 项目MP4 | `project_video.mp4` | 122.133秒、1280×720、25.7MB，≤5min且≤200MB |
| 辅助ZIP | `auxiliary_material.zip` | 144项、小于0.5MB、完整性通过且≤200MB |

验证：`evidence/submission_artifact_validation.json`。团队Cuisine、队长/唯一成员崔明浩、
西安交通大学和单人分工已填写；参赛组别仍待确认，官方文件名仍是占位；项目PDF还必须在 EfficientAD-M/
2060 结果返回后删改待验证内容并重新审校。

## 8. 当前允许和禁止声明

允许：PatchCore 已验证公开数据指标；RCBR 是正式负结果；真实视频功能链路可运行；在所测
离线 replay 上 GuardedAdapt 的有害更新率和回滚指标；提交草稿满足格式约束。

禁止：EfficientAD-M 已通过门槛；Edge Engine 已冻结；GTX2060<200ms；CPU<2s；真实视频
工业准确率；GuardedAdapt 提高生产准确率或阻断所有有害更新；四件材料已经可直接上传；
首次、首创、SOTA、国际领先、全面超越或奖项保证。

## 9. 剩余关键路径

1. 等待安全空闲 GPU，运行 EfficientAD-M 45任务并执行冻结质量门；若通过立即部署。
2. 同时取得远程 GTX2060连接参数；M通过后运行独立2500基准并更新 claim ledger。
3. 用户提供团队元数据，替换提交物占位和官方文件名。
4. 新目录/新环境CPU/静态复现已完成；证据为
   `evidence/clean_reproduction_20260828.json`。训练和2060仍因外部资源分别记录。

唯一主动作：**在不抢占他人 GPU 的前提下启动并完成 EfficientAD-M 冻结质量门。**

可并行：获取2060连接参数；填写团队元数据；人工审阅5分钟内视频和项目PDF；许可证签核。
