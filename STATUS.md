# STATUS

updated_at: 2026-08-28T20:18:00+08:00
current_phase: PRELIMINARY_SUBMISSION_FREEZE
overall_status: EFFICIENTAD_M_RUNNING_GPU2_3_CPU_RELIEF_SUBMISSION_METADATA_PARTIAL

## One-sentence truth

RCBR 已冻结为正式负结果；PatchCore 固定为 Accuracy Engine，EfficientAD-M当前仅保留
GPU2/3上的11个worker，GPU4--7上我们的12个worker已按用户要求停止以解除CPU瓶颈；真实视频5/5与 GuardedAdapt
75-run反馈回放已完成，四件初赛草稿的格式/大小约束均通过；Cuisine、崔明浩、西安交通大学
和单人分工已写入，参赛组别仍待确认，M质量门、真实
GTX2060时延仍阻止正式上传；清洁目录/环境CPU与静态复现已经通过。

## Freeze completion snapshot

- 导师冻结决定已固化：`docs/18_PRELIMINARY_SUBMISSION_FREEZE.md`。
- RCBR 不再修改/扫参/进入摘要或主创新；HeteroMemory、GuardedFusion、MaskedPrototype、
  TriSynth 停止；其余扩展数据集/模型暂停。
- 新增统一 Accuracy/Edge `InferenceEngine`，单元测试通过。
- EfficientAD-M/S 独立100+30 runner、聚合 gate、2500 p50/p95/p99 benchmark和安全8卡脚本
  已完成；M正式范围为15类×seeds143–145=45任务。
- 2026-08-28 17:05以commit `cc555b4`启动的批次在首任务阶段中断并保留诊断证据：先发现
  后台会话回收，再发现启动器未显式传播单任务退出码且PTY中断未清理后台worker。0个正式
  metrics产生，不能计入算法结果。现已修复退出码传播、worker fail-fast和信号清理并通过
  受控失败测试。
- 修复commit `5ee4b9e`的双worker批次
  `efficientad-m-frozen-20260828T091500Z-gpu2-3`运行约32分钟后由用户授权中止并改为吞吐优先
  调度；两个任务均只产生中间checkpoint、0个正式metrics，不计入算法结论。
- 共享GPU加速启动器保留默认独占模式，新增显式`EVOINSPECT_ALLOW_SHARED_GPU=1`和重复GPU
  slot；按启动前显存余量门禁，不终止其他用户进程。计划23 workers：GPU2×6、GPU3×5、
  GPU4--7各×3；GPU0/1因仅余约3.1GiB/89MiB而排除。每进程CUDA上限0.12、CPU线程1、
  DataLoader worker 1；45任务最多两轮，12小时是调度目标而非已验证承诺。
- commit `bcc5dfd`的正式加速批次`efficientad-m-frozen-20260828T095200Z-shared23`已启动；
  23/23 CUDA任务进入训练，GPU2--7利用率98--100%，各进程约1278MiB，启动审计各卡仍余
  约12--18GiB，0个失败标记、0个正式metrics。GPU0/1未使用，其他用户进程未被修改。
- 2026-08-28 19:44按用户指令仅终止GPU4--7上属于本项目的12个timeout/Python进程树；
  GPU2/3的11个CUDA训练继续，其他用户进程未动。停止后GPU4--7本项目CUDA进程为0，
  项目CPU占用由约2582%降至约1041%，系统load开始由34回落至23；12个中断任务保留日志与
  checkpoint但不计为正式结果，待GPU2/3任务完成或其他卡用户进程结束后重新运行。
- 当前完整Git历史已推送到私有仓库`https://github.com/CuiMiles/EvoInspect-130`；默认分支
  `master`，本地与远端HEAD均为`b402211`（首次推送核验时），checkpoint和实验大目录未跟踪。
- 2060冻结交接已完成：连接只读检查、通过质量门后构建自包含bundle、远端独立环境安装和
  2500基准脚本均就绪；当前SSH配置仍无目标主机。
- OpenCV实拍视频：5/5、2944/2944帧、98.131秒解码完成；正常视频无逻辑异常，其他视频
  产生12个skip/reorder/repeat/missing事件；仅功能验证。
- GuardedAdapt真实分数replay：15类、5 seeds、75 runs；harmful-update rate为0.0267，
  accepted-update rate为0.8533，拒绝更新rollback 11/11；不声称生产准确率提高。
- 四件草稿约束通过：简介PDF 169中文字符/268非空白字符；项目PDF 5页；MP4 122.133秒、
  25,724,846字节；ZIP小于1MB且完整。团队、队长、学校和单人分工已填写，参赛组别与正式
  命名仍缺。
- 从commit `44d6b0c`的全新目录和全新venv完成README复现：pytest 60/60、ruff、mypy、
  GuardedAdapt重放、提交校验、45任务dry-run和pip check全部通过，未使用GPU。

## Completed before freeze (historical)

- 完整读取并固化 `docs/13_Advisor_reply.md`；不再按导师的逐日日程等待，所有当前可写的
  RCBR 实验代码一次完成。
- 停止 HeteroMemory、GuardedFusion 和 MaskedPrototype；PatchCore 只保留为强精度基线。
- 固定官方 Anomalib tag `v2.3.0`、commit
  `091ca6aca92c8d0e416394f79e52f5a3cea3db73`、Apache-2.0，代码 checkout 干净。
- 隔离 EfficientAD 环境已安装到 `/home/CuiMinghao/envs/evoinspect-efficientad`，未修改现有
  项目/PatchCore 环境；Python 3.11.16、Torch 2.6.0+cu124、Anomalib 2.3.0 可导入，
  EfficientAD-S、Engine 与 Folder 均可实例化。
- EfficientAD 预训练教师已下载到仓库忽略目录，Imagenette 已下载到外部模型目录；两份
  教师文件记录独立 SHA-256，Imagenette 共 13,395 个文件。
- 安装时发现 `imagecodecs 2026.3.6` 不再提供 CPython 3.11 wheel 且源码 limited-ABI 编译
  失败；安装器已固定最新兼容二进制版 `2026.1.14`，重跑成功且 `pip check` 无冲突。
- 实现 RCBR v1：空间正常风险校准、多尺度不一致/高频/位置候选、5 折 ROI 收益估计、
  NMS、实测成本表、最多 4 ROI 的时延/面积硬预算、共享模型局部复检、单调融合和显式回退。
- 每个类别/seed 只训练一个 EfficientAD-S，复用权重与局部推理评估统一下采样、全网格、
  固定 Top-K、不确定性、风险校准和完整 RCBR 六种对照；45 个开发任务而不是 270 次训练。
- 评估器 schema v2 加入固定相对面积 Tiny ≤0.1%、Small 0.1%–1%、Large >1%；原
  q25/q75 保留为兼容补充；两档 AUPRO 均对各面积切片计算。
- 启动器按四类 seed-130、四类补 seeds 131–132、其余 11 类 × 3 seeds 顺序执行；smoke
  或全开发门失败即非零退出，不自动扫参。
- 确认 seeds 138–142 必须同时提供显式环境解锁和通过开发门生成的逐文件 freeze manifest；
  代码/配置或开发门证据发生变化即拒绝确认。
- 新增 2500×2500、batch=1、warmup=100、1000 次的 RTX 3090 合成分辨率端到端性能脚本，
  分解解码、预处理/传输、全局模型、路由、局部模型、后处理和序列化；不冒充 2060 证据。
- GPU 启动器跳过繁忙卡、不触碰现有进程；对使用卡加协作锁并在任务前复检。2026-08-24
  本轮 dry-run 时 8 张 RTX 3090 均为 20 MiB、0% 且无 compute process，但该状态会变化。
- 5000-step RCBR pilot 已完成 12/12，保存 72 个受控策略结果；其 smoke gate 失败，宏平均
  full_rcbr 相对 PatchCore 的 AUPRO@0.05 差值为 -0.17024，Overall F1 差值为 -0.20082，
  因此未扩展其余 11 类，也未解封确认 seeds。
- 正式 70,000-step RCBR raw-score-space fusion smoke 已完成 12/12；`smoke-gate.json` 明确
  `passed=false`。平均 ΔAUPRO@0.05=+0.015647、最差类别=-0.105517、Overall F1=-0.150921、
  Unseen F1=-0.165300，五项预注册 gate 检查全部失败；development 扩展与确认 seeds 138--142
  均未启动。完整负结果报告见
  `reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7/analysis.md`。
- 对已完成但被 gate 否决的 wood-s130 checkpoint 补做 RTX 3090 GPU4、2500×2500、batch=1、
  warmup=100、repeats=1000 延迟测量：正常样本端到端 p50/p95/max=
  350.153/362.552/383.206 ms；scratch 样本为 371.293/386.795/420.942 ms。两次均为 0 ROI，
  仅作工程诊断，不代表最终正向模型或 GTX 2060。
- 已生成最终报告证据索引 `docs/16_FINAL_REPORT_EVIDENCE_20260825.md`，明确主表、负结果、
  延迟范围和未完成项；可据此写报告，但官方最终提交仍为 PARTIAL。
- 已完成 2026-08-27 指导教师版提交就绪度审计：
  `docs/17_ADVISOR_STATUS_AND_SUBMISSION_READINESS_20260827.md`。结论为研究报告可写，但正式
  作品提交不就绪；官方四件提交物均未生成，关键系统/硬件证据仍缺失。
- 新环境内全仓库 `pytest` 47/47 通过；`ruff check .` 通过；严格 `mypy src` 检查 15 个源码
  文件无问题；两个 bash 脚本语法检查和完整 development dry-run 通过。
- 修复并实测 `benchmark_rcbr_latency.py`：补齐修订后 raw-score 融合接口和 PyTorch 2.6
  可信 `PosixPath` checkpoint allowlist；在 GPU 3 的 RTX 3090 上完成 2500×2500、batch=1、
  warmup=100、repeats=1000 的合成分辨率工程基准，端到端 p50=687.479 ms、p95=844.530 ms、
  max=1007.035 ms。该结果使用 5000-step wood checkpoint，只是 RTX 3090 工程测量，不是
  最终模型、原生高分辨率精度或 GTX 2060 证据。
- 针对先前 `evaluate_saved_localization.py` 的 `f1_fixed_threshold` 报错，用官方
  PatchCore 75-run source aggregate 做完整 CPU 重评，75/75 通过且 failures 为空；新结果
  `reports/experiments/upstream-patchcore-localization-reeval-20260824T172300-keycheck/aggregate.json`
  的 Overall F1=0.922353、AUPRO@0.05=0.724099，与既有强基线一致。
- 新增确定性视频顺序/逻辑 FSM 与两级 GuardedAdapt 控制器：即时阈值/记忆更新可逆，候选
  模型需通过反馈收益、锚定回归和影子验证门禁，并保留版本回滚；新增 CPU 测试后全仓库
  pytest 为 54/54，新增模块 ruff 和 mypy 通过。该项是工程闭环证据，不代表真实视频或
  反馈收益。
- 根 Git 已初始化并用于可追溯代码快照；大体积实验目录、权重、NPZ、FAISS 和生成参考被
  排除，未来训练可记录真实 commit。

## Formal smoke outcome

正式修订 smoke 批次已结束：

- batch：`reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7`
- 配置：`configs/baselines/efficientad_s_100_30.yaml`，70,000 steps
- 范围：capsule、hazelnut、transistor、wood × seeds 130--132，共 12 个任务
- 结果：12/12 `metrics.json` 完成，`smoke-gate.json` 为 `passed=false`；无正式 `aggregate.json`
  是预期行为，因为 gate 失败后启动器不会扩展 development。
- gate 失败项：平均 AUPRO@0.05、类别不下降比例、最差类别、Overall F1、Unseen F1 均未达标。
- GPU 安全：任务只使用 GPU 4--7；完成后 GPU 4--7 已释放，GPU 0--3 的其他用户任务未被触碰。
- 详细数值与 provenance：
  `reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7/analysis.md`

训练调度修订：

- `configs/baselines/efficientad_s_100_30.yaml` 新增 `validation_every_n_epochs: 20`；该项只
  控制 EfficientAD 的校准 quantile 刷新频率，不改变训练 loss、步数、数据划分或测试协议。
- `scripts/efficientad_rcbr_100_30.py` 在 fit 后用最终权重对 held-out calibration set 重算一次
  quantile，确保推理归一化不依赖中间 epoch。
- 首轮中断目录只作为工程诊断证据，不能计入性能统计；新批次必须使用新 batch stamp。
- 已终止优化批次：`reports/experiments/rcbr-smoke-20260824T160100Z-rcbr-rawfusion-70k-vf20`，四类
  seed-130 已进入训练，GPU 4--7 保持空闲；截至 16:31 已生成 4 个训练 checkpoint，4 个
  初始 quantile 阶段完成，但尚无 `metrics.json` 或 `smoke-gate.json`；随后发现 GPU 0--2
  出现其他用户进程，本批次已只终止我方进程组并保留 checkpoint，不能计入性能统计。
- 新重跑批次启动时 GPU 4--7 均为 20 MiB/0%；当前仍由 GPU 安全 watchdog 监控，旧批次
  仅保留为中断工程诊断，不能计入性能统计。

## Running now

EfficientAD-M 正式质量门是当前唯一允许的长训练。修复后的45任务新batch正在GPU 2和3
分片执行。不得改配置、重复启动、启动S或恢复其他冻结路线。

## Not run or not yet accepted

- EfficientAD-M修复后正式新批次运行中，当前0/45产生最终metrics，质量门未知；S fallback
  尚未启动。
- 真实 GTX2060连接参数和实测结果缺失；不得写200ms达标。
- 2500 EfficientAD frozen checkpoint benchmark尚未运行。
- 提交草稿仅缺参赛组别和官方文件名；其余团队元数据已填写。

## Existing verified metrics (unchanged)

固定 upstream PatchCore，MVTec AD 标准协议 seed 0：图像 AUROC 0.9909、全像素 AUROC
0.9813。固定 upstream PatchCore，MVTec AD 15 类、seeds 133–137、100+30 式协议：

- Overall F1 0.9224、Image AUROC 0.9817、Unseen F1 0.8715；
- 全像素 AUROC 0.9811、pixel AP 0.5521；
- AUPRO@0.30 0.9342、AUPRO@0.05 0.7241、PRO@1% FPR 0.5764；
- 这些既有定位结果使用旧 q25/q75 面积切片；固定相对面积结果须由新重评产生；
- bottle seed-133、RTX 3090、FP32、224×224、batch=1 的旧模型段 p50 70.76ms、
  p95 81.33ms，不是 2500 端到端或 GTX 2060 证据。

## Failed / eliminated (unchanged)

- HeteroMemory v1/v2、GuardedFusion v1/v3、layer2 路由、MaskedPrototype v4 已淘汰。
- GuardedFusion v2-safe 被正式上游 PatchCore 显著否定，只保留为负结果和研究转向证据。
- 不允许恢复这些路线或在确认集上测试它们。

## Claims allowed today

- 固定 PatchCore 的既有复现结果及其精确协议/硬件边界。
- RCBR 是“已实现、工程链路通过但 5000-step pilot 和正式 70,000-step smoke 均未通过预注册
  gate 的负结果”；可报告正式 gate 数值和失败诊断，不可描述性能收益或优于 PatchCore。
- 可报告 5000-step wood checkpoint 在 RTX 3090 上的合成 2500×2500 时延分解，但必须明确
  checkpoint、GPU、输入重采样和不代表 GTX 2060/官方 200 ms。
- 六种对照、数据隔离、门控、回退和 GPU 安全代码已经存在并通过 CPU/静态测试。
- OpenCV 5段实拍视频的解码、组件事件和FSM功能结果，但必须标注桌面功能验证而非工业benchmark。
- GuardedAdapt 75-run离线MVTec真实分数replay的有害更新率、接受率、回滚率和CPU适应时延，
  但不得描述为生产准确率提高或真实用户研究。
- 四件提交物“草稿已生成且格式/大小约束通过”，不能说“正式可上传”。

## Claims forbidden today

- RCBR 已经提升 AUPRO/F1、已经实时、优于 PatchCore、满足 200ms 或可作为最终模型。
- EfficientAD-M已通过质量门、Edge Engine已冻结、GTX2060<200ms、CPU<2s或原生2500精度。
- 真实视频工业准确率/泛化；GuardedAdapt生产准确率提高或完全阻断有害更新。
- 四件材料已完成正式元数据、命名、人工审校并可直接上传。
- “首次”“首创”“SOTA”“国际领先”“全面超越”或任何获奖保证。

## Blockers / remaining work

- GPU 2已出现安全空闲窗口；长任务运行中仍须监控是否出现其他用户进程。
- 无远程GTX2060主机/IP、SSH端口、用户名、认证方式和工作目录。
- EfficientAD-M质量门、冻结checkpoint与2500时延尚无结果。
- 参赛组别未知，提交草稿尚不能最终定稿。
- MVTec 许可/赛事用途、预训练权重分发、组织方标注/接口/时延口径仍需人工或书面确认。

## Next primary action

持续监控并完成 EfficientAD-M 15类×3 seeds冻结质量门；通过后立即
冻结Edge Engine并进入2500/GTX2060部署基准，失败则保存证据后只做一次S Pareto判断。

## Parallel work

- 获取GTX2060主机/IP、SSH端口、用户名、认证方式和远程工作目录。
- 确认参赛组别，审校PDF/MP4并执行正式命名。
- 人工许可证签核和组织方接口/时延口径书面澄清。
- 清洁复现已完成；代码或依赖发生实质修改后需重新执行。
