# STATUS

updated_at: 2026-09-01T02:01:03+08:00
current_phase: BOUNDED_FOUR_ROUTE_SCREEN_EDGEFUSION_AND_PARETO_COMPLETED_DINO_QUEUED
overall_status: FINAL_ARTIFACTS_FROZEN_EXPLORATORY_SCREEN_RUNNING_WHEN_SAFE_GPU_RELEASES

## Bounded four-route screen reopened by the user (2026-09-01)

The immutable `submission/final` package remains unchanged. A new, explicitly bounded screen
was registered from the supplied final-sprint decision: EfficientAD-S+M EdgeFusion, AnomalyDINO,
Dinomaly, and GuardedAdapt-Pareto. The two completed CPU-side routes are recorded below; the
remaining DINO routes are queued behind a conservative GPU supervisor.

- EfficientAD-S+M EdgeFusion finished all 15 categories from existing strict-v2.1 maps without
  retraining. Fixed alpha=0.5 and support-derived normalization produced Overall F1 `0.917409`,
  eligible Unseen F1 `0.842150`, and Image AUROC `0.969500`; 15/15 coverage and zero leakage. It
  narrowly failed the preregistered AUROC gate (`0.970000`) and is not promoted or benchmarked on
  the 2060. Evidence: `reports/experiments/edgefusion-20260901/summary.json`.
- GuardedAdapt-Pareto evaluated 30 fixed development policies over 50 development replays and
  held out 25 audit replays. No policy met all development requirements simultaneously: the
  best-acceptance points retained harmful rates above 2% or blocked only 50% of harmful candidates;
  stricter points dropped acceptance below 50%. The report is a negative result with zero leakage,
  and no audit policy was selected. Evidence:
  `reports/experiments/guarded-adapt-pareto-20260901/report.json`.
- AnomalyDINO and Dinomaly are now registered as six-category seed-143 screens, with all output
  written outside `submission/final`. DINOv2-Small weights are available locally. AnomalyDINO
  cable completed on CPU as an entry-point smoke/evaluation run (F1 `0.870370`, AUROC
  `0.933190`); the remaining five AnomalyDINO categories and all six Dinomaly categories await a
  GPU with no compute process. No GPU is considered free merely because utilization is low.
- The partial AnomalyDINO aggregate is available at
  `reports/experiments/final-sprint-20260901/anomalydino-summary.json` (1/6 complete, so the
  six-category gate is intentionally `passed=false`). It is not a model-selection result.
- `scripts/monitor_final_sprint.py` polls every 30 seconds (therefore providing at least a
  half-hourly audit trail), requires no compute-app PID, >=8 GiB free, and <=5% utilization, and
  never kills processes. It will launch at most one route per safe GPU and then continue the queue.
  Current snapshot at this update: all eight GPUs have other-user compute processes; no project
  GPU process has been started.

## Additional three-route completion (2026-08-31)

The three previously unexecuted branches of the user-requested six-route screen are now complete
under `docs/26_ADDITIONAL_ROUTES_SCREEN_20260831_PREREGISTRATION.md`. DefectAdapter-130, the
official SuperSimpleNet architecture with a strict-manifest adapter, and the official DRA base
route each completed six categories at seed 143. All 18 runs wrote predictions before opening test
truth, recorded `dirty=false`, and have zero test-label leakage.

- DefectAdapter: Overall F1 `0.841964`, eligible Unseen F1 `0.727339`, Image AUROC `0.949546`.
- SuperSimpleNet: Overall F1 `0.578926`, eligible Unseen F1 `0.380210`, Image AUROC `0.849865`.
- DRA: Overall F1 `0.591371`, eligible Unseen F1 `0.360305`, Image AUROC `0.791648`.

All three fail the fixed `0.905/0.84/0.97` screen. Because DRA did not pass, the full AHL feature
generation and meta-learning pipeline is not unlocked under the preregistered stop rule. No route
is promoted, no additional seed or parameter search is allowed, and `submission/final` remains
unchanged. Machine-readable evidence is
`reports/experiments/additional-routes-screen-20260831/additional-routes-summary.json`.

## User-requested parallel screening (2026-08-31)

The previously machine-validated `submission/final/` package was held as the immutable fallback
during experimental screening; no exploratory metric was allowed to rewrite its claims. A new,
explicitly exploratory screen was started from `configs/experiments/parallel_screening_20260831.yaml`
after the user requested a six-route/parallel sprint. The implemented first screen is deliberately
limited to three train-free EfficientAD-S variants on six categories (`cable`, `capsule`, `screw`,
`carpet`, `transistor`, `wood`) at seed 143: S-384, S-512, and fixed non-overlapping 2x2 tile-S.
It reuses completed checkpoints and strict support/test manifests; it does not retrain or alter the
frozen final materials. The GPU supervisor is `scripts/monitor_parallel_screening.py`, samples
`nvidia-smi` every 30 seconds (so the log contains many more than half-hourly snapshots), allows at
most two processes per startable GPU, requires >=4 GiB free and <=5% utilization, and never kills a
process it did not launch. At launch, only GPU5 was
startable; GPUs 0--4 and 6--7 had other-user compute activity. No screening result is eligible for
the abstract or final package until an explicit aggregate review.

## Exploratory screen results (2026-08-31)

The first screen is complete: all 18 pre-registered runs (three EfficientAD-S variants times six
categories at seed 143) exited successfully with zero test-label leakage. The means are:

- S-384: Overall F1 `0.736895`, eligible Unseen F1 `0.575690`, Image AUROC `0.755735`, model p95
  `17.928 ms`.
- S-512: Overall F1 `0.716230`, eligible Unseen F1 `0.529432`, Image AUROC `0.674494`, model p95
  `27.630 ms`.
- StaticTile-S: Overall F1 `0.723587`, eligible Unseen F1 `0.562900`, Image AUROC `0.689808`,
  model p95 `39.515 ms`.

The corresponding frozen S-256 six-category mean is Overall F1 `0.820185`, eligible Unseen F1
`0.677041`, and Image AUROC `0.936540`; all three train-free variants therefore degrade and are
not promoted. The machine-readable aggregate is
`reports/experiments/parallel-screening-20260831/screening-summary.json`.

The HeteroResidual-S six-category screen also completed 6/6 with zero leakage. Its aggregate is
Overall F1 `0.820185`, eligible Unseen F1 `0.677041`, Image AUROC `0.936540`, and model p95
`8.026 ms`; it equals the frozen base because the fixed leave-one-defect-type-out rule rejected
five heads (one category accepted a head without changing the six-category mean). It is a negative
exploration and is not exported or placed in submission materials. Evidence is
`reports/experiments/heteroresidual-screen-20260831/heteroresidual-summary.json`.

SuperSimpleNet, AHL/DRA, DINO and GLASS do not have a registered 100+30 evaluator in this checkout,
so they are not launched. The separately registered EfficientAD-S384 training screen was queued on
idle GPUs 4--7: six categories, seed 143, 384x384 training, 70,000 steps, strict support-only
evaluation, one task per GPU and 30-second polling. GPUs 0--3 continued to be left untouched when
occupied by other users. The supervisor automatically started `transistor` and `wood` after the
first four tasks released GPUs; all six tasks exited successfully. Its preregistration is
`docs/25_EFFICIENTAD_S384_SCREEN_20260831_PREREGISTRATION.md`.

## EfficientAD-S384 training screen result (2026-08-31)

The six pre-registered S384 runs completed 6/6 with zero test-label leakage. Under the same strict
support-only evaluator as the frozen S-256 comparison, means were Overall F1 `0.865589`, eligible
Unseen F1 `0.709348`, Image AUROC `0.932945`, and model p95 `10.375 ms`. The frozen S-256 six-class
means were Overall F1 `0.820185`, eligible Unseen F1 `0.677041`, and Image AUROC `0.936540`.
S384 therefore improved F1 (+4.54 percentage points) and Unseen F1 (+3.23 points) but reduced
Image AUROC (-0.36 points), with weak per-category generalization (cable Unseen F1 `0.095238`).
It does not pass the frozen quality gates and does not beat the comparison on all registered
dimensions; it is recorded as a mixed/negative exploration and is not promoted. No further
EfficientAD resolution training is started under the preregistered stop rule, and the experiment
does not change final claims. A later introduction-PDF text-layer repair is separately recorded as
submission-only maintenance. The machine-readable aggregate is
`reports/experiments/efficientad-s384-screen-20260831/s384-summary.json`.

## One-sentence truth

GuardedAdapt-Risk 与 EfficientAD-M 均已完成冻结实验并正式失败：M的45/45 strict-v2.1
Overall F1=0.903604通过，但Unseen F1=0.820986和Image AUROC=0.956915未过线；0 failure、
0泄漏。EfficientAD-S唯一一次15类seed143筛查也已完成：Overall F1=0.890785通过，但
Unseen F1=0.798847和Image AUROC=0.963851未过线；0 failure、0泄漏。M/S冻结质量门结论仍保留；
S384训练筛选已完成6/6，Overall F1=0.865589、eligible Unseen F1=0.709348、Image AUROC=0.932945，F1有所提升但AUROC下降，未通过质量门，不改变冻结结果或提交材料。真实GTX2060上，ONNX FP16的S/M 2500×2500端到端p95分别为
151.343/166.165 ms，速度均过200 ms目标；但两者冻结质量门仍失败，所以没有合格Edge Engine。

HeteroCal-130已按预注册协议完成45/45五组消融：完整方法Overall F1=0.898108、eligible
Unseen F1=0.818395、Image AUROC=0.957035，均未达到0.90/0.83/0.97门；只有3/45 run通过
support类型留一选择，零测试标签泄漏，总门`passed=false`。按12小时硬停止规则，HeteroCal
不导出、不上2060、不进简介或主视频，不再调参；算法工作结束。

竞赛材料已改为正向证据叙事：简介突出PatchCore、EfficientAD-M真实2060、GuardedAdapt-v1
和视频GT；项目PDF固定6页且删除封面PRELIMINARY/失败警告；项目视频重构为约158.87秒，
覆盖图像热力图、双模架构、真实2060、代表视频事件与反馈接受/拒绝/回滚；辅助ZIP重构为
评委入口、41.5MB ONNX模型、Demo、模型卡和四份证据摘要，不再打包AGENTS/STATUS/旧计划。

## Freeze completion snapshot

- EfficientAD-S 15/15 checkpoint与strict-v2.1重评已完成，15类齐全、14个eligible unseen
  run、toothbrush 1个run明确N/A、0 failure、0 test-label leakage。Overall F1=0.890785
  （门槛0.89，通过），eligible Unseen F1=0.798847（门槛0.83，失败），Image AUROC=0.963851
  （门槛0.97，失败），总门`passed=false`。S的RTX 3090、256x256模型段诊断p50/p95均值为
  6.877/17.624 ms，相比M的14.721/34.758 ms更快，但不是GTX2060或2500端到端证据；S只
  保留为速度Pareto负结果/硬件诊断候选；本次S384训练筛选另行记录，不覆盖该冻结结论。

- EfficientAD-M 45/45 checkpoint与strict-v2.1重评全部完成，15类齐全、42个eligible unseen
  run、toothbrush 3个run明确N/A、0 failure、0 test-label leakage。Overall F1=0.903604
  （门槛0.89，通过），eligible Unseen F1=0.820986（门槛0.83，失败），Image AUROC=0.956915
  （门槛0.97，失败），总门`passed=false`。M冻结为负结果，不修改学习率、分辨率、步数、
  seed或阈值策略；本轮S384训练是独立的六类seed143探索。

- GuardedAdapt-Risk 已按 commit `342ac7a`、冻结划分哈希
  `28cc1dcb86bf6ac481ee323133227689ee409151d169fafe48b7436e1803c2f4` 完成：旧75次
  分数回放 + 72次真实图像漂移 + 72次10%确定性错误反馈，共219次、0泄漏。Risk阻断
  50/50个有害v1候选、harmful-update rate=0、rollback=219/219，但拒绝219/219个更新，
  accepted-update rate=0（门槛≥0.40），被接受更新收益与三组风险UCB不可计算；总质量门
  `passed=false`。该路线冻结为负结果，不调参、不恢复第二创新路线。
- 实例49225420已实机确认是6 GiB NVIDIA GeForce RTX 2060。固定2500×2500、batch=1、
  warmup=100、repeats=1000下，PyTorch FP32的S/M端到端p95为327.922/331.273 ms，均失败；
  ONNX Runtime CUDA FP16的S/M端到端p95为151.343/166.165 ms，model-only p95为
  8.205/19.355 ms，均通过200 ms速度目标，单样本二值决策与PyTorch参考一致。该证据只证明
  实际硬件速度与数值保真，不推翻M/S质量门失败，也不证明原生2500分辨率异常检测精度。
- 视频真实顺序经素材提供者纠正并冻结为`cup -> bottle -> mouse`。保留原始5段视频及哈希，
  新增v1.1人工GT、±0.5秒窗口和一对一最大二分匹配；2944/2944帧重放后19个GT事件匹配18个，
  Micro Precision=Recall=F1=0.947368。前4段F1=1.0，返工视频F1=0.8；唯一错误源于前端不输出
  REMOVE动作，未通过修改GT消除。该结果只属于固定机位桌面功能验证。
- EfficientAD strict evaluator 已修复少样本类别边界：toothbrush 沿用冻结训练划分
  38个正常训练 + 10个正常校准，并使用22个support anomaly；不使用development/test定阈值。
  协议缺陷产生的33份旧结果已可恢复归档，commit `7272064`下统一重评当前36个checkpoint，
  36/36成功、dirty=false、泄漏事件0；最终质量结论仍须等待45/45。
- 从commit `e359e06`的全新clone和全新Python 3.11 venv按README完成最终冻结CPU/静态复现：
  pytest 67/67、ruff、mypy 20个源码文件、pip check、45任务dry-run、全部shell语法和提交物
  约束均通过，未启动GPU。证据为`evidence/final_clean_reproduction_20260829.json`。

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
- 每小时监控/恢复脚本`monitor_and_resume_efficientad.sh`已启动；日志写入当前批次
  `hourly_monitor.log`。触发条件为GPU2/3本轮11个worker全部退出且原启动器结束；届时仅
  移动失败结果目录到可恢复备份并按原run-id重跑缺失项，不会重复已完成项。
- 2026-08-29 07:22前GPU2/3保留任务已全部退出；批次现有12个正式metrics、22个主动停止/驱动
  中断failure，未生成完整quality gate。07:49--07:50连续三次`nvidia-smi -L`均无法连接
  NVIDIA驱动，系统无本项目CUDA进程；恢复监控已改为驱动不可见时持续记录但禁止启动。
- 07:53解除受限执行环境后确认物理GPU驱动正常；GPU0--3均为20MiB、0%且无计算进程，
  GPU4--7仍有其他用户任务。恢复计划固定为GPU0--3各3个slot，共12路，仅重跑33个缺失项。
- 22个中断`result`目录已移动为`result.interrupted-20260829T075400`备份；12个正式metrics保留。
  07:54恢复批次已在GPU0--3启动12/12 CUDA任务，四卡利用率98--99%、各约3.9GiB，0个
  当前failure；GPU4--7其他用户进程未修改。
- 2026-08-29 17:27快照：批次26/45 metrics、0个当前failure、9/15类，10个本项目CUDA
  worker仍在GPU0--3运行。阶段宏平均Overall F1=0.872878、24个eligible run的Unseen
  F1=0.779026、Image AUROC=0.945544。当前scorer下即使剩余19项AUROC全为1.0，最终上限
  也只有0.968537，低于0.97；这些数字因下述协议偏差不得作为正式M质量结论。
- evaluator审计发现三项阻塞：baseline runner未读取已存在的30张`support_anomaly`而使用额外
  development异常定阈值；项目使用99.5%分位图像分数而固定上游EfficientAD使用`amax`；
  toothbrush单缺陷类型导致`unseen=null`且现聚合器会失败。最终路线和停止条件见
  `docs/20_FINAL_ROUTE_DECISION_20260829.md`。
- 2060冻结交接已完成：连接只读检查、通过质量门后构建自包含bundle、远端独立环境安装和
  2500基准脚本均就绪；当前SSH配置仍无目标主机。
- OpenCV实拍视频：5/5、2944/2944帧、98.131秒解码完成；修正工序后19个GT事件匹配18个，
  Micro P/R/F1均为0.947368；仅固定机位桌面功能验证。
- GuardedAdapt真实分数replay：15类、5 seeds、75 runs；harmful-update rate为0.0267，
  accepted-update rate为0.8533，拒绝更新rollback 11/11；不声称生产准确率提高。
- 四件草稿已按最终视频GT和2060结果重建并通过约束：简介PDF 129中文字符/252非空白字符；
  项目PDF 6页；MP4 122.133秒、28,887,378字节；ZIP 676,204字节且完整。团队、队长、学校
  和单人分工已填写，参赛组别与正式
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

当前无本项目GPU训练、评测或远端计算进程。HeteroCal-130已完成并按失败门硬停止；不得调参、
导出或上2060。四件正式命名文件已重建到`submission/final/`并完成机器验收，
`evidence/submission_artifact_validation.json`记录`constraints_passed=true`和
`final_upload_ready=true`。2026-08-31进一步完成全部6页项目PDF和1页简介的渲染审查、项目
视频4766/4766帧完整解码及7个代表时间点抽帧审查、辅助ZIP完整性检查；简介PDF重新生成了
干净文本层，修复旧PDF文本抽取重复和标题缺失问题。全仓pytest 72/72、ruff、mypy 22个源码
文件、pip check和全部shell语法通过。评委版辅助Demo已在一次性本地ONNX Runtime 1.22.0 CPU环境实跑，代表输入输出
normal、score=0.188335，模型哈希与2060证据一致；远端2060实例当前端口拒绝连接，不影响已
完成的硬件证据。尚需参赛者本人播放视频、审阅PDF并在官方平台上传。

从commit `e9733ad`的全新clone和全新Python 3.11 venv再次按README完成最终复现：pytest
72/72、ruff、mypy 22个源码文件、pip check、全部shell语法和四件正式命名文件验收均通过；
证据为`evidence/final_clean_reproduction_20260830.json`。

## Not run or not yet accepted

- EfficientAD-M和S均已完成且质量门失败；目前没有通过冻结质量门的Edge Engine。
- GTX2060实机基准已完成；允许精确报告ONNX FP16速度过线，但不得称M/S质量合格或已形成
  合格Edge Engine。
- Codex已完成PDF逐页渲染、视频全帧解码和代表帧审查；尚未由参赛者本人完成最终审阅、
  学校审核、官方平台上传和上传后回下载校验。
- 项目视频当前为无旁白画面版；机器格式合格，但是否补录讲解音轨需参赛者人工决定。

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
- 可报告5段实拍桌面视频的事件级Micro Precision=Recall=F1=0.947368（18/19，一对一
  ±0.5秒匹配）；必须注明GT在系统审查后由用户确认、固定机位功能验证和非工业benchmark。
- GuardedAdapt 75-run离线MVTec真实分数replay的有害更新率、接受率、回滚率和CPU适应时延，
  但不得描述为生产准确率提高或真实用户研究。
- 可报告真实GTX2060、2500×2500重采样输入、batch=1、100 warmup/1000 repeats下的精确
  速度：ONNX FP16 S/M的model-only p95为8.205/19.355 ms，端到端p95为
  151.343/166.165 ms；必须同时说明FP32端到端失败、质量门失败和不代表原生2500精度。
- 四件提交物“草稿已生成且格式/大小约束通过”，不能说“正式可上传”。

## Claims forbidden today

- RCBR 已经提升 AUPRO/F1、已经实时、优于 PatchCore、满足 200ms 或可作为最终模型。
- EfficientAD-M/S已通过质量门、合格Edge Engine已冻结、CPU<2s或原生2500精度。
- 真实视频工业准确率/泛化；GuardedAdapt生产准确率提高或完全阻断有害更新。
- 四件材料已经通过机器和内部渲染/解码审查，但不得声称已完成参赛者本人审校、学校审核、
  官方平台上传或上传后回下载校验。
- “首次”“首创”“SOTA”“国际领先”“全面超越”或任何获奖保证。

## Blockers / remaining work

- EfficientAD-M/S与GuardedAdapt-Risk均未通过冻结正向门，当前核心创新与Edge质量证据不足。
- GTX2060速度已闭环，但不能抵消M/S质量门失败；当前只有“速度达标的失败质量候选”。
- 正式文件需要用户人工打开审阅并上传；本轮未重复运行完整性和哈希校验。
- MVTec 许可/赛事用途、预训练权重分发、组织方标注/接口/时延口径仍需人工或书面确认。

## Next primary action

完成正式命名复制、评委版ZIP远端Demo烟测和四件文件最终机器验收，然后只做人工审阅与上传。

## Parallel work

- 人工打开6页PDF和158.87秒MP4检查文字、画面与声音需求。
- 人工许可证签核和组织方接口/时延口径书面澄清。
- 最终代码修改后执行一次全新目录复现。
