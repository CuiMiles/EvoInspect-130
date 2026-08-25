# STATUS

updated_at: 2026-08-25T08:32:00+08:00
current_phase: G1_RCBR_FORMAL_SMOKE_REJECTED_SYSTEM_DEPLOYMENT_PIVOT
overall_status: FORMAL_RCBR_SMOKE_FAILED_RCBR_PIVOT

## One-sentence truth

固定上游 PatchCore 强基线与既有负结果保持不变；RCBR 5000-step pilot 与唯一 raw-score-space
fusion 机制修订的正式 70,000-step smoke 均未通过预注册 gate。正式批次 12/12 完成，宏平均
ΔAUPRO@0.05=+0.015647 但低于 +0.025 门槛，ΔOverall F1=-0.150921，ΔUnseen F1=-0.165300；
RCBR 性能扩展已停止，转入 PatchCore 强基线 + 系统/部署贡献的报告收敛路线。

## Completed

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

## Ready to run

RCBR smoke gate 已失败，不得运行 development 或确认 seeds。下一阶段应冻结 PatchCore 强基线，
补齐最终延迟/部署证据，并整理报告；正式 smoke 的原始命令、输出和判定规则见
`docs/14_RCBR_EXPERIMENT_EXECUTION_PLAN.md`。

## Not run or not yet accepted

- 正式 70,000-step 修订 smoke 已完成 12/12，但未通过 smoke gate；RCBR 不得作为最终正向
  性能结果；
- 5000-step pilot 与正式 70,000-step smoke 均已保留为负结果证据；
- 已对被否决的 70k wood checkpoint 重跑正式 2500 延迟循环；仍未取得通过 gate 的最终模型，
  当前测量为 RTX 3090 合成分辨率工程诊断，且 local ROI 分支未被样本触发；
- 未读取或运行 seeds 138–142；
- 没有新增 accuracy、AUPRO、F1、ROI 面积或时延实测值。

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

## Claims forbidden today

- RCBR 已经提升 AUPRO/F1、已经实时、优于 PatchCore、满足 200ms 或可作为最终模型。
- GTX 2060、CPU、原生 2500 高分辨率、小缺陷跨数据集、视频逻辑和反馈闭环已经完成。
- “首次”“首创”“SOTA”“国际领先”“全面超越”或任何获奖保证。

## Blockers / remaining work

- 正式 smoke 已失败；RCBR 算法创新必须降级为负结果，不能再继续扫参、补齐其余类别或扩展确认集。
- development 与确认 seeds 138--142 已按 gate 规则保持封存。
- AHL/DRA 少监督开放集基线、MVTec AD 2、MVTec LOCO、视频/反馈协议的真实数据实测、
  GTX 2060、CPU 和最终提交包仍未完成；视频 FSM 与 GuardedAdapt 工程骨架及 CPU 测试已完成。
- MVTec 许可/赛事用途、预训练权重分发、组织方标注/接口/时延口径仍需人工或书面确认。
- 最终性能基准必须使用通过开发门后选定的 checkpoint 重跑；当前延迟结果不能替代该步骤。
- 若 smoke 失败，导师只允许最多一次机制级修订；不得用确认种子调参。

## Next primary action

冻结 PatchCore 主基线与 RCBR 负结果/延迟证据，完成报告、模型包、使用说明和 claim ledger；
若能获得 GTX 2060 或等价设备，再做独立硬件复测，但不得再启动 RCBR development 或 confirmation。

## Parallel work

- 可并行取得 GTX 2060/等价低端设备，但不占用当前开发 GPU 任务。
- 可并行下载并登记 MVTec AD 2 / LOCO、集成 AHL 或 DRA、实现视频 FSM 和反馈回滚骨架。
- 可并行完成人工许可证签核、组织方书面澄清和最终提交材料框架。
