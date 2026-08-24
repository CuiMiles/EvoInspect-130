# STATUS

updated_at: 2026-08-24T11:12:00+08:00
current_phase: G1_RCBR_CODE_READY
overall_status: READY_FOR_USER_DEVELOPMENT_RUN

## One-sentence truth

固定上游 PatchCore 强基线与既有负结果保持不变；导师已批准 RCBR 主线，代码、六种受控
对照、固定相对面积评测、两级门控、封存确认锁和共享 GPU 安全启动器均已通过 CPU/静态
验证，但 EfficientAD-S/RCBR 尚未正式训练，因此当前没有新的算法指标或可提交创新结论。

## Completed

- 完整读取并固化 `docs/13_Advisor_reply.md`；不再按导师的逐日日程等待，所有当前可写的
  RCBR 实验代码一次完成。
- 停止 HeteroMemory、GuardedFusion 和 MaskedPrototype；PatchCore 只保留为强精度基线。
- 固定官方 Anomalib tag `v2.3.0`、commit
  `091ca6aca92c8d0e416394f79e52f5a3cea3db73`、Apache-2.0，代码 checkout 干净。
- 新增隔离 EfficientAD 环境安装器，不修改现有项目环境或 PatchCore 环境；预训练教师和
  Imagenette 下载使用上游给定 SHA-256。
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
- 全仓库 `pytest` 44/44 通过；`ruff check .` 通过；严格 `mypy src/evoinspect` 通过；两个
  bash 脚本语法检查和完整 development dry-run 通过。
- 根 Git 已初始化并用于可追溯代码快照；大体积实验目录、权重、NPZ、FAISS 和生成参考被
  排除，未来训练可记录真实 commit。

## Ready to run

1. `bash scripts/setup_efficientad_env.sh`
2. `EVOINSPECT_DRY_RUN=1 bash scripts/run_rcbr_experiment_suite.sh development`
3. `bash scripts/run_rcbr_experiment_suite.sh development 2>&1 | tee logs/rcbr-development.log`

完整命令、输出和判定规则见 `docs/14_RCBR_EXPERIMENT_EXECUTION_PLAN.md`。

## Not run in this code-preparation turn

- 未安装新的 EfficientAD 环境；未下载 EfficientAD 教师或 Imagenette；
- 未启动 GPU 训练、推理、PatchCore 开发参考 CPU 重评或正式 2500 时延循环；
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
- RCBR 是“已实现并通过工程测试的待验证候选方法”；可描述设计，不可描述性能收益。
- 六种对照、数据隔离、门控、回退和 GPU 安全代码已经存在并通过 CPU/静态测试。

## Claims forbidden today

- RCBR 已经提升 AUPRO/F1、已经实时、优于 PatchCore、满足 200ms 或可作为最终模型。
- GTX 2060、CPU、原生 2500 高分辨率、小缺陷跨数据集、视频逻辑和反馈闭环已经完成。
- “首次”“首创”“SOTA”“国际领先”“全面超越”或任何获奖保证。

## Blockers / remaining work

- 首先需要用户运行 45 次开发训练并返回完整 batch；在此之前无法判断 RCBR 是否保留。
- AHL/DRA 少监督开放集基线、MVTec AD 2、MVTec LOCO、视频 FSM、反馈/影子发布/回滚、
  GTX 2060、CPU 和最终提交包仍未完成。
- MVTec 许可/赛事用途、预训练权重分发、组织方标注/接口/时延口径仍需人工或书面确认。
- 若 smoke 失败，导师只允许最多一次机制级修订；不得用确认种子调参。

## Next primary action

用户运行 `scripts/run_rcbr_experiment_suite.sh development`，随后把命令末尾给出的完整
batch 路径和 `logs/rcbr-development.log` 返回，先分析 smoke/full gate 再决定唯一一次修订
或冻结；不要直接运行 confirmation。

## Parallel work

- 可并行取得 GTX 2060/等价低端设备，但不占用当前开发 GPU 任务。
- 可并行下载并登记 MVTec AD 2 / LOCO、集成 AHL 或 DRA、实现视频 FSM 和反馈回滚骨架。
- 可并行完成人工许可证签核、组织方书面澄清和最终提交材料框架。
