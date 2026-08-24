# EvoInspect-130 研究与参赛执行包

## 项目定位

**中文名称：** 智检演化 130

**英文名称：** EvoInspect-130

**副标题：** 面向 100 张正常样本与 30 张缺陷样本快速迁移的可信自学习实时 AOI 系统

本执行包按“数月科研与工程迭代”设计，不以九天交付为边界。目标不是仅做出可演示原型，而是形成一套具有明确算法贡献、可复现实验、实时部署能力、反馈闭环和高完成度答辩材料的作品。

> 重要现实：官网当前页面显示，2026 届报名截至 2026-08-25、作品提交截至 2026-09-01。本包按用户要求提供长期研究版路线；若仍参加当前届，需要另行制作缩减版，不能把数月路线误当作当前期限内必然可完成的承诺。

## 先读顺序

1. `AGENTS.md`
2. `project_spec.yaml`
3. `docs/01_OFFICIAL_AOI_SPEC.md`
4. `docs/02_RESEARCH_BLUEPRINT.md`
5. `docs/03_ALGORITHM_DESIGN.md`
6. `docs/04_EVALUATION_PROTOCOL.md`
7. `docs/05_NOVELTY_CLAIMS_AND_PACKAGING.md`
8. `docs/07_SYSTEM_AND_PRODUCT_REQUIREMENTS.md`
9. `docs/10_MILESTONES.md`
10. `prompts/P1_MASTER_RESEARCH_BUILD.md`

## Codex 使用方式

不再使用二十多个碎片化提示词。仓库只保留三个提示词：

- `P1_MASTER_RESEARCH_BUILD.md`：首次启动，负责建仓、复现、研发、实验和持续记录。
- `P2_CONTINUE_RESEARCH.md`：后续每次会话继续工作时重复使用。
- `P3_FINAL_REDTEAM_AND_SUBMISSION.md`：算法冻结后用于独立审查、文档、演示和提交打包。

长期规则、技术要求、实验矩阵和创新声明都放在仓库文件中，由 Codex 每次自行读取。这样提示词只负责“启动和调度”，不会重复塞入所有需求。

## 工作原则

- 按国一/华为专项奖的作品标准建设，国三只是下限目标，不承诺奖项。
- 创新可以是增量创新、组合创新、任务特定创新和工程方法创新，但不得把已有方法直接改名为“全球首次”。
- “包装”指把真实差异、实验收益和工业价值讲清楚，不是伪造独创性或指标。
- 每个创新点必须对应：最近工作、精确差异、消融实验、失败边界和可复现证据。
- 最终在线图不得依赖大语言模型或大扩散模型；8×3090 用于教师模型、缺陷生成、并行实验和蒸馏，最终模型必须轻量。

## 官方源

- 大赛赛题页面：
  https://cpipc.acge.org.cn/cw/contestNews/detail/2c9088a5696cbf370169a3f8101510bd/2c9080179e403028019e496a8feb14b9?page=0
- 华为赛题 PDF 镜像（AOI 位于第 1-3 页）：
  https://statics.scnu.edu.cn/pics/yjsy/2026/0617/1781679006478549.pdf
