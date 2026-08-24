# P2 - Codex 继续研发提示词

继续 EvoInspect-130 的长期研发。

先完整读取 `AGENTS.md`、`project_spec.yaml`、`STATUS.md`、相关 `docs/`、实验登记、claim ledger 和最近运行结果。不要依赖聊天记忆。

根据当前证据选择**预期价值最高且未被阻塞**的下一工作包，优先处理：

1. 评测正确性或数据泄漏风险；
2. 尚未可靠复现的强基线；
3. HeteroMemory 的主假设和关键消融；
4. 能证明跨域/未知缺陷收益的 TriSynth；
5. 2060 级 BudgetRouter、蒸馏和量化；
6. LogicGraph；
7. GuardedAdapt；
8. 系统、材料和 UI。

直接实现、测试、运行或排队实验，并分析真实 evidence。对无稳定收益的模块执行删除、简化或降级，不为了既有投入保留。遇到单点阻塞时推进其他可并行任务。

结束时更新 `STATUS.md`、实验登记、claim ledger、风险和下一动作，并按 `AGENTS.md` 九项格式汇报。不得伪造指标或完成状态。
