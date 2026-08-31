# GuardedAdapt-Pareto 预注册协议

冻结时间：2026-09-01（运行任何参数网格结果前）

## 研究问题

在已有 75-run 离线 PatchCore 反馈回放中，v1（高接受率）和 Risk（零接受率）之间是否存在
可复核的中间风险—收益工作点。该实验只重新计算已保存的候选反馈增益、gate-anchor 回退和
独立 audit 结果，不修改检测模型或提交包。

## 隔离

十个类别（bottle、carpet、grid、hazelnut、leather、metal_nut、pill、screw、tile、toothbrush）
仅用于选择 `min_feedback_gain` 与 `max_gate_anchor_regression`。五个类别（cable、capsule、
transistor、wood、zipper）在参数选择完成后才打开，且不反向修改参数。候选由已记录的
ThresholdUpdate 指标定义；不读取任何新测试数据，不产生模型权重。

## 固定网格与门槛

网格共 30 个点，参数和 tie-break 写入配置。开发集选择目标是：接受率≥50%、有害率≤2%、
有害候选阻断率≥80%、接受更新 target gain>0、回滚成功率100%，并最大化接受更新的 target
gain。审计集使用相同门槛；任一泄漏或失败只记录为负结果。

机器入口：`scripts/evaluate_guarded_adapt_pareto.py`。
