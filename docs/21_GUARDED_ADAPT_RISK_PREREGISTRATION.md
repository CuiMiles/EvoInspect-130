# GuardedAdapt-Risk 预注册协议

冻结日期：2026-08-29（北京时间）  
状态：代码与参数冻结后、正式 replay 指标计算前

## 1. 方法边界

GuardedAdapt-Risk 是唯一新增方法线；不修改 PatchCore backbone，不恢复 RCBR、GuardedFusion、
HeteroMemory 或其他检测模型。增量 PatchCore 记忆本身不声明原创，贡献限定为：漂移触发、
有界候选、正常/已见/未见群组风险预算、32样本影子发布与精确回滚的统一发布协议。

## 2. 冻结组件

1. 最近正常分数以32样本为窗口；两样本 KS `p<0.01` 且中位数偏移大于参考 IQR 的0.25倍；
   连续两个窗口成立才生成候选。
2. 阈值候选最多移动反馈分数 IQR 的0.25倍。
3. 记忆候选只加入操作员标为正常的 patch embedding；先选离 Champion 最远的候选，再执行
   greedy diversity；最多替换原 FAISS 正常记忆的5%，且每次不超过256个 patch。被替换项为
   第二近邻距离最小的冗余原记忆。backbone和原模型目录不变。
4. 风险门使用2000次配对 bootstrap；四个单侧检验以 Bonferroni 控制家族置信度95%。必须
   同时满足新工况 F1 增益 LCB>0、历史正常 FPR 增量 UCB≤1%、已见缺陷 FNR 增量 UCB≤1%、
   未见缺陷 FNR 增量 UCB≤2%。
5. 离线门通过后在后续32个隔离样本影子验证；Champion继续提供正式决策。影子风险超限时
   恢复 Champion 特征和阈值，并逐决策检查完全一致。

全部常数只来自 `configs/innovations/guarded_adapt_risk.yaml`。结果产生后不允许修改这些常数
并重跑；任何必要代码修复必须登记、重新冻结并把旧结果作废。

## 3. 数据与隔离

- 旧静态 replay：固定75个 PatchCore 真实分数流；增加 GuardedAdapt-Risk 的统计阈值门，
  明确标记此子实验没有图像或 patch，不能证明记忆更新。
- 图像漂移 replay：bottle、cable、capsule、carpet、transistor、wood，seeds 133--135，四个
  固定漂移，共72次。
- 错误反馈 replay：相同72次，反馈训练标签按 sample-id SHA-256 确定性翻转约10%。
- 总量：75+72+72=219。

100+30式支持集内部固定分出参考窗口、两个漂移窗口、反馈训练、反馈验证和32样本 shadow；
transistor公开数据在该固定划分中只有22张support anomaly，因此六类统一采用8/7/7张缺陷
作为反馈训练/验证/影子；影子另取25张正常样本，保持32样本且各角色不重叠。其余类别的
额外support anomaly不参与本实验。
最终测试按 `normal/seen/unseen` 和 sample-id 哈希轮转为 target、gate anchor、audit anchor。
三者无 sample-id 重叠。模型先对所有分区固定 Champion/Candidate 分数，门禁只读取 target/gate；
shadow决定晋升；audit最后打开，仅用于估计有害更新和质量门，不反向影响候选。

四种漂移参数固定为亮度1.20、色温RGB=(1.10,1.00,0.90)、高斯模糊半径1.50、JPEG质量55。

## 4. 对照与硬门

统一比较 NoUpdate、NaiveUpdate、BoundedThreshold、GuardedAdapt-v1、GuardedAdapt-Risk。
Risk只有同时满足以下条件才进入核心创新主张：总回放≥200；harmful≤1.5%；相对v1下降≥50%；
接受率≥40%；被接受更新 target F1 平均增益≥0.01；三组风险上界不超预算；有害候选阻断≥80%；
回滚100%；泄漏0。低接受率、无正收益或任一隔离失败均判定不通过。

## 5. 允许与禁止声明

通过时只允许写“在所测冻结 MVTec 离线 replay 上降低有害更新并保护历史群组风险”。不得写
真实操作员研究、生产准确率提高、首次提出增量记忆、完全阻断有害更新或奖项保证。失败时
保留完整负结果，降级为工程安全机制，不寻找第二条创新路线。

## 6. 冻结结果（2026-08-29）

正式实验完成219/219次回放，划分哈希为
`28cc1dcb86bf6ac481ee323133227689ee409151d169fafe48b7436e1803c2f4`，18个真实图像任务均
记录commit `342ac7a`且`dirty=false`，泄漏事件为0。

GuardedAdapt-Risk 将 harmful-update rate 降为0，阻断50/50个事后判定有害的v1候选，
219次拒绝均精确回滚；但它拒绝219/219个更新，accepted-update rate=0，低于预注册的40%，
被接受更新的target收益和群组风险UCB均不可计算。因此总质量门`passed=false`。这证明当前
统计门禁过于保守，只能作为安全机制负结果，不能作为核心创新或正向收益声明；按本协议不再
修改门限、不重跑、不寻找第二条创新路线。

机器可读证据：
`reports/experiments/guarded-adapt-risk-20260829-preregistered-e17419c/report.json`。
