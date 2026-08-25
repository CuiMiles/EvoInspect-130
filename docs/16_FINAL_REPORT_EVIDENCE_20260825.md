# 最终报告证据索引（2026-08-25）

本文件把当前已经可以进入报告的真实结果、负结果和限制集中列出。它不是官方提交模板，
也不替代尚未获得的组织方书面澄清。所有数字应从列出的机器可读证据复核，不得从聊天记录
复制。

## 1. 当前可采用的主叙事

建议把作品写成“固定上游 PatchCore 强基线 + 受控 RCBR 失败诊断 + 可信系统闭环”的诚实
研究报告：RCBR 已实现并完成唯一允许的机制级修订，但在预注册正式 smoke 上没有通过，
因此不作为正向性能方法；系统闭环（视频顺序 FSM、GuardedAdapt 门禁/回滚）目前有工程
级 CPU/fixture 证据，不宣称真实视频或反馈收益。

## 2. 可直接写入主结果表

### 2.1 固定上游 PatchCore 强基线

协议：MVTec AD 15 类、seeds 133--137、100+30 支持集隔离、测试标签不用于阈值选择。

| 指标 | 结果 |
|---|---:|
| Overall F1 | 0.9224 |
| Image AUROC | 0.9817 |
| Unseen F1 | 0.8715 |
| Full pixel AUROC | 0.9811 |
| Pixel AP | 0.5521 |
| AUPRO@0.30 | 0.9342 |
| AUPRO@0.05 | 0.7241 |
| PRO@1% FPR | 0.5764 |

证据：
`reports/experiments/upstream-patchcore-100-30-mvtec15-5seed-8gpu-20260823T235656Z-29160/aggregate.json`；
定位重评 75/75 成功证据：
`reports/experiments/upstream-patchcore-localization-reeval-20260824T172300-keycheck/aggregate.json`。

### 2.2 RCBR 正式负结果

协议：四类（capsule、hazelnut、transistor、wood）× seeds 130--132，70,000 steps/任务，
共 12/12 完成。与固定 PatchCore 逐类别配对比较。

| 指标 | full RCBR | PatchCore | Δ |
|---|---:|---:|---:|
| AUPRO@0.05 | 0.655265 | 0.639618 | +0.015647 |
| AUPRO@0.30 | 0.849750 | 0.906595 | −0.056845 |
| Fixed-small AUPRO@0.05 | 0.670734 | 0.775643 | −0.104910 |
| Image AUROC | 0.927320 | 0.987407 | −0.060087 |
| Overall F1 | 0.770043 | 0.920964 | −0.150921 |
| Unseen F1 | 0.760808 | 0.926108 | −0.165300 |
| PRO@1% FPR | 0.564093 | 0.481752 | +0.082341 |

预注册 smoke gate：平均 ΔAUPRO@0.05 要求 ≥0.025，实测 +0.015647；类别不下降仅 2/4，
最差类别 −0.105517；Overall F1 和 Unseen F1 门槛也均失败。该结果只能写成“正式负结果
和诊断”，不能写成提升、实时或优于 PatchCore。

证据：
`reports/experiments/rcbr-smoke-20260824T164000Z-rcbr-rawfusion-70k-gpu4-7/analysis.md`、
`smoke-gate.json`、12 个 `metrics.json` 和该批次 `experiment_registry.csv`。

### 2.3 工程闭环证据

- 全仓库 pytest：54 passed；ruff 和 mypy（EfficientAD 环境）通过。
- 六个确定性序列 fixture 场景：6/6 正确识别事件模式。
- GuardedAdapt：即时阈值/记忆更新可逆；候选模型需要反馈收益、锚定回归、影子验证和
  回滚门禁。

证据：`evidence/system-closure-20260825.txt`、
`reports/experiments/system-closure-sequence-fixture-20260825T054200Z/report.json`、
`docs/15_SYSTEM_CLOSURE.md`。

## 3. 可写但必须带范围的延迟结果

被 smoke gate 否决的 70k wood-s130 checkpoint，在 RTX 3090 physical GPU4、FP32、batch=1、
合成 2500×2500、100 warmup、1000 repeats 下测量：

| 样本 | p50 | p95 | max | ROI |
|---|---:|---:|---:|---:|
| good-000 | 350.153 ms | 362.552 ms | 383.206 ms | 0 |
| scratch-000 | 371.293 ms | 386.795 ms | 420.942 ms | 0 |

这是被否决 checkpoint 的 RTX3090 工程诊断；源图先 resize 到 2500×2500，local model 分支
未被这两个样本触发。不得外推为 GTX2060、CPU、原生高分辨率准确率或官方 200ms 达标。

证据：`evidence/rcbr-latency-20260825.txt` 及其两个 latency JSON。

## 4. 当前不能写成已完成的内容

- RCBR 提升 AUPRO/F1、实时、优于 PatchCore或满足 200ms；
- GTX2060 或 CPU 实测；
- 真实视频准确率、真实反馈收益、遗忘曲线、在线发布效果；
- MVTec AD 2、MVTec LOCO、AHL/DRA 等尚未完成的正式对照；
- seeds 138--142 或完整 15 类 RCBR development；
- 任何“首次/首创/SOTA/国际领先/全面超越”表述；
- 官方提交格式、大小、匿名、视频时长、接口等未知事项的猜测。

## 5. 报告/提交状态判定

- **研究报告可写：** 是。可用 PatchCore 强基线、RCBR 正式负结果、工程闭环和范围受限的
  RTX3090 延迟诊断组成完整、可复现、带限制的报告。
- **官方最终提交可宣称全部要求已满足：** 否。GTX2060/CPU、真实视频/反馈、额外数据集和
  官方提交细则仍缺证据。
- **下一主动作：** 以本索引冻结主表、生成图表和模型使用说明；不再启动 RCBR development
  或 confirmation。若取得真实 GTX2060/等价设备，再单独补测并更新 claim ledger。

## 6. 复现入口

```bash
pytest -q
ruff check .
mypy src
```

正式 smoke 的原始启动和 gate 规则见 `docs/14_RCBR_EXPERIMENT_EXECUTION_PLAN.md`；当前
状态与禁止声明见 `STATUS.md` 和 `evidence/claim_ledger.csv`。
