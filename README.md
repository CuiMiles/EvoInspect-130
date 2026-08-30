# EvoInspect-130

EvoInspect-130（智检演化130）是一套面向工业新产品“100正常+30缺陷”快速适配的离线图像/
视频AOI系统。系统以PatchCore作为精度模式、EfficientAD-M ONNX FP16作为实时模式，以组件
检测+FSM识别装配逻辑，并通过GuardedAdapt对操作员反馈执行候选门禁、版本化和回滚。

## 已验证结果

- PatchCore，MVTec AD 15类×5 seeds：Overall F1 0.9224、Image AUROC 0.9817、Unseen F1 0.8715；
- EfficientAD-M，15类×3 seeds：Overall F1 0.9036、Image AUROC 0.9569、Unseen F1 0.8210；
- 真实6GB GTX2060，2500×2500重采样、batch=1、100 warmup、1000 repeats：EfficientAD-M
  ONNX FP16 model-only p95 19.4ms、端到端p95 166.2ms；
- GuardedAdapt-v1，75个真实分数流离线回放：有害更新率2.67%、接受率85.33%、拒绝更新
  回滚11/11；
- 五段固定机位实拍视频：19个GT事件匹配18个，事件Micro F1 0.9474。

上述为公开基准、真实硬件和桌面功能验证，不代表官方隐藏集或生产泛化，不保证奖项。测试
标签不参与阈值、训练、早停或模型选择。允许声明以`evidence/claim_ledger.csv`为准。

## 快速验收

```bash
EVOINSPECT_PY=/home/CuiMinghao/envs/evoinspect-efficientad/bin/python
"$EVOINSPECT_PY" -m pytest -q
"$EVOINSPECT_PY" -m ruff check .
"$EVOINSPECT_PY" -m mypy src
```

运行反馈回放：

```bash
PYTHONPATH=src "$EVOINSPECT_PY" scripts/evaluate_guarded_adapt_replay.py \
  --config configs/innovations/guarded_adapt_replay.yaml \
  --output /tmp/evoinspect-guarded-replay/report.json
```

运行视频功能验证：

```bash
PYTHONPATH=src "$EVOINSPECT_PY" scripts/evaluate_video_demo.py \
  --input-dir data/video/video_5 --output-dir /tmp/evoinspect-video-demo
PYTHONPATH=src "$EVOINSPECT_PY" scripts/evaluate_video_events.py \
  --predictions /tmp/evoinspect-video-demo/report.json \
  --ground-truth data/derived/video/desktop_assembly_gt_v1.1_frozen.json \
  --output /tmp/evoinspect-video-event-evaluation.json
```

## 提交物

正式命名文件位于`submission/final/`。简介和6页项目文档由
`scripts/build_submission_pdfs.sh`生成；约158.87秒项目视频由
`scripts/build_submission_video.py`生成；评委版辅助包由`scripts/build_auxiliary_zip.sh`生成，
内含41.5MB EfficientAD-M ONNX模型、Demo、模型卡与机器证据摘要。

算法实验已经停止。HeteroCal-130、GuardedAdapt-Risk和RCBR的预注册负结果只保留在证据与
边界附录，不进入简介、封面或视频KPI。
