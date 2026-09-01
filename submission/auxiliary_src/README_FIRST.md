# 智检演化130辅助材料

从这里开始。该包面向评委快速复核，不是内部研究仓库归档。

**四个核心结果：PatchCore F1 0.9224｜真实GTX2060 p95 166.2ms｜有害更新减少71.4%｜视频事件F1 0.9474。**

## 30秒快速检查

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-lock.txt
./run_demo.sh demo/sample_input.png
```

输出位于`demo_output/`：`result.json`包含异常分数、二值决策、模型版本和分阶段时延，
`heatmap.png`为异常热力图。默认模型是EfficientAD-M bottle/seed143代表checkpoint导出的
ONNX FP16文件，用于展示离线推理接口；公开主结果是15类聚合结果，不应把单类Demo外推。

## 证据导航

- `evidence_summary/main_results.json`：主结果汇总；
- `evidence_summary/gtx2060_benchmark.json`：真实2060、1000次端到端测量；
- `evidence_summary/video_event_metrics.json`：五段实拍视频事件评测；
- `evidence_summary/guarded_adapt_metrics.json`：75-run反馈回放；
- `model_card.md`：模型用途、限制和协议；
- `appendix/research_boundaries.md`：负结果和适用边界。

完整源码位于`src/`，最终实时配置位于`configs/final_realtime.yaml`。

## 五段视频功能验证素材

`videos/`目录包含五段固定机位桌面装配演示视频，文件名直接标明场景：正常流程、错序、缺少
mouse、bottle重复和返工后继续正确顺序。视频为当前FSM的带标题标注派生结果，便于评委复核；
它们只用于功能验证，不外推为工业benchmark。对应的事件级GT和评测结果见
`evidence_summary/video_event_metrics.json`。
