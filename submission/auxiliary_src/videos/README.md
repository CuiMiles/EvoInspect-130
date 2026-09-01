# 五段视频功能验证素材

这些文件是固定机位桌面装配视频经过当前 OpenCV/FSM 流程生成的带可读标题标注版本，便于评委
直接查看。它们对应冻结工序 `cup -> bottle -> mouse`：

1. `01_normal_cup_bottle_mouse.mp4`：正常流程
2. `02_order_violation_bottle_before_cup.mp4`：错序
3. `03_missing_mouse.mp4`：缺少 mouse
4. `04_bottle_repeated.mp4`：bottle 重复
5. `05_rework_then_correct_sequence.mp4`：返工后继续正确顺序

这些视频仅作固定机位功能验证，不是工业 benchmark；对应事件级真值和指标见
`../evidence_summary/video_event_metrics.json`。
