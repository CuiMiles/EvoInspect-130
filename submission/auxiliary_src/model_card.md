# EfficientAD-M实时模式模型卡

- 来源：Anomalib 2.3.0 EfficientAD-M，固定bottle/seed143代表checkpoint；
- 输入：RGB，任意尺寸由入口重采样至256×256，batch=1；
- 输出：异常图、图像分数、二值决策与时延；
- 格式：ONNX内部FP16、FP32输入输出，约41.5MB；
- 真实硬件：6GB NVIDIA GeForce RTX 2060；
- 2500×2500重采样端到端p95：166.165ms，100次预热、1000次测量；
- 15类聚合：Overall F1 0.903604、Unseen F1 0.820986、Image AUROC 0.956915。

限制：该代表ONNX文件不是15类通用checkpoint；MVTec每类独立适配。2500输入会缩放到256，
不证明原生高分辨率微小缺陷能力。聚合AUROC和Unseen F1低于项目自设0.97/0.83研究门，
精度优先场景应使用PatchCore。模型只用于离线竞赛演示，不保证生产泛化。

