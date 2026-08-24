# PatchCore-lite bottle 三 seed 初步报告

> 这是单一类别、小样本的预实验，不是完整 MVTec AD 均值，不是官方 PatchCore
> 复现，也不是赛事最终结果。

## 数据与协议

- 数据：MVTec AD `bottle`，来自固定提交
  `Voxel51/mvtec-ad@30a183a3b96e3aef953f230784b123b719b09d97` 的社区镜像。
- 完整性：292/292 张图逐文件匹配镜像 LFS SHA-256；manifest SHA-256 为
  `dc0cb96c23d9f0eb3abef6ffcf581c1d750708d6c81ff7868ddc3f3b38700f8d`。
- 每个 seed：100 正常 + 30 个 seen 缺陷作为支持集；20 正常 + 6 个 seen 缺陷作为
  development；最终测试为 20 正常 + 6 seen + 21 unseen contamination，共 47 张。
- 阈值只由 development 选择。推理进程只读取无标签 `test_inputs.csv`；评测进程只在
  预测完成后读取无图像路径的 `test_truth.csv`。三个 seed 均通过 ID 不交叉和视图隔离检查。
- 模型：torchvision Wide-ResNet-50-2 ImageNet V2 特征，layer3 的 2048 个随机正常 patch
  记忆向量构成 PatchCore-lite；另设全局特征 logistic regression 作为监督对照。

## 真实指标

| 方法/切片 | AUROC mean±sd | AP mean±sd | 固定阈值 F1 mean±sd | Accuracy mean±sd |
|---|---:|---:|---:|---:|
| PatchCore-lite overall | 0.9969±0.0028 | 0.9977±0.0021 | 0.9338±0.0240 | 0.9291±0.0246 |
| PatchCore-lite seen | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 |
| PatchCore-lite unseen | 0.9960±0.0036 | 0.9963±0.0035 | 0.9132±0.0319 | 0.9187±0.0282 |
| Linear head overall | 0.9117±0.0396 | 0.9613±0.0166 | 0.8337±0.0381 | 0.8369±0.0325 |
| Linear head unseen | 0.8865±0.0509 | 0.9425±0.0243 | 0.7876±0.0396 | 0.8211±0.0282 |

PatchCore-lite 每个 seed 的 overall 固定阈值 F1 分别为 0.9615、0.9200、0.9200；漏检数
分别为 2、4、4，且没有假阳性。当前证据支持继续推进 patch memory 路线，但测试集很小，
不可据此作泛化或优越性结论。

## 训练资源与时延

- 训练和推理只使用物理 GPU 1（RTX 3090），每个任务前检查无其他 compute process，
  `CUDA_VISIBLE_DEVICES=1`；未终止、抢占或迁移任何其他进程。结束后 GPU 1 为 20 MiB、0%。
- 峰值 PyTorch 显存：477,419,008 bytes（约 455.3 MiB）。
- 适配耗时：seed 130 为 103.80 s，包含首次下载 275,905,729-byte backbone；缓存后 seed
  131/132 分别为 7.41/7.34 s。因此 103.80 s 不能作为纯训练耗时，缓存后均值为 7.38 s。
- RTX 3090、batch=1、224×224、预热 10 次、不含模型加载与输出文件 I/O：三 seed 的
  端到端 p50 均值 31.18 ms，p95 均值 42.20 ms，最大观测 206.98 ms。
- 单个适配产物 `model.pt` 为 8,451,282 bytes，但还依赖 275,905,729-byte 外部 backbone
  权重；不得把 8.45 MB 当作完整部署体积。

上述时延不是 2500×2500，也未在 GTX 2060 实测，不能声称达到官方 200 ms 参考目标。

## 产物与限制

- 汇总：`reports/experiments/pc-lite-bottle-3seed-20260823T123000Z.json`
- 分 seed：`reports/experiments/pc-lite-bottle-s{130,131,132}-20260823T123000Z/`
- 代码质量：ruff 通过，mypy 12 个源文件通过，pytest 12/12 通过。
- Git provenance：工作区 `.git` 仍为不可用只读占位，commit 只能记录为 `UNAVAILABLE`。
- 许可证：MVTec AD 是 CC BY-NC-SA 4.0；社区镜像身份、赛事用途和预训练权重分发仍待人工签核。

