# EvoInspect-130

EvoInspect-130（智检演化130）是面向工业组装图像/视频的离线 AOI 研究原型。当前处于
`PRELIMINARY-SUBMISSION FREEZE`：PatchCore 固定为 Accuracy Engine；EfficientAD-M 正在按
同一 MVTec AD 100+30 隔离协议复现，EfficientAD-S 仅作为速度 fallback；GuardedAdapt 是唯一
主要系统创新。RCBR 已正式失败并冻结，只保留为研究决策负结果。

本仓库不保证奖项，不声称未实测的 GTX 2060/CPU 时延，也不使用测试标签调阈值、挑模型或
早停。允许/禁止声明以 `evidence/claim_ledger.csv` 和 `STATUS.md` 为准。

## 已验证证据

- 固定上游 PatchCore，MVTec AD 15 类、seeds 133–137、100+30 式隔离协议：Overall F1
  0.9224、Image AUROC 0.9817、Unseen F1 0.8715；非官方隐藏集。
- RCBR 70,000-step 四类三种子正式 smoke：12/12 完成但 gate 失败，禁止作为正向创新。
- OpenCV 真实视频功能链路：5/5 视频、2944/2944 帧解码，FSM 输出 skip/repeat/reorder/
  missing/unknown 词汇和时间区间；固定机位桌面演示，不是工业 benchmark。
- GuardedAdapt：15 类五种子、75 个冻结 PatchCore 真实分数流离线反馈回放；不是生产用户研究，
  也不把 target gain 宣称为生产准确率提高。
- 四件初赛材料已有约束合格的草稿；团队元数据、正式命名、EfficientAD-M 与 GTX 2060 仍是
  最终上传阻断项。

机器证据入口见 `docs/18_PRELIMINARY_SUBMISSION_FREEZE.md`。

## 清洁环境快速验收

下面的命令只安装 CPU/视频测试依赖，不下载数据、不启动 GPU：

```bash
EVOINSPECT_BOOTSTRAP_PYTHON=/home/CuiMinghao/envs/evoinspect-efficientad/bin/python
"$EVOINSPECT_BOOTSTRAP_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,yaml,metrics,images,video]'
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
```

运行冻结真实分数反馈回放：

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_guarded_adapt_replay.py \
  --config configs/innovations/guarded_adapt_replay.yaml \
  --output /tmp/evoinspect-guarded-replay/report.json
```

运行视频功能验证（原始视频放在 `data/video/video_5/*.mp4`，原始数据不提交）：

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_video_demo.py \
  --input-dir data/video/video_5 \
  --output-dir /tmp/evoinspect-video-demo
```

## EfficientAD-M/S 冻结实验

隔离环境和上游固定方式见 `scripts/setup_efficientad_env.sh`。正式 M 路线为 15 类 × seeds
143–145，共 45 个独立任务；启动器只选择无计算进程、显存不高于 256 MiB 且利用率不高于
5% 的 GPU，并使用协作锁，不终止任何其他用户进程。

先做无 GPU dry-run：

```bash
EVOINSPECT_DRY_RUN=1 scripts/run_efficientad_frozen_8gpu.sh m
```

确认共享服务器有安全空闲卡后启动 M：

```bash
scripts/run_efficientad_frozen_8gpu.sh m
```

只有 M 未通过冻结质量线且 S 具有速度/质量 Pareto 价值时才运行：

```bash
scripts/run_efficientad_frozen_8gpu.sh s
```

M 质量线固定为 Overall F1 ≥0.89、Unseen F1 ≥0.83、Image AUROC ≥0.97、15 类任务完整、
test-label leakage=0。失败后不得无边界调参。

## 2500×2500 与 GTX 2060

训练完成后，从冻结 run 中选择预先声明的 checkpoint，在目标设备运行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
/path/to/evoinspect-efficientad/bin/python scripts/benchmark_efficientad_latency.py \
  --checkpoint RUN/result/model.ckpt \
  --metrics RUN/result/metrics.json \
  --test-inputs RUN/test_inputs.csv \
  --config configs/baselines/efficientad_m_100_30.yaml \
  --output RUN/latency-2500.json \
  --physical-gpu 0
```

该基准固定 batch=1、2500×2500、warmup=100、repeats=1000，并分别报告 model-only 与
end-to-end p50/p95/p99。只有 `nvidia-smi` 实际记录为 GTX 2060 的报告才能进入对应声明。

## 提交物

草稿位于 `submission/drafts/`，约束检查：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_submission_artifacts.py \
  --output evidence/submission_artifact_validation.json
```

辅助 ZIP 可通过 `scripts/build_auxiliary_zip.sh` 重建。正式上传前必须填写团队名称、参赛组别、
作者/成员、学校和分工，并按官方文件名重命名。
