# GTX 2060 远程实测交接

更新时间：2026-08-28。此文档是连接远程服务器前的冻结操作边界，不包含任何虚构主机或硬件
结果。

## 当前边界

- 本仓库没有远程主机、端口、用户名、认证方式或工作目录；SSH 配置中也没有 2060 条目。
- 本机 8 张 RTX 3090 均有其他用户计算进程，未启动 EfficientAD-M。
- 因 M 的 45 个任务和质量门尚未完成，当前不存在可合法打包的 Edge checkpoint。
- 2060 只测冻结模型，不用其结果调阈值、选 checkpoint 或继续扫参。

## 已预先冻结的硬件测量选择

质量门必须先整体通过。硬件时延测量固定使用 EfficientAD-M `bottle / seed 143` checkpoint；
此选择在训练前写入 `configs/baselines/efficientad_m_100_30.yaml`，只代表同构 M 网络的部署
时延，不用来代表 15 类精度，也不根据测试指标或时延挑选。

输入为一张明确指定并打包的源图，基准在内存中将其缩放为 2500×2500。报告 batch=1、
warmup=100、repeats=1000，分别记录 decode、preprocess/transfer、model-only、postprocess、
serialization 和 end-to-end 的 p50/p95/p99/max。它不是原生 2500 图像精度实验。

## 本地 GPU 空闲后

```bash
scripts/run_efficientad_frozen_8gpu.sh m
```

仅当生成的 `quality-gate.json` 中 `passed=true` 后，使用预声明 run 构建包：

```bash
PYTHONPATH=src python scripts/prepare_remote_2060_bundle.py \
  --quality-gate BATCH/quality-gate.json \
  --checkpoint BATCH/runs/efficientad-m-bottle-s143-*/result/model.ckpt \
  --metrics BATCH/runs/efficientad-m-bottle-s143-*/result/metrics.json \
  --test-inputs BATCH/runs/efficientad-m-bottle-s143-*/test_inputs.csv \
  --config configs/baselines/efficientad_m_100_30.yaml \
  --output artifacts/evoinspect-2060-bundle.tar.gz
```

工具固定复制 CSV 第一行指向的源图，也可用 `--image` 显式提供。本工具拒绝未通过质量门、
非 M、非 bottle/seed143、缺少开发阈值、tracked worktree 脏或覆盖已有包的情况。

## 用户接通 SSH 后

先做只读检查，不安装、不传文件：

```bash
scripts/check_remote_2060_connection.sh USER@HOST
```

确认输出确为 GTX 2060、GPU 无其他计算进程、磁盘空间充足后，再把 tar.gz 传到用户指定目录
并解压。远端在解压目录执行：

```bash
EVOINSPECT_BOOTSTRAP_PYTHON=python3 scripts/setup_remote_2060_env.sh
scripts/run_remote_2060_benchmark.sh
```

环境安装需要远端可访问 Python 包源。运行器默认拒绝非 2060 和已有计算进程，使用协作锁，
且拒绝覆盖结果目录。结果目录包含 `latency-2500.json`、设备完整信息、依赖版本和 bundle
manifest；只有这些机器证据审计通过后才能更新 claim ledger。

## 连接时必须提供

1. SSH 主机/IP、端口、用户名和认证方式（推荐先配置 SSH alias）；
2. 用户有写权限且空间足够的远程工作目录；
3. 是否允许访问 PyPI/PyTorch wheel 源，或已有兼容 Python 3.11 环境；
4. `nvidia-smi` 显示的实际设备名、驱动/CUDA，以及运行窗口内 GPU 是否独占空闲。

若 M 质量门失败，不构建 M 部署包；保存失败证据后只评估一次 S 的质量/速度 Pareto，不能
为了 2060 结果无边界调参。
