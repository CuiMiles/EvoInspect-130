# 数据、合成与许可证计划

## 1. 数据原则

- 只使用许可允许的公开数据和预训练模型；
- 原始数据只读，派生数据可重建；
- 记录下载 URL、版本、哈希、许可证和引用；
- 不把公开基准测试标签用于阈值和模型选择；
- 不把不同数据集的近重复图像跨划分泄漏；
- 任何自建视频或合成数据都公开生成脚本和来源。

## 2. 数据角色

- MVTec AD / DAGM：基础外观异常和官方对齐；
- MVTec LOCO AD：逻辑和结构组合；
- MVTec AD 2：高分辨率、小缺陷、复杂成像和漂移；
- VisA / MPDD / Real-IAD / HSS-IAD：跨域、材料和真实变体扩展；
- 自建可复现视频扰动集：顺序、缺件、多件、断帧、模糊。

## 3. 100+30 支持集生成

为每次试验保存 manifest：

- normal_support 100；
- anomaly_support 30；
- development；
- final_test；
- seen_anomaly_types；
- unseen_anomaly_types；
- content hashes；
- random seed；
- 数据集和类别版本。

支持集不足时，不静默重复。必须在报告中明确使用的替代规则。

## 4. 合成数据来源

合成样本分三层：

- 图像级/像素级外观合成；
- 几何和组件结构扰动；
- 视频状态和顺序扰动。

每个合成样本保存：源图哈希、缺陷来源、mask、变换参数、生成模型版本、seed、结构过滤结果、难度分数和最终是否入选。

## 5. 训练/评测隔离

- 训练合成器时不能使用最终测试图；
- 不能根据最终测试效果反复选择合成策略；
- 合成样本不能与测试中的真实缺陷形成像素级复制；
- 任何由基础模型产生的伪标签都记录置信度，低置信度不进入主训练集。

## 6. 许可证登记模板

`data/dataset_registry.yaml` 每项包含：

- name / version；
- source_url；
- paper_url；
- license；
- redistribution_allowed；
- commercial_use_allowed；
- downloaded_at；
- archive_hash；
- local_path；
- citation_text；
- restrictions；
- reviewer。

上游代码和模型在 `third_party/THIRD_PARTY_NOTICES.md` 中登记。
