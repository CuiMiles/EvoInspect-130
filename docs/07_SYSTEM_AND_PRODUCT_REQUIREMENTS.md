# 系统与产品需求

## 1. 用户角色

- 算法工程师：创建产品、适配模型、查看实验；
- 产线操作员：实时检测、确认误检/漏检、查看解释；
- 质量负责人：查看趋势、版本、回归和审计；
- 评委：一键演示、查看指标来源和系统完整度。

## 2. 核心流程

### 新产品适配

1. 导入 100 正常 + 30 缺陷；
2. 自动校验格式、重复、分辨率、标注和异常分布；
3. 学习对齐、组件、正常记忆和缺陷方向；
4. 运行适配与校准；
5. 在内部开发集验证；
6. 生成版本、模型卡和适配报告。

### 图片检测

输出：正常/异常、分数、mask/框、组件、known/unseen、置信度、最近正常证据、时延和模型版本。

### 视频检测

输出：帧级外观热力图、组件状态、事件时间段、逻辑/顺序异常解释和告警等级。

### 反馈更新

操作员选择误报/漏报/新缺陷/正常变体，修正区域或类别。系统显示更新影响、候选版本回归结果和是否允许发布。

## 3. CLI

至少支持：

- `data validate`
- `data split`
- `train baseline`
- `adapt product`
- `evaluate`
- `infer image`
- `infer video`
- `feedback ingest`
- `feedback build-candidate`
- `model promote`
- `model rollback`
- `export onnx`
- `export tensorrt`
- `benchmark latency`
- `report generate`
- `package verify`

## 4. API

建议：

- `POST /v1/products`
- `POST /v1/products/{id}/adapt`
- `POST /v1/inspect/image`
- `POST /v1/inspect/video`
- `POST /v1/feedback`
- `POST /v1/candidates/build`
- `POST /v1/models/{version}/promote`
- `POST /v1/models/{version}/rollback`
- `GET /v1/models`
- `GET /v1/evidence/{run_id}`
- `GET /healthz`
- `GET /readyz`

## 5. UI 页面

1. 总览与产线状态；
2. 新产品 130 样本适配；
3. 图片实时质检；
4. 视频与顺序异常；
5. 未知缺陷聚类与复核；
6. 反馈中心；
7. 候选版本、回归门禁和回滚；
8. 性能/精度/漂移仪表盘；
9. 模型和数据审计。

## 6. 非功能要求

- 离线可运行；
- 所有推理请求可追踪；
- 失败显式返回，不静默回退；
- 模型版本、配置和阈值不可隐式变化；
- 支持断点恢复和批量评测；
- 支持 CPU fallback，但未满足 2s 时明确告警；
- 容错：损坏图片、空视频、不同编码、OOM、模型缺失、并发；
- 单元、集成、端到端和性能回归测试齐全。

## 7. 解释性要求

解释界面至少回答：

- 哪个区域异常；
- 哪个组件/步骤异常；
- 更像已知缺陷还是未知缺陷；
- 与哪个正常样本差异最大；
- 置信度和需不需要人工复核；
- 哪个模型版本做出决定；
- 反馈后模型为何变化。

解释来自模型证据和结构化规则，不生成不可验证的自然语言原因。

## 8. 最终验收

- 一条命令从样本适配到 1000+ 测试推理；
- 一条命令生成所有主表和图；
- 一条命令完成清洁环境 smoke test；
- 图片、视频、反馈、发布、回滚全部可离线演示；
- 任何主张能从界面跳到 run_id 和 evidence；
- 演示断网不受影响。
