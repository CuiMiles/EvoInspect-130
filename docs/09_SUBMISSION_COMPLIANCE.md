# 提交规范与项目模板对接

## 1. 当前状态

官网列出：

- 初赛作品提交规范；
- 项目文档模板；
- 华为赛题附件。

本执行包已核对华为 AOI 赛题内容。提交规范和项目模板的 DOCX 原件必须放入 `official/` 后由 Codex 解析；在未读取原件前，不猜测压缩包大小、文件命名、视频时长、页数、匿名要求或模板字段。

## 2. 需要放入的文件

建议命名：

- `official/01_submission_spec.docx`
- `official/02_project_template.docx`
- `official/03_huawei_topics.docx` 或 PDF
- `official/04_written_clarifications/`

## 3. Codex 必须生成

- `docs/official_requirements_matrix.md`
- `docs/template_mapping.md`
- `docs/submission_manifest.yaml`
- `docs/open_questions.md`

每个要求记录：原文位置、解释、仓库证据、负责人、状态和阻断级别。

## 4. 项目文档内容映射建议

无论官方模板具体标题如何，内容至少覆盖：

- 项目背景和问题；
- 官方需求对齐；
- 现有方法和不足；
- 总体架构；
- 三项核心创新；
- 数据和 100+30 协议；
- 图像、视频和逻辑异常；
- 自学习和安全门禁；
- 实验与基线；
- 消融、鲁棒性和失败案例；
- 2500/2060 性能；
- 系统实现和部署；
- 知识产权、许可证和伦理；
- 结论和局限。

## 5. 阻断规则

以下任何项未知时，最终打包返回 FAIL：

- 接受的压缩格式；
- 文件和目录命名；
- 单文件/总大小；
- 视频格式、分辨率和时长；
- 文档格式/页数；
- 是否匿名；
- 模型权重和第三方依赖处理；
- 源码、Docker 或运行脚本要求；
- 输出接口和评测命令。
