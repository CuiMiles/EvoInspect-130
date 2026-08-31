# 初赛提交物工作区

`drafts/`中保留可审查草稿，`final/`中保存正式命名提交文件。团队Cuisine、企业赛题组、
队长崔明浩、西安交通大学和单人分工已经登记。真实GTX 2060结果已进入机器证据。

最终材料以四条已验证正向证据为主线：PatchCore精度模式、EfficientAD-M真实GTX2060实时
模式、GuardedAdapt-v1安全反馈和视频FSM。GuardedAdapt-Risk、RCBR与HeteroCal负结果只在
辅助材料边界附录中简述，不进入简介、封面或视频KPI。

机器约束验收记录写入 `evidence/submission_artifact_validation.json`；生成源文件保留在本目录，
避免只保存不可修改的二进制成品。

简介PDF由`/usr/bin/python3`下的ReportLab确定性生成，避免LibreOffice HTML导入产生重复文本层；
项目文档仍由LibreOffice生成。执行入口统一为`scripts/build_submission_pdfs.sh`。
