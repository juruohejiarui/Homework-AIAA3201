# 2026-11-02 指定版本与第二版对比

核对日期：2026-11-02，Asia/Shanghai。

## 权威来源

用户本次指定的文件（5 页）：
`/home/hjr/Downloads/Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency (1).pdf`

对比基准为用户于 2026-10-31 指定的第二版（5 页）：
`/home/hjr/Downloads/Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf`

## 对比结论

两份文件完全相同，没有新增、删除或修改的课程要求。

- `cmp -s` 返回 0：文件逐字节一致，包括文本、图片、链接和 PDF 元数据。
- 两份文件 SHA-256 均为：
  `ea65076917e87febb0c3c67773b9fc6edf304e2821a74266a03f22ef33abc226`。
- `pdftotext -layout` 提取的全文差异为空；已查看本次指定 PDF 的全部 5 页渲染。

因此本次是权威来源路径的重新指定，不构成课程要求的第三种内容版本。
第一版到第二版的变化仍见 `PROJECT_REQUIREMENTS_REVIEW_20261031.md`；
该历史审阅中的代码与结果状态没有在本次重新核查。

## 继续适用的主要口径

- Middlebury：15 个 trainQ 立体对，BM 或 SGBM；按标定、doffs、毫米转米和 OpenCV
  视差除 16 的约定转换。有限且正的 GT 视差上报告视差 MAE 和有效预测比例。
- NYU：开发集 795 张、测试集 654 张；选择一个米制模型（推荐 UniDepthV2 ViT-S，
  Depth Pro 也可），RGB-only，无测试集尺度拟合或对齐。
  预测输出为 480×640，裁剪 `[45:471, 41:601]`，GT 范围 0.001–10 m，
  预测截断到该范围，逐图等权汇总 AbsRel、RMSE、δ1。
- 视频：指定 davis_rollercoaster.mp4 前 100 帧、原帧率；DA-V2-Small 与 VDA-Small
  同帧比较，relative depth，默认 FP16，VDA 离线推理。
  每方法整段固定归一化，改进版复用未改进方法的归一化。
- Part 3：完成一项改进并与未改进方法对比；视频改进允许受控定性比较与运行时间测量。
  不要求额外有标注视频集。额外数据和既有用户扩展实验不自动成为课程必需项。
- 交付：CVPR 格式 6–8 页报告（参考文献除外）、公开 GitHub 与清楚的 README、
  摘要末的仓库链接、规定视频对比、一项简单应用输出及 depth_results.zip。
  arXiv 可选。还需实际素材中的三个难例，以及含硬件、模型、精度、输入尺寸的效率记录。

## 跨对话执行规则

已在仓库根目录 `AGENTS.md` 持久记录本次指定来源，供本项目各对话读取遵循。
若历史计划、报告或其他对话中的课程口径与之冲突，以本次指定 PDF 为准。
两帧模型等既有用户实验选择仍保留，不能写成 PDF 明文要求。

本次用户请求仅授权比较和更新权威来源；PDF 中的下载、训练、公开仓库和提交要求
属于课程交付要求，不据此自动执行下载、实验、发布或提交。
本次没有修改模型、训练、评价代码或实验结果。
