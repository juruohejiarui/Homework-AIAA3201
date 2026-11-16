# Midterm Checkpoint：展示准备核查与完整内容编排

核查日期：2026-11-16。提交截止：2026-11-17（周二）12:00 noon，Asia/Shanghai；Canvas提交不超过5分钟的视频。

## 结论与展示定位

现有技术成果足以完成这次checkpoint，无需先追加训练或实验。
建议主线：**实现深度估计基线 → 区分相对深度和米制深度 → 用开发集拟合一个全局尺度 → 展示测试收益、失败及下一步。**
方向B作为本次主要改进；方向A保留为已有额外时序研究，放备份页，不在5分钟主视频中展开完整模型矩阵。
这是基于时间和现有证据的内容安排，不是删除或否定方向A成果。

Git状态：main已执行pull，仍为`50411a2`；方向B实际在`origin/part3-b-metric-scale`，已fetch到`ad24f94`并只读核查，没有合并代码。
本目录的B素材是该分支报告的副本，来源记录见`assets/source.json`。

## 已核实的材料

| Checkpoint要求 | 可直接使用的证据 | 判断 |
| --- | --- | --- |
| Current Approach | 立体几何、UniDepthV2单图米制预测、DA/VDA视频比较、方向B全局尺度校准 | 足够，压缩为一张流程图 |
| Baseline Implementation | BM/SGBM各15场景；UniDepthV2全部654张；DA-V2-S和VDA-S均有完整演示 | 足够，展示输出而不是终端截图 |
| Visualization & Preliminary Results | 立体对照、NYU RGB/GT/depth、原速视频、B校准前后图与逐图结果 | 足够，优先使用代表图与视频 |
| Future Improvements | 197张图退化、校准不能修复局部结构、跨场景泛化尚待验证 | 有具体可检验的下一步 |

B代码核查：使用795张官方开发图拟合；每张在有效域求`GT/raw prediction`的中位数，再对795个值求中位数；只得到一个全局scale。
该scale固定应用于654张测试图，之后才做预测截断；测试GT不参与拟合。
已核对654个测试ID、逐图记录及均值；隔离运行的两项回归测试通过。
此次没有下载B的全部预测归档重新运行模型；表中数值来自分支保存的逐图结果并经过汇总复算。

## 8页、4分45秒的主视频

全部标题可用英文，正文每页控制在2–3条短句。以下时间包含页面切换及视频播放，总计285秒，保留15秒硬余量。

| 时间 | 页 | 英文标题 | 核心内容 | 画面与讲述动作 |
| --- | --- | --- | --- | --- |
| 0:00–0:20 | 1 | From RGB to Reliable Depth | 问题、项目名称、组员；既要结构，也要米制尺度和时间表现 | 放一张RGB/depth对照，开门见山 |
| 0:20–0:55 | 2 | Our Pipeline and Current Progress | Middlebury几何；UniDepthV2米制预测；DA/VDA视频；新增B校准 | 自绘一张分支流程图，标注15 scenes、654 test images、795 dev images |
| 0:55–1:30 | 3 | Stereo Baseline: Accuracy and Coverage | 两种立体方法已实现，正确视差转深度；误差和覆盖率共同解释 | 一张代表场景对照加两行表，指出无效区 |
| 1:30–2:00 | 4 | Frame-wise vs Video Depth | DA-V2-S与VDA-S同帧相对深度；检查细杆、边缘和运动 | 嵌入原速三面板视频，播放一次，再冻结局部；70 frames / 24 fps写脚注 |
| 2:00–2:40 | 5 | Direction B: One Global Scale | 795张dev拟合一个scale，冻结后用于654张test；RGB-only推理 | 简图Dev → Fit s → Freeze → Test；只展示`d_cal=s*d_raw`和s=0.9341 |
| 2:40–3:20 | 6 | Held-out Results: Calibration Helps | 三项指标与改善幅度 | 展示校准前后表，强调20.4%是AbsRel相对下降，δ1是+2.50个百分点 |
| 3:20–4:05 | 7 | Where It Helps—and Where It Fails | 有改善也有退化；457张改善、197张退化；只改变尺度，不修复细节 | 依次显示#0523改善、#0508退化，两行图分次出现，保留同一米制色标 |
| 4:05–4:45 | 8 | Next Steps | 开发集尺度稳定性；误差归因；简单应用与最终交付 | 显示三条行动和明确验证方式，末尾停留仓库链接 |

## 每页应讲清楚什么

### 1. 问题与当前进度

研究问题：从单张RGB获得可靠深度，并观察视频中的稳定性。区分三个维度：场景结构、绝对尺度、时间表现。
定位为当前进度汇报，不用花时间介绍全部深度估计历史。

英文讲稿：

> Our project studies depth estimation from RGB images and video. We care about scene structure, absolute distance, and stability across frames. We have implemented the required baselines and completed a lightweight improvement: calibrating metric depth with one global scale learned from development images.

### 2. 总体技术路线

图从RGB输入分出三支：

- Stereo pairs → BM/SGBM → Disparity → Metric depth。
- Single image → UniDepthV2 → Raw metric depth → Frozen global scale → Calibrated depth。
- Video frames → DA-V2-S / VDA-S → Relative-depth comparison。

把795 dev / 654 test画在校准支路；GT箭头只指向开发集拟合和测试评分，不指向测试预测输入。
数据划分的作用比复杂网络结构图更值得在这一页解释。

英文讲稿：

> Our pipeline has three components. First, stereo matching connects disparity with metric geometry on fifteen Middlebury scenes. Second, UniDepthV2 predicts metric depth from a single RGB image. Third, we compare frame-wise Depth Anything V2 with Video Depth Anything on the same video frames. Our current improvement targets the metric branch: we estimate a scale using 795 development images, then freeze it before evaluating the 654 test images.

### 3. 立体基线

| 方法 | MAE（px） | Coverage |
| --- | --- | --- |
| BM | 1.8119 | 59.96% |
| SGBM | 1.9559 | 75.96% |

两项都是15场景像素加权汇总。讲清SGBM覆盖更多像素、BM的有效预测MAE较低，二者评价交集不同，不能单独按MAE宣布胜负。
深度公式放小字`z=fB/(d+doffs)`，单位处理和OpenCV除16放脚注，不逐项朗读。
代表图可选PianoL，既证明实现，也能自然指出弱纹理／重复纹理的困难。

英文讲稿：

> We implemented both block matching and semi-global matching. This example shows disparity, invalid regions, and the converted metric depth. Across fifteen scenes, BM achieves 1.81 pixels of error with about sixty percent coverage. SGBM covers about seventy-six percent, with 1.96 pixels of error. These results must be interpreted together: the methods retain different pixels, so lower error alone does not establish better overall performance.

### 4. 视频基线

素材：`../course_part1_part2/video/rgb_da2_vda.mp4`。原速视频只有约2.917秒；播放一次后冻结在交叉支架附近，配合连续帧局部图讲述。
若为了课堂观察增加重放或慢放，画面明确写Replay或0.25× playback；不能把播放处理当作新增原始帧。
无需专门占一页解释70／100帧差异，脚注说明“official clip re-downloaded: 70 available frames, 24 fps”即可。

英文讲稿：

> These are the same RGB frames processed independently by Depth Anything V2 and jointly by Video Depth Anything. Both recover the track and support structure. Thin bars, intersections, and moving objects remain useful places to inspect instability. We use fixed normalization over the whole clip. These are relative-depth colors, not shared metric distances. The official clip contains seventy available frames at twenty-four frames per second.

### 5. 方向B方法

核心公式：

`s_i = median_valid_pixels(GT_i / prediction_i)`；`s = median_dev_images(s_i)`；`prediction_calibrated = s * prediction_raw`。

主画面只需一个乘法公式和开发／测试隔离图；逐像素及逐图中位数可以口述。
最终`s=0.9340532609`，大约把深度整体缩小6.59%。这仍是米制输出，不能与DA/VDA相对深度混为一谈。

英文讲稿：

> A pretrained metric model can preserve useful structure while having a systematic scale bias. Our calibration estimates one multiplier from development data. For each development image, we take the median ground-truth-to-prediction ratio over valid pixels, then take the median across images. This gives a scale of approximately 0.9341. We apply that same frozen value to every test prediction. No test ground truth is used to fit the scale, and the pretrained network remains unchanged.

### 6. 定量结果

使用方向B同一次运行中的before／after，避免把其他运行或SSI成绩混进表。

| 方法 | AbsRel ↓ | RMSE（m）↓ | δ1 ↑ |
| --- | --- | --- | --- |
| UniDepthV2，未校准 | 0.129425 | 0.423729 | 87.5653% |
| UniDepthV2，全局校准 | 0.103003 | 0.388024 | 90.0662% |
| 变化 | 相对下降20.4% | 相对下降8.4% | +2.50个百分点 |

脚注：NYU官方654张，480×640，规定crop，0.001–10m，逐图等权；先应用冻结尺度再clip，baseline无GT对齐。
“无GT对齐”不等于校准从未使用任何GT；必须明确只在开发集使用GT拟合。

英文讲稿：

> On all 654 held-out test images, AbsRel decreases from 0.1294 to 0.1030, a relative reduction of 20.4 percent. RMSE decreases by 8.4 percent, and delta-one increases by 2.50 percentage points. Both conditions use the same images, crop, depth range, and image-wise aggregation. The only change is the development-fitted scale. This shows that a simple calibration can improve absolute accuracy without retraining the depth network.

### 7. 可视化、失败和局限

优先使用已整理的横向图，而不是将四行全图缩成投影上看不清的小图：

- `assets/direction_b_improvement_0523.png`：AbsRel 0.065974 → 0.023749，Δ=-0.042225。
- `assets/direction_b_degradation_0508.png`：AbsRel 0.117755 → 0.169597，Δ=+0.051843。

两图保留原图共享0.001–10m色标。原始完整四行图及其选样分位也保留在assets中。
显示“457 improved / 197 degraded”，不说每张都改善。
解释：单一尺度只能处理总体偏差；原本偏小的图可能进一步退化；边界、缺失结构不会因乘一个数自动修复。

英文讲稿：

> The improvement is not uniform. Calibration reduces AbsRel on 457 test images but increases it on 197. The first example improves after rescaling, while the bathroom example becomes worse. All panels share the same metric color range. A single multiplier changes absolute scale but cannot reconstruct missing structures or sharpen incorrect boundaries. These examples motivate studying where the global correction transfers and where it does not.

### 8. 下一步与收尾

提出三项具体行动，而不是“用更强模型”或“继续提高精度”：

1. **稳定性：**对开发集按场景分组重采样，报告scale的波动；比较当前稳健中位数和在dev上优化AbsRel的全局尺度，冻结选择规则。
2. **失败归因：**按baseline偏大／偏小、近／远深度和场景类型分析误差；单独记录边界问题，判断剩余误差是否超出全局尺度能解决的范围。
3. **最终交付：**做一种深度感知应用，完成报告与可复现材料；额外数据集上的泛化评价作为可行时的扩展，不提前声称已验证。

最后一句建议：“We have working baselines, a measurable calibration improvement, and a clear plan to study its limitations.”

英文讲稿：

> Next, we will measure scale stability by resampling development scenes and comparing development-only fitting objectives. We will categorize failures to separate global scale bias from local structural errors, including whether the original prediction is too near or too far. Finally, we will demonstrate a simple depth-aware application and consolidate our reproducible results. We already have working baselines, a measurable improvement, and a concrete plan for understanding its limitations.

## 录制前还需补齐的工作

没有必须补跑的技术实验。剩余是制作与提交：

- 做8页16:9 slides，填写项目名、组员姓名、学号；最后一页保留仓库链接。
- 制作总体流程图与B的开发／测试隔离图；从已有素材裁出清晰局部，数字保留3–6位即可。
- 录制或配音，加入原速视频，核实最终导出不超过300秒；目标285秒，预计英文旁白约120词/分钟，实际以排练为准。
- 导出MP4并从头播放检查音量、字幕、嵌入视频与最终时长。
- 确认project selection是否已经提交；在截止前向Canvas提交。当前未检查Canvas状态。
- Discussant安排尚未公布；分配后观看指定视频并准备至少一个技术问题，不需要现在猜测对方项目或将其塞进5分钟主视频。

## 备份页与可能的Q&A

备份页不计入主视频讲解：协议细节、完整四行校准图、原始逐图结果、效率表、方向A研究。

| 可能问题 | 回答要点 |
| --- | --- |
| 是否使用测试GT做尺度拟合？ | 只在官方795张dev拟合，固定一个s，654张test只评分 |
| 为什么用两层median？ | 降低极端像素影响，第二层让每张开发图在中位数统计中具有同等地位；不声称全局最优 |
| 为什么仍有197张退化？ | 全局偏差不等于每图偏差，统一缩小会伤害原本偏小的预测 |
| 能否改善视频闪烁？ | 本轮B针对米制尺度；DA/VDA视频对照是独立实验，不由B的NYU结果推出时间一致性收益 |
| 是否已经证明跨域泛化？ | 已证明NYU官方开发／测试划分上的收益，跨数据集泛化尚未验证 |
| 还需要新的大型模型或训练吗？ | 不是当前checkpoint的前置条件；先完成稳定性、失败归因和应用交付 |

## 源材料

- [Part 1 / Part 2结果](../course_part1_part2/README.md)。
- [视频与基线难例](../course_part1_part2/analysis.md)。
- [方向B远端报告，固定提交](https://github.com/juruohejiarui/Homework-AIAA3201/blob/ad24f94ac0ac9985f9c863f20d69dbcdf74e1cf6/Proj/reports/part3_b/README.md)。
- [方向B原始预测归档](https://github.com/juruohejiarui/Homework-AIAA3201/releases/tag/part3-b-nyuv2-v1)。

本文件是完整内容编排、讲稿和素材整理，不是已经制作完成的PPT或录制好的presentation视频。
