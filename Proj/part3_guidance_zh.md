# Part 3 指南：轻量改进、可控消融与应用

本文件依据课程 PDF 第 5.3 节，以及第 2、6 节的完整性、评价与公平比较要求。它是接下来开展 Part 3 的方案，不是已经完成的创新实验，不承诺分数一定提高。

## 1. 课程到底要求什么

Part 3 必须完成。需要识别复现系统的一个**可测量局限**，实现轻量改进，提供受控消融和失败案例。即使主指标不提升，也应分析原因；忽略 Part 3 会被判为 Incomplete。

课程给了六种可选方向，并不要求全部做，也不强制按建议路线实现：

| 方向 | 主要工作 | 必须量化的内容 | 主要风险 |
|---|---|---|---|
| A Flow-guided fusion | 双向光流、遮挡剔除、置信融合 | 光流对齐 MAE、尺度变化、边界/拖影、开销 | 传播错误；移动相机真实 Z 改变 |
| B Metric scale calibration | 已知物体、相机高度、地面或稀疏深度标定 | 米制误差与标定噪声鲁棒性 | 测试 GT 泄漏、室内假设失效 |
| C Edge-aware refinement | RGB/语义边界引导深度 | Depth-edge P/R/F1、深度误差 | 把颜色纹理复制成假深度边缘 |
| D Uncertainty-aware smoothing | TTA、置信或模型分歧决定平滑区域 | 稳定性-细节折中、置信校准 | 置信不准；TTA 大幅增加耗时 |
| E Efficient deployment | ONNX、量化、动态分辨率等 | 精度、速度、显存 Pareto 曲线 | 只降低分辨率未分析退化 |
| F Depth-aware application | 距离排序、遮挡、bokeh、碰撞提示等 | 下游任务准确率/误差/时延 | 只演示好看的效果，没有任务评价 |

最终提交仍需要 RGB、depth、应用输出三者齐全的 demo。仅导出一个深度视频不能替代应用评价。

## 2. 建议主线：A，带最低限度的不确定性门控

当前 Part 1/2 已跑通，实际结果见 FIRST_RUN_zh.md。官方短视频上 VDA 相对 DA-V2 的 光流对齐 MAE 降低约11.7%，因此 Part 3 应继续检查 VDA 残余的边界、遮挡和快速运动问题，而不是把复现 VDA 本身当作创新。该视频不是自录素材，不能替代正式评价片段。

在现有 DA-V2 / VDA 结果基础上，优先尝试：

**前后向光流可靠性 + 深度变化门控 + RGB 边界门控的时间融合。**

优点：不训练基础模型；复用已有 OpenCV 光流和评估；参数少、消融清楚。首轮先使用 VDA 原始输出作为 D_t，也可以在 DA-V2 上做补充实验，判断方法对不同 baseline 是否有效。

要先从首轮结果找证据：VDA 在快速运动、遮挡切换、细结构附近仍有什么具体不稳定。不能先确定“融合一定更好”，然后只挑选有利片段。

强 baseline 已经稳定时，改进可能不明显甚至变差，这正是需要报告的结果。

## 3. 数学定义与实现顺序

### 3.1 统一表示与保留基线

- Metric：D_t 为米制 optical-axis Z。
- Relative：保留模型原始 inverse-depth-like q_t；没有可信标定时不把它叫作米制深度。
- 不为每帧单独对齐 GT，不为追求低 光流对齐 MAE 做逐帧尺度重标定。
- 评估基线与融合结果时共用**从未修改的 baseline 确定的一组整段 gauge 参数**。不要分别根据融合后结果重新估计 P1/P99，否则改变的归一化范围会污染分数。
- 保存 baseline raw、fusion raw、光流 mask、confidence、参数配置与精确源帧索引。

### 3.2 正确的 warp

对前帧 t-1 与当前帧 t：

~~~text
F: t-1 -> t
B: t -> t-1
W(D_(t-1))(x) = D_(t-1)(x + B(x))
~~~

在当前坐标使用 backward flow 去前帧采样。不要把 forward flow 直接加到当前网格。

`depth_project.evaluation.flow.warp_and_valid()` 返回 warp 函数和前后向一致性掩码，可以复用。光流必须由相同 RGB 帧计算，不能从待融合深度推导它的运动然后自证稳定。

### 3.3 遮挡与置信门控

~~~text
M_bounds = 采样点位于图像内
M_fb     = ||B + W(F)||² <= 0.01 (||B||² + ||W(F)||²) + 0.5

r_D      = |D_t - W(D_(t-1))| / (|D_t| + |W(D_(t-1))| + epsilon)
c_D      = exp(-r_D / tau_D)

c_RGB    = exp(-|I_t - W(I_(t-1))|_mean / tau_RGB)
c_edge   = 1 - normalized_RGB_edge_strength

c        = M_bounds * M_fb * c_D * c_RGB * c_edge
w_prev   = alpha * clip(c, 0, 1)
D_fused  = (1 - w_prev) D_t + w_prev W(D_prev_fused)
~~~

RGB 归一化至 [0,1]，所有阈值必须与单位匹配。边界归一化尺度从独立开发片段确定并固定；不要每帧挑一组能让输出更稳定的阈值。

第一帧直接使用 D_0。新增区域、遮挡、场景切换时优先当前预测。若图像变化很大、有效光流比例过低或时间间隔异常，应 reset，避免把旧场景传播进来。

建议先固定：

~~~yaml
alpha: 0.3
tau_depth_relative: 0.1
tau_rgb: 0.1
reset_valid_fraction: 0.1
use_forward_backward_mask: true
use_depth_gate: true
use_rgb_gate: true
use_edge_gate: true
~~~

这些是开发起点，不是已验证的最佳参数。Metric 米数门限与 relative arbitrary-unit 门限不可混用。相对输出带有 shift 自由度时，r_D 的数值也受 gauge 影响；应先确定一个固定、可复现的整段表示，不用每帧重新归一化。

### 3.4 移动相机的几何限制

像素对齐不保证 optical-axis Z 不变：相机前后运动会让同一世界点的 Z 真实改变。强行平滑会造成“拖在后面”的距离估计。

本方案不声称是完整 3D motion compensation。因此：

- 主定量诊断优先静态相机或受控纯旋转片段。
- 移动片段同时报告拖影、遮挡恢复速度和尺度变化代理。
- 不把 moving clip 上的 raw 光流对齐 MAE 当作真值深度准确率。
- 若要进一步改进，应引入独立位姿与内参，把前帧点云变换到当前相机坐标并重投影，再处理 Z-buffer/遮挡；这会增加实现成本，不是第一轮必需。

## 4. 如何设计公平的消融

固定同一视频、同一源帧、分辨率、checkpoint、光流、色标/gauge、评估掩码、计时范围。不能每个方法各删掉一批困难像素。

至少准备以下行：

| ID | 方法 | 用途 |
|---|---|---|
| B0 | 原始 VDA | 必须保留的主 baseline |
| A0 | alpha=0 的融合程序 | 验证代码路径恒等性，应与 B0 数值一致 |
| A1 | 不做 motion warp 的 EMA | 证明 warp 是否有必要；容易产生拖影 |
| A2 | flow warp + 固定权重 | 检查运动补偿本身 |
| A3 | A2 + forward-backward mask | 检查遮挡剔除 |
| A4 | A3 + depth/RGB gate | 检查置信门控 |
| A5 | A4 + edge gate | 完整方法；检查细节保持 |

另做 alpha ∈ {0, 0.1, 0.3, 0.5, 0.7} 的曲线。参数在独立开发片段上选择，再冻结用于评价片段；不能在结果最好的测试片段上选参数，然后报告同一片段的无偏性能。

至少增加两条机制测试：

1. 遮挡 mask 全 false 时，结果应等于当前帧。
2. 第一帧、reset 帧不应引用上一个场景的历史。
3. 可选：同一输入重复多帧不应出现凭空漂移；此测试是 sanity check，不是实际视频证据。

## 5. 数据如何准备，哪些不能混为一谈

### 5.1 真实自录视频

推荐分开发片段与评价片段，分别记录：

- static：相机固定，静态场景；可增加光照变化作为另一个难度条件。
- moving：缓慢平移/纯旋转/回到起点，尽量含已知固定物体。
- occlusion：人或物体经过前景，观察被遮区域恢复。
- thin/specular：细杆、镜面、玻璃。
- rapid/blur：快速转向与运动模糊。

PDF 没有给具体片段时长、数量；这些是实验设计建议。若 Canvas 另有 mandatory clips，必须覆盖其要求。

### 5.2 目前的官方演示素材

仓库自带的 demo 可以用于系统调试、比较与计时，必须在报告中标明来源。它不等于你们自录的 static/moving 视频，也不能伪称已满足全部课程素材要求。

### 5.3 NYUv2 是固定图像 benchmark

标准 labeled 的 654 测试图不是一个按文件名排列就能得到的连续视频。

**禁止把这些图随机或按 ID 串起来做 temporal fusion，然后报“NYU 视频精度”。**

图像米制主表继续使用 Part 2 的 frame-wise 预测。如果要给 Part 3 加有 GT 的受控实验，可以从单张 RGB-D 做已知的平面内平移/旋转生成序列，对 RGB 与 depth 同步变换，评价共同有效区：

- 这是合成的平面内图像运动实验，Z 值按构造保持。
- 不等价于真实 6DoF 相机运动，不具备完整新视角遮挡。
- 深度插值要避免跨边界线性混合；说明插值与有效 mask。
- 明确区分真实视频结论与这种 sanity/controlled test。
- 如果只给预测深度额外加噪声，用于机制分析可以，但不能冒充模型原本的误差分布。

## 6. 指标必须同时约束“稳定”和“正确”

推荐主表：

~~~text
Method | 光流对齐 MAE ↓ | valid ratio | median-scale proxy |
Boundary P/R/F1 ↑ | ghosting/recovery measure |
ms/frame ↓ | peak MiB ↓
~~~

有 GT 的受控实验加 AbsRel、RMSE、δ1；真实无 GT 视频不编造米制准确率。

除 光流对齐 MAE 外，建议：

- 固定 ROI 时间标准差（静态相机且 ROI 没有真实运动）。
- 遮挡结束后恢复到当前预测/参考值的帧数。
- 深度边界的空间扩散宽度或边界 F1。
- 运动边界前后残影幅度。
- 回到相同相机姿态时固定物体深度偏差；同时说明姿态误差。
- 应用任务的时延/正确率，而不仅是彩色视频。

常数深度能得到很低 光流对齐 MAE，因此只报告 光流对齐 MAE 改善不足以证明方法更好。必须展示稳定性改善没有靠抹掉物体和边界获得。

评估 mask 应主要由 RGB 双向光流与公共 GT 有效区域决定，不能用完整方法自己的置信阈值把难例删掉再与 baseline 比。

## 7. 应用 demo：优先选一个可测量的任务

### 可选 A：目标距离排序

- 自录至少数个不同距离的物体，独立量测真实距离排序。
- 固定 ROI 或使用明确的目标 mask，取有效深度的鲁棒中位数。
- 评价 pairwise ranking accuracy、相邻帧排序翻转次数和 latency。
- Relative 模型可做排序，但不能宣称精确米制测距。
- Metric 模型可同时报告距离 MAE；若 ROI 对应物体表面，真值定义也应是该表面，而不是物体中心。

### 可选 B：有真值参照的点击测距

- 使用 metric 模型、可信 K、当前图像尺寸下的像素坐标。
- Z 是 optical-axis depth；相机到点的欧氏距离是 sqrt(X²+Y²+Z²)，两者不同。
- 若测两点距离，应先各自反投影到 3D 再算欧氏距离，不能直接相减 Z。
- 报告不同距离、不同角度、透明/反光物体的误差。
- Part 2 原始 metric 结果与 Part 3 使用外部标定的结果分组，不能混淆“无校准”与“有参照校准”。

### 可选 C：碰撞提示

- 预先确定距离阈值、连续帧确认次数与测试路线。
- 评价 precision/recall、误报警率、警报延迟。
- 这只是课程实验 demo，不是经过安全认证的防撞系统；不要用危险的真实接近实验收集数据。

最终画面至少同时显示 RGB、depth、应用状态/3D 输出；保留 raw 数据便于复查。

## 8. 如果主指标没有改善，报告怎么写

可以分析以下情况，但必须用数据支持：

- VDA 本身已很稳定，额外融合带来的改善小。
- Farnebäck 对运动模糊、无纹理、镜面不可靠。
- 正确的快速深度变化被误判为不稳定，从而产生 lag。
- 遮挡剔除过于保守，实际可融合像素太少。
- alpha 太大抹掉薄结构；太小几乎无效果。
- Relative gauge、可视化归一化、帧索引或光流方向错误造成假改善/假退化。

最低交付仍是完整实现、baseline、受控消融、失败分类、效率和应用证据。不要删掉负结果，也不要在没有显著差异时写“明显优于”。

## 9. 推荐文件与命令规划

现有可直接复用：

- `depth_project.data.video`：统一帧。
- `depth_project.inference`：保留 baseline raw。
- `depth_project.evaluation.flow.warp_and_valid`：公共双向光流。
- `depth_project.evaluation.metrics`：深度及边界指标。
- `depth_project.visualization.video`：固定色标；用于 Part 3 时需把 baseline gauge 复用给融合结果。

下一阶段建议新增（以下是规划，当前不能当作已存在的 runnable CLI）：

~~~text
part3_fusion.py
configs/part3_fusion.yaml
tests/test_part3_fusion.py
outputs/part3/<clip>/<ablation>/
outputs/part3/tables.csv
~~~

part3_fusion.py 应接收 frames、raw prediction、alpha、门控开关、reset 配置，输出 fused NPZ、confidence/mask、时间统计与 JSON 参数。第一步先做 alpha=0 恒等性验证，再跑单段短 clip，最后跑冻结参数的所有评价片段。

## 10. 报告与提交检查

- 明确指出“我们在原始系统上发现的问题”，展示 baseline 证据。
- 用自己的话说明方法，标清光流方向、融合坐标系、遮挡与 reset。
- 画一张清晰 pipeline 图：RGB pair -> flow+mask -> warp previous -> confidence -> fuse current -> application。
- 消融表、alpha 曲线、局部边界放大、遮挡恢复、失败例和 runtime 俱全。
- 说明数据来源、预训练权重、代码复用、后端/精度模式与 submodule 补丁。
- 对没有真值的尺度代理、合成实验局限、相机运动限制保持准确表述。
- README 可从数据与权重一步复现；最终视频、应用输出打包 depth_results.zip。
- 正文 6–8 页、CVPR camera-ready、公开 GitHub、摘要末尾仓库链接、arXiv 与 Canvas 首页 ID 等提交要求仍适用。

技术参考：课程 PDF §5.3/§6；[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)；[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)。本指南的融合公式与消融方案属于建议实现，而非课程提供的唯一算法。



术语澄清：本方案沿用首轮开发的光流对齐 MAE；论文的三维重投影 TAE 由 `depth_project.evaluation.tae` 调用官方 evaluator 单独评价，不能用本方案的光流值填论文表格。
