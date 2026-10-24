# Direction A：相邻帧可训练深度融合器（设计草案）

本目录位于项目 `Proj/src/explore_a`。当前仅建立设计，不代表模型、数据适配器或训练入口已经实现。未启动下载、训练或推理。

ScanNet 子集下载器已单独实现，执行方式和官方/本地校验边界见 [下载与校验说明](SCANNET_DOWNLOAD.md)。这不代表融合器或训练入口已实现。

当前用户要求下载 **v1 完整发布数据**，使用 [v1 全集入口与说明](SCANNET_V1_DOWNLOAD.md)，不再按20场景子集执行。

## 1. 当前假设与范围

- previous depth map 暂按紧邻上一帧解释，等待老师澄清。
- 第一版使用上一帧的原始预测，不递归使用融合结果；每个样本为同一相机、同一序列的真实相邻帧对。
- 冻结基础深度模型和光流模型，只训练融合器。RAFT 与 Farneback 为可切换后端，不需要同时串联。
- 已确定三种融合结构全部纳入计划，遮挡处理分规则与模型两个版本；RAFT/Farneback 交叉组合。训练数据确定使用 KITTI Raw + annotated train 与 NYUv2 Raw，可抽取子集。底层深度基线与具体抽样预算尚未选定。
- 最新实验矩阵、confidence 定义、抽样方案与 ScanNet 镜像调查见 [实验计划](EXPERIMENT_PLAN.md)；本文其他结构描述保留为设计背景，以实验计划中的已确认决策为准。

## 2. 各组件职责

| 组件 | 输入与输出 | 权重需求 |
|---|---|---|
| 深度模型 | RGB → 深度/相对逆深度 | 复用已发布权重，冻结 |
| RAFT | 两帧 RGB → 二维光流 | 有官方/torchvision 预训练权重，冻结 |
| Farneback | 两帧 RGB → 二维光流 | 传统算法，不需要权重 |
| warp | 待采样图与坐标 → 对齐后的图 | 重采样操作，不需要权重 |
| 历史有效性判断 | 双向流、越界和预测有效性 → mask | 第一版用规则，不需要权重；这是可信对应代理，不是真实遮挡标签 |
| 融合器 | 当前与对齐历史的信息 → 历史融合权重 | 本项目训练 |

## 3. 样本与对齐

原始样本为 `(I_prev, I_cur, G_cur, valid_gt_cur, sequence_id, frame_ids)`；有前帧 GT 时另存 `G_prev` 与其有效 mask。GT 仅用于训练监督/评价，不作为融合器输入。

冻结模型生成 `D_prev, D_cur`；RGB 生成前向流 `F(prev→cur)` 与后向流 `B(cur→prev)`。在当前像素 x 处，用 `x + B(x)` 采样上一帧，得到 `D_warp` 与 `I_warp`。不能直接把前向流当作反向采样坐标。

前后向残差为 `B(x) + F(x+B(x))`，结合坐标是否越界、预测是否有限/有效得到 mask M。残差不是完美遮挡检测，阈值在开发集确定。

可复用 `depth_project/evaluation/flow.py` 的双向 Farneback、warp 与有效性判断思路；训练模块应独立组织，不将评价入口变成训练入口。

## 4. 建议输入：12 通道

所有量在当前帧坐标系，深度编码及误差特征使用训练时确定的尺度规范。

| 特征 | 通道 |
|---|---:|
| 当前 RGB | 3 |
| 对齐后的上一帧 RGB | 3 |
| 当前预测深度编码 | 1 |
| 对齐后的上一帧预测深度编码 | 1 |
| 两份深度编码的绝对差 | 1 |
| 历史有效 mask M | 1 |
| 前后向光流一致性残差的标量幅值 | 1 |
| RGB 对齐残差的标量幅值 | 1 |
| 合计 | 12 |

米制路线可使用固定规范的 log-depth 作为输入编码，最终仍融合米制预测。无效历史以安全占位值填充，并显式传入 mask。可选增加光流 2 通道、深度梯度或图像特征；每个新增信息源应有对应消融。

Depth Pro 是现成米制基线；VDA metric 必须配套 metric 权重。DA-V2 与相对 VDA 输出相对逆深度，不能直接当米制深度混用。若研究相对路线，须另定义训练目标及帧间尺度处理，不能逐帧用测试 GT 对齐后再融合。

## 5. 建议结构

第一候选是轻量 U-Net：12 通道输入，三尺度编码器（如 16/32/64 通道）、跳跃连接解码器、1×1 卷积和 sigmoid，输出 H×W 的历史权重 a。通道宽度为初始设计，没有实测资源承诺。

`w = M * sigmoid(logits)`

`D_out = (1-w) * D_cur + w * D_warp`

无有效历史时回退当前预测。首帧和跨场景位置也回退当前预测。训练只更新 U-Net。

更简单的结构对照为浅层 CNN；后续可研究“门控 + 小残差头”。纯门控输出受两份候选深度的逐像素区间约束，不能纠正两者共同的尺度偏差。残差头能突破该限制，但须与纯门控分开消融。

## 6. 训练目标与限制

- 基础目标：当前 GT 有效像素上的鲁棒深度损失，例如 log-depth 的 Smooth L1。它可直接通过融合公式训练门控，不必另制作权重标签。
- 可选边界/梯度损失只在有效 GT 邻域计算。
- 单独的逐帧监督不保证更好的时序表现，必须评价时序与深度精度的共同变化。
- 需要时序训练损失时，应另外构造带前后帧监督的连续样本；匹配预测与 GT 的真实时间变化，并使用可靠对应 mask。不能简单强迫 `D_out_cur = warp(D_out_prev)`。
- 光流只对齐二维位置，不修正相机/物体运动造成的真实 Z 变化。该限制应通过运动场景评价，必要时让门控拒绝历史；几何重投影属于另一个可研究扩展。
- VDA 离线输出可能包含未来帧信息；融合器只用前帧不等于完整系统是因果在线系统。

## 7. 训练数据候选

| 数据 | 相邻帧融合训练用途 | 注意事项 |
|---|---|---|
| KITTI Raw + annotated train | 连续 RGB + 当前帧稀疏米制 GT；复用现有资产的室外路线 | 按 drive 划分；GT 只覆盖有标注帧和有效像素；保留当前评价 drives |
| NYUv2 Raw | 室内 RGB-D 连续样本 | RGB/深度同步、传感器转换和投影；训练场景避开官方测试场景 |
| Virtual KITTI 2 | 合成连续 RGB、稠密 GT；可选 GT flow 用于对照 | 按地点划分；同地点天气变体不跨训练/测试；训练主要输入仍用预测流，GT flow 仅作理想对应对照 |
| ScanNet | 室内连续 RGB-D、内参与位姿 | 授权、导出、RGB/深度对齐；同一物理空间不得跨划分 |
| TUM RGB-D | 小规模真实室内序列 | 时间戳关联、有效深度、序列划分；规模和多样性有限 |
| Bonn-RGBD | 动态场景评价候选 | 现有论文复现使用的数据优先保留为独立评价，不直接混入训练 |
| NYUv2 labeled / 预处理图像镜像 | 单图监督；若能追溯 Raw，可将 labeled 训练图配其真实上一帧 | 独立图片不可按文件名拼相邻帧；镜像须先确认时间元数据 |
| DIODE / Middlebury 静态双目 | 不作为本方案直接视频配对训练源 | DIODE 单图与左右双目对都不等同时间相邻帧 |
| 自录 RGB 视频 | 部署与时序评价 | 没有独立深度真值时，不能用于上述 GT 监督；不能用低闪烁证明米制准确 |

数据选择依据：`../../multi_dataset_training_zh.md`、`../../nyuv2_download_zh.md`、`../../part3_guidance_zh.md`。文档记录已有 KITTI Raw/深度包、NYUv2 labeled 和 Bonn；完整 Raw 是否下载/处理完、KITTI train 是否展开尚未在本次核实。

## 8. 工作流程与对照

按序列划训练/开发/测试 → 冻结一种深度基线 → 预计算预测、双向流和配对清单 → 缓存特征 → 训练融合器 → 在完整保留序列上评价。

首轮对照：原始预测、固定权重融合、带有效性 mask 的固定融合、学习式融合。模型结构、输入特征、RAFT/Farneback 切换分别控制变量，不混为一次改动。

同时报告米制/相对协议下的精度、边界、光流时序误差、尺度变化、延迟与内存；检查运动拖影、遮挡、新出现区域与场景切换。正式时序评价的对应估计最好独立于所优化的输入光流，并结合有 GT 的精度，避免仅对同一 warp 误差进行优化。

## 9. 官方资料

- RAFT 与权重：https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html
- Farneback：https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- KITTI：https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
- NYUv2：https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Virtual KITTI 2：https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
- ScanNet：https://github.com/ScanNet/ScanNet
- TUM：https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
