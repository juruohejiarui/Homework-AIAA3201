# 多数据集训练：KITTI、NYUv2 与扩展数据

2026-10-23。训练范围包括现有 KITTI 和 NYUv2，不限定为室内 NYUv2。下列是数据集建议，不代表替你选择 Part 3 方向，也没有自动启动下载或训练。

## 1. 数据来源与用途

体积为下载包大小，解压、预测缓存和训练样本缓存另算。ScanNet/TUM 的数字是可选择的下载预算，不是声称官方全集只有这么大。每项按不超过1TB考虑；若你的1TB限制是全部新增数据的总量，应累计计算。

| 数据集 | 数据特性 | 对训练的作用 | 对课程方向的支持 | 下载规模和状态 |
|---|---|---|---|---|
| KITTI Raw + depth annotated | 室外驾驶、连续 RGB、标定；累积 LiDAR 标注仍然稀疏 | 室外米制监督；Raw 连续帧还可做时序/自监督，但没有标注的帧不能伪装成有 GT | A/D 的时间信息，B 的米制尺度，C 的稀疏监督；E/F 的室外输入 | 已有 Raw 与深度包；深度 annotated ZIP 约14GB，train 可从已有包展开 |
| NYUv2 labeled | 室内、1449对，有标准795/654划分 | 795训练图可训练小头；规模较小，不能代表完整 Raw | B/C/D 的图像监督，E 的室内校准 | MAT约2.97GB，已存在 |
| NYUv2 预处理训练镜像 | 室内 H5；数据卡标称47,584训练图 | 避免自己从全部传感器 Raw 开始准备图像训练；需遵守其原始处理定义 | B/C/D 的图像模块；保留时序元数据后才能讨论视频训练 | HF 14个TAR共35.15GB，新增下载脚本的 prepared 模式 |
| NYUv2 完整 Raw | 原始 RGB/深度流，需同步和投影 | 自行建立室内连续 RGB-D 训练集，工作量主要在预处理 | A/D 的时序学习；B/C 的米制监督 | 38个ZIP共421.54GB；脚本 full 模式 |
| Virtual KITTI 2 | 合成室外连续双目、稠密深度、相机参数；可选双向光流和语义 | 给 KITTI 补稠密边界、遮挡和运动监督；需要评估合成到真实的域差异 | A/C/D 可直接利用额外真值；B/E 也可用 | RGB+depth约15GB量级；增加双向光流再约57GB。以官方各包实际长度为准 |
| DIODE | 真实室内外 RGB、较稠密深度及有效掩码，单图/扫描裁片 | 跨室内外训练，减少仅对 NYUv2/KITTI 单一域适应；25,458张训练图 | B/C/D 图像监督；不把裁片拼成 A 的视频 | 官方 depth train 81GB + val 2.6GB；法线另需130.6GB，不做法线任务可不下 |
| ScanNet v2 的训练场景子集 | 室内 RGB-D 视频、内参和重建相机位姿 | 多视角/重投影训练；注意静态重建场景与动态遮挡的差异 | A/D 的几何一致性；C/F 可选语义标签 | 建议自行设定200–400GB的 .sens 下载预算；不是全集大小。当前仍需官方授权 |
| TUM RGB-D 的序列子集 | 室内连续 RGB-D、时间戳、相机参数与轨迹 | 小规模时序训练或独立验证；场景多样性有限，不宜靠邻帧数量夸大规模 | A/D 的几何与动态实验，B/F 的应用验证 | 可先设20–50GB下载预算，按官方序列页标注体积选包；不需要另下重复 ROS bag |
| Bonn-RGBD | 动态室内 RGB-D，有相机轨迹 | 可按场景划训练与测试；本项目已有论文评价用途，更适合保留独立评价集 | A/D 的动态遮挡，F 的任务稳定性 | 全集16.4GB，已经下载，无需重下 |

新增数据中，Virtual KITTI 2 补充可控的稠密时序真值，DIODE 补充真实室内外单图；ScanNet/TUM 补带位姿的室内视频。它们各自补不同缺口，数据总量并不能替代所需监督类型。无需为了覆盖六个方向把这些数据全部下载。

若按总新增下载量控制在1TB以内，一个容量示例是：NYUv2 Raw 421.54GB + HF训练镜像35.15GB + Virtual KITTI 2 RGB/深度约15GB + DIODE83.6GB + ScanNet选定200GB + TUM选定20GB，合计约775GB。这里不含已有 KITTI/Bonn 等、未必需要重复下载的 labeled、解压输出与缓存；不是必须执行的下载组合。若1TB指磁盘总占用而非下载包，完整 Raw 和展开副本同时保留就需要更谨慎地分配空间。

目前只核实了 NYUv2 的 HF 训练子集与 Bonn 原包镜像。其他候选在本文链接官方来源，不宣称任意 HF/ModelScope 同名仓库都与官方包等价。若将来改用镜像，需先确认其确实含所需 RGB、深度、时间和标定，而不是仅有图片缩略图或不同 split。

## 2. 现有 KITTI 如何投入训练

现有 annotated ZIP 中已经有 train；此前为论文实验只展开了 val。使用系统 unzip 展开训练标注，不重复下载、不重复全量哈希：

```bash
dataset="$HOME/Documents/ResourcesLow/Datasets/KITTI"
unzip -q -n "$dataset/archives/data_depth_annotated.zip" 'train/*' -d "$dataset/depth"
```

展开后为 `KITTI/depth/train/<drive>/proj_depth/groundtruth/image_02|image_03/*.png`。训练 RGB 来自 `KITTI/raw/<date>/<drive>/image_02|image_03/data/`，按 drive、camera、frame ID 对应；只用 RGB 与监督实际相交的帧，不能把所有 Raw 帧都当成带深度标签。

同一网络可以只输入单张 RGB，用 annotated 深度监督；这仍是 monocular depth estimation。只有将稀疏深度也作为网络输入时，才变成 depth completion。有效 GT 区域计算监督损失，0值为空洞；不要将0当真实距离，也不要随意填满 GT 后当成传感器测量。

现有 KITTI 论文评价的13个 val drive 应继续保留为评价。训练划分按 drive，而不是随机拆同一段视频的帧。若改用 Eigen split，要使用对应的固定帧清单并排除重叠；不能把 Eigen 与 depth-completion split 的指标放在同一列。

使用 KITTI 训练后再测 KITTI，属于有域内训练的结果。原冻结模型的论文复现继续保留；不要把微调模型在 KITTI 的结果继续叫作对 KITTI 的零样本复现。NYUv2 和其他数据也遵循相同区分。

## 3. 多数据集训练真正增加的工作

| 工作 | 具体要做什么 | 主要消耗 |
|---|---|---|
| Dataset adapter | 各数据集各自读取 RGB、深度、有效掩码、内参和可选位姿；统一输出米制浮点深度 | CPU、工程时间 |
| 单图/序列采样 | 图像小头可混合单图；时序网络只能从真实连续序列中取窗口，不能跨场景拼接 | 数据清单、I/O |
| 深度与坐标统一 | 分清相机 Z 与沿视线距离；缩放/裁剪时更新内参；位姿方向保持一致 | 几何处理 |
| 深度范围 | NYUv2 常用10m、KITTI 常用80m是具体协议裁剪；按数据集保留训练有效范围，不把所有数据硬裁成10m | 配置和 loss mask |
| 数据集平衡 | 显式控制不同来源的采样概率与 loss 权重；不要让 Raw 的大量相似邻帧淹没其他域 | 训练设计 |
| 开发集与测试集 | 按场景/drive划分；训练源与域外测试源分别列出；选择参数只用开发集 | 实验管理 |
| 缓存 | 冻结模型时缓存预测/特征，后续只训练轻量模块；缓存记录源模型、尺寸、单位与帧 ID | 首轮推理、磁盘容量 |
| 评价 | 分别报告室内、室外及每个数据集，另列统一汇总方式，不直接平均所有像素 | GPU推理、结果分析 |

常见单位：KITTI annotated PNG 除以256得到米；NYUv2 labeled 的 `depths` 已为米；NYUv2 Raw PGM 需要官方传感器转换和投影，不能套用同一常数；Virtual KITTI 2 深度 PNG 为厘米，除以100；TUM 16位 PNG 除以5000。DIODE 使用原始 NPY 浮点深度与 mask，接入重投影前还需从作者工具确认距离定义及内参，不能凭扩展名假定为 Kinect Z。

Virtual KITTI 2 的相同地点在不同天气/视角下仍共享几何，应按地点划分，不能把天气变体随机拆成训练/测试；它源自真实 KITTI 跟踪场景，使用前也应核对与真实评测场景的对应关系。ScanNet 同一空间的不同 scan 应归在同一划分。NYUv2 Raw 中测试场景的邻帧不能进入训练。

## 4. 对 GPU 资源的影响

多加数据集通常增加训练步数、读取和总时间，**不会仅因磁盘数据变多就增加单步峰值显存**。单步显存主要取决于可训练模型、输入大小、batch、序列长度、优化器，以及是否保留基础模型反向计算。

- **冻结基线、训练小头：**可离线缓存室内/室外预测；训练压力主要来自小网络，数据规模主要影响总步数。不要把全数据缓存同时加载到 GPU。
- **基础模型微调：**需要保存编码器反向激活与优化器状态；不同数据域可以交替 batch，不必同时常驻多个基础模型。
- **时序联合训练：**连续 T 帧的反传通常远重于单图，光流若也参与训练还要另计。短窗口、冻结光流、截断历史与完整长视频反传是不同预算。
- **高分辨率：**KITTI 横向很宽，DIODE 为1024×768；若直接保留原图，显存与时间可能明显高于480×640。改变尺寸时应同步内参与评价说明。

规划人力时，已有适配器可复用的数据通常只需整理 split；NYUv2 全 Raw 的同步投影、ScanNet `.sens` 导出，以及新增数据的坐标与有效 mask 定义才是主要数据准备工作。完整下载脚本不等于这些训练适配器已经实现。A–F 的方法工时和训练资源估计见 [Part 3 指南](part3_guidance_zh.md)。

## 5. 来源

- [KITTI depth 官方说明](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) 与 [Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php)。
- [NYUv2 官方](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[训练镜像](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2)，下载见 [NYUv2 指南](nyuv2_download_zh.md)。
- [Virtual KITTI 2：下载、格式、厘米单位和光流](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)。
- [DIODE：官方分包体积、划分和数据格式](https://diode-dataset.org/)。
- [ScanNet：授权、.sens 和场景组织](https://github.com/ScanNet/ScanNet)。
- [TUM RGB-D 下载](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)、[格式、深度比例与轨迹](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)。
- [Bonn 动态 RGB-D](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)。
