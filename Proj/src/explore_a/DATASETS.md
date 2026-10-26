# 方向 A：配置启用的数据与实际训练数据

2026-10-24。本次核对时，三来源默认配置尚未生成训练清单，也没有已完成的训练。以下“启用”指代码配置，不表示数据已下载、预处理或训练完成。运行 `bash Proj/explore_a.sh status` 可查看当前真实状态。

## 当前配置

| 配置 | 启用的数据源 | 输出位置（Proj下） |
|---|---|---|
| **默认 `mixed.yaml`** | **KITTI＋TUM RGB-D＋NYUv2 Raw** | `outputs/explore_a/nyu_tum_kitti` |
| `kitti_tum.yaml` | KITTI＋TUM，NYU未完成时使用 | `outputs/explore_a/kitti_tum` |
| `kitti.yaml` | KITTI单来源，保留此前配置 | `outputs/explore_a/kitti` |
| `mixed_scannet.yaml` | 上述三来源＋ScanNet，保留此前四来源配方 | `outputs/explore_a/mixed` |

ScanNet不再是默认训练的前置条件。显式请求某个来源时，其数据缺失或没有可用训练/开发样本会报错，不静默省略该来源。

## 各来源具体使用什么

| 来源 | 训练数据 | 开发集与held-out test |
|---|---|---|
| NYUv2 | 全部Raw场景中符合划分的真实相邻RGB，以及同步、投影后的传感器深度；不限定为labeled或预处理子集 | 654个官方测试索引对应的完整房间、同房间a/b扫描、已有八个评价场景和其他历史评价场景保留test；剩余房间约15%开发、其余训练 |
| TUM RGB-D | 全部普通序列中划为训练的相邻RGB-D，默认Freiburg 2 | 默认Freiburg 1开发、Freiburg 3测试；已知同地点变体通过group映射归同一划分；validation/标定包不参与首轮训练 |
| KITTI | Raw连续RGB和annotated train深度的全部有效交集，含两路彩色相机；无GT当前帧不能伪装为监督样本 | 已有13个评价drive整体保留test，包含另一相机和导出副本；剩余训练drive约15%划开发 |
| ScanNet（可选） | 完成下载的v1 `.sens`，按官方train及场景组筛选 | 官方val和历史评价空间保留test；同一`sceneXXXX`重扫描不跨划分；其余训练空间划开发 |

所有来源先按drive/房间/物理场景划分，再构造帧对。GT不作为模型输入，也不使用test选择checkpoint；开发集仍负责选择最佳模型。

## “全部Raw”的含义

完整下载覆盖官方Raw分包；训练池遍历所有允许场景和所有相邻RGB锚点，不设抽样上限。跨场景、时间间隔过大、无法同步深度、缺少文件或无效测量不作为有效监督。训练阶段仍为每组30,000步、数据源等概率、有放回采样；全量候选池不等于保证遍历每帧一轮。

`nyu_depth_v2_labeled.mat`含1449张处理后的图像，是Raw的子集。当前不会仅拿795张labeled训练图代替Raw，也不会将它们再重复添加一遍。MAT及 `splits.mat`用于定位并排除654测试图所属完整房间；当前Raw损失使用真实传感器有效深度，而不是MAT中额外填洞的depths。

准备阶段的 `data.json` 保存每来源train/dev/test帧对数、NYU场景组数和同步剔除计数。实际下载完成前不预报一个未经计算的最终训练样本数。

所有训练及评价均直接读取原始数据并即时预测，固定清单不包含预计算预测文件。
