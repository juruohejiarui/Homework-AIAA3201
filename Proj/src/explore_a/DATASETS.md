# 当前三源数据与固定划分

三个 DA 模型共用 `outputs/explore_a/mixed_h5_balanced/data.json`。
本次只精简配置和入口，没有修改清单或重新划分。训练数据位于
`~/Documents/Resources/Datasets`，共享预测缓存另在 ResourcesLow。

| 来源 | train 帧对 | 完整 dev 帧对 | held-out test 帧对 |
|---|---:|---:|---:|
| KITTI | 68,470 | 17,428 | 6,852 |
| TUM | 47,554 | 8,514 | 21,542 |
| NYUv2 H5 | 38,420 | 7,466 | 1,418 |

训练实际使用固定 10% dev 子集；按 KITTI:NYUv2:TUM=3:3:2 抽来源，
来源内按视频帧对数平方根抽序列，再抽帧对。完整候选池不等于每轮遍历。

KITTI 是 Raw 连续 RGB 与有效深度的交集；历史评价 drive 整体隔离。
TUM 当前固定清单仍是 Freiburg 2 训练、Freiburg 1 dev、Freiburg 3 test；
这描述现有清单，并非建议依据相机编号重新划分。官方 validation 单独用于视频评价。
NYUv2 使用 H5 的 RGB 与米制 GT，不需要额外 Raw 来获取 RGB；在同一场景内使用
数字排序的相邻保留帧，通常编号差 5，无可靠时间戳。物理房间及官方测试房间隔离保持不变。

NYU labeled 654 张官方测试图另作单图评价，不按编号拼成视频，不参与训练或选模。
ScanNet 读取器保留为可选数据工具，不属于当前训练或四个视频 benchmark。
GT 只用于监督与评分，不进入融合输入、预测缓存或 DGX 请求。

[操作与配置](DA_TRAINING.md) · [NYUv2 H5](NYUV2_H5_GLOB.md)
