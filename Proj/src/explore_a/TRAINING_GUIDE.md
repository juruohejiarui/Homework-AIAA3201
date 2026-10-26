# 方向 A：训练与评价

当前推荐consumer_hybrid.yaml：模型输出逐项复用，缺失项即时计算且不写回。
超过4,000 RGB帧的视频只跳过新增缓存，仍完整参与训练、开发和最终test。

冻结Depth Pro、RAFT和MFTIQ，只训练融合头及遮挡学生。支持online直接读取原始数据运行冻结模型，或cache读取DGX生成的FP32模型输出。
缓存生成、传输、环境及独立结果目录见[DGX_CACHE.md](DGX_CACHE.md)。当前本机RTX 4090 D 48GB用于运行；真实峰值由日志记录。

## 启动与模块

环境、KITTI＋TUM起步命令见[KITTI_TUM_START.md](KITTI_TUM_START.md)。
权重由用户下载，地址和目标位置见[MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md)。
单组训练按需检查并加载模型：Farneback规则融合仅需Depth Pro；RAFT融合增加RAFT；
MFTIQ融合及蒸馏增加UOM、ResNet、DINOv2；学生融合使用已训练的冻结学生。
`check-weights`独立命令仍检查完整矩阵所需全部资源，不执行推理。

| 模块 | 职责 |
|---|---|
| `data/` | 数据集适配、GT单位与同步、固定划分、历史评价排除 |
| `online/` | CPU原始帧读取、有界预读、冻结模型即时计算、数值来源记录 |
| `geometry/` | 反向warp、边界、前后向一致性、12/13通道特征 |
| `models/` | 三种融合器、遮挡学生、本地冻结模型适配 |
| `training/` | 等概率来源采样、优化、恢复、开发集选择、test监测 |
| `evaluation/` | 精度、边界、时序、独立/共享评价、异步结果保存 |
| `execution.py` | 有界预读和结果写入队列，异常传播 |

单组与矩阵入口保持薄调度。完整流程为检查权重、准备/验证manifest、训练两个学生、
训练18组融合器、共享评价、汇总。既有manifest复用，不再次prepare：

```bash
bash Proj/explore_a.sh --config Proj/configs/explore_a/kitti_tum.yaml all
```

## 数值与训练

previous使用真实前一帧的原始Depth Pro深度，不递归使用融合结果。
冻结Depth Pro为单帧FP16，输出转为FP32处理，不经过中间文件的FP16量化。
RAFT Large C_T_SKHT_V2为FP32、12次更新，正反向batch 2；OOM后回退逐方向。
Farneback沿用原参数。MFTIQ保持FP32，只在需要它的方案运行。
训练抽样不保留历史预测；有序评价最多保留紧邻上一帧的原始深度，跨场景/相机清空。

融合器使用12通道、log-depth和两个相对confidence logits；CNN三层32通道，
U-Net为16/32/64；残差版修正最多±10%。学生13通道、16/32/64 U-Net，
按光流分别蒸馏MFTIQ软标签，称为历史可用性估计，不是真实遮挡概率。

KITTI画布256×832，室内384×512；等比例缩放和padding，内参同步变换。
KITTI GT除以256，TUM除以5000；稀疏GT前向投影取最近表面，不填洞。
NYU Raw使用传感器转换及RGB投影；ScanNet读取.sens的depth_shift及标定。
GT只进入损失或指标，不进入模型特征。

每组60,000步、有效batch 8、有放回等概率数据源采样、种子2026。
AdamW学习率1e-4、weight decay 1e-4、warmup 500后cosine、梯度裁剪1、FP16 AMP。
有效GT上log-depth Smooth L1 beta=0.1；残差幅度正则系数0.01。
默认microbatch 8，OOM后4×累积2；冻结模型常驻显存也计入峰值。
原先仅轻量头的显存估计不能作为当前总占用保证。

## 开发集与held-out test

| 阶段 | 频率与用途 |
|---|---|
| dev | 每2,000步及结束；融合器按AbsRel、学生按BCE选择best.pt |
| test监测 | 默认关闭（heldout_every=0）；最终test在矩阵全部训练结束后一次完成 |
| 最终评价 | 最佳checkpoint，完整精度、边界、时序、效率和少量可视化 |

融合test监测保存七项深度指标、逐序列/数据集/宏平均及有效样本计数，
首次同时记录原始Depth Pro基线。`heldout.jsonl`、`heldout.csv`保留恢复记录，
同一步不重复追加；`heldout_baseline.json`保存基线。学生仅报告对MFTIQ的heldout BCE。
`training.heldout_every: 0`可关闭test监测，不改变训练身份。当前四套配置均默认关闭，不执行step 0 test。

开发AbsRel保持未裁剪预测；test监测和最终评价使用相同的GT掩码、范围及画布。
test不参与反向传播、学习率、早停或checkpoint选择。评价结束恢复模式和随机状态。
每次完整评价均需重新运行冻结模型，会增加墙钟时间。人工依据test调方案后，
这批test已参与开发，不能称从未查看的最终测试集。

## 划分、来源与恢复

数据配置及全量Raw说明见[DATASETS.md](DATASETS.md)。当前NYU未完成时使用
kitti_tum.yaml；mixed.yaml为KITTI＋TUM＋NYU，mixed_scannet.yaml额外加入ScanNet。
显式要求的来源缺失会报错，不静默降级。

KITTI保护历史13个drive、全部帧、两路相机及导出副本；剩余drive约15%开发。
TUM默认fr2训练/fr1开发/fr3测试，已知同地点变体按group归评价侧。
NYU排除官方654测试图所属完整房间、原八个评价场景和其他历史场景，同房间a/b归组。
ScanNet按sceneXXXX物理空间归组，官方val和既有评价空间排除于训练。
新增历史评价drive会拒绝继续旧清单，不静默修改固定划分；没有地理证据时不保证不同drive没有场景重叠。

即时计算实验以原始清单、预处理、实际冻结权重和在线协议建立身份。
改变划分、优化超参或模型须使用新结果目录。旧协议实验不能自动续接，
当前协议重跑相同命令从last.pt恢复优化器、scheduler、AMP和随机状态。
结果、清单、模型权重和下载数据互相独立。

## 最终评价与耗时解释

各分支保存`evaluation-test/metrics.json`、`frames.jsonl`及少量可视化；
`summary.json/csv`按数据源汇总。指标为米制、不按GT做尺度/偏移对齐。
边界为自定义log-depth Sobel F1，稀疏GT无有效3×3邻域时省略并报告覆盖率。
时序对应统一用Farneback，仅真正相邻的已评价输出参与；有双帧GT则扣除真实深度变化，
有可靠位姿则增加静态几何诊断，不冒充论文3D TAE。

独立evaluate的在线耗时从预读等待到预测结束，含冻结前端及统一Farneback准备，
不含模型加载、指标计算和结果写入；另报包括指标/写入的整体墙钟与吞吐。
共享evaluate-matrix一次读取测试集，共用每对图像的前端结果；融合器耗时单列，
独立端到端延迟为空，整体峰值包括全部常驻模型与评分张量。
JSONL写入和图片保存全部成功后才发布最终metrics.json，后台错误会传回主线程。

## 观察训练与评价

`observability/`独立负责进度、会话、标量和图表。默认开启终端tqdm、Matplotlib PNG
及TensorBoard；后台重定向运行时每30秒输出纯文本进度。开发集与
最终/矩阵评价均有帧对总量和ETA。详见[KITTI_TUM_START.md](KITTI_TUM_START.md)。
每个实验生成`plots/`和`tensorboard/`，可用`plot [--run NAME] [--tensorboard]`
从已有日志重建。新`validation.jsonl`补充开发分数、最佳值、各来源分数和耗时；
`training.jsonl`保留原字段并增加更新耗时和吞吐，heldout日志增加评价耗时。
观察配置不进入实验身份；恢复会话会屏蔽旧无效尾部，TensorBoard清理旧视图后重放。
关闭图表或TensorBoard不会关闭原始标量日志。服务必须由用户手动启动，默认监听本机。
