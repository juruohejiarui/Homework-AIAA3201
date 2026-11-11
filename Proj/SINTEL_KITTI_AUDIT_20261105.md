# Sintel / KITTI 数据划分与论文使用审计

日期：2026-11-05。此审计只新增记录，不修改模型、训练划分、评估程序或已有结果。

## 结论

1. Sintel 官网公开的是 depth training GT。将其用作未在 Sintel 深度上训练的模型的评估数据是可行的；这不等于存在另一套公开的 depth testing GT，也不等于所有预训练组件均未见过其 RGB。
2. 当前实际运行的 Sintel benchmark 为 training/clean，23 个序列、1,064 帧；当前融合头训练、dev、held-out 清单均无 Sintel。
3. 当前实际运行的 KITTI video benchmark 为 depth/val 对应左相机的 13 段、2,806 帧。全部 2,806 帧也在项目 held-out test 中；与训练、dev 的当前帧、前一帧及物理序列交集均为零。两份 KITTI 结果不能视为独立测试证据。
4. DA-V2 官方仓库 metric_depth 下提供的 652 帧 KITTI 列表，与项目 train/dev/test 清单分别重合 396/95/161 帧；与 video benchmark 重合 143 帧。当前融合头 checkpoint 不适合在完整该列表上声称未见测试成绩。这里统计的是清单归属，不声称每条训练记录都实际被随机采样过。
5. 当前 RAFT 权重为 raft_large_C_T_SKHT_V2-ff5fadd5.pth。官方说明它在 Sintel、KittiFlow、HD1K、FlyingThings3D 混合数据上进行了 Sintel 阶段微调。因此 RAFT 分支的 Sintel 结果不能称为整个系统对 Sintel 完全零样本。其监督为光流；不能直接写成使用了 Sintel 深度 GT。KittiFlow 与当前 KITTI depth 图像的逐帧映射未在本次审计中建立，不能据此声称整条 RAFT 管线在 KITTI 上无图像重合。

## 四篇论文和官方代码

| 模型 | Sintel | KITTI | 与当前项目的关系 |
|---|---|---|---|
| Depth Anything V2 | 论文表 2 为零样本相对深度评估；公开论文及仓库没有提供足以锁定该表 Sintel clean/final、逐帧列表和数量的完整证据。不能把 VDA 对 DA-V2 的评估设置直接归为 DA-V2 原论文设置。 | 论文同时有表 2 的零样本相对深度结果、表 4 的 KITTI 域内微调结果。官方 metric_depth 仓库另提供 652 帧、28 段的 RGB/公开 GT 列表；该文件不能单独证明相对深度表 2 的全部预处理设置。发布的 outdoor metric 模型训练于 Virtual KITTI，不能与表 4 真 KITTI 微调模型混淆。 | 当前相对 DA-V2-S 前端不等于论文 KITTI 微调模型；当前 video benchmark 不等于该 652 帧列表。 |
| UniDepth V2 | 论文作为未见的室外评估数据；官方 Sintel 类 test_split 明确是 training.txt。类默认深度范围为 0.001–1000 m。论文/仓库未给出可在本次核对中直接枚举的 HDF5 内 training.txt，故不捏造精确评估帧数和 RGB pass。 | Eigen 划分的 KITTI 微调实验，以及官方 Depth Prediction 榜单，属于不同评测；基础零样本权重与 KITTI 微调权重也须区分。官方类 test_split 为 kitti_eigen_test.txt，另有 kitti_test.txt benchmark 配置。 | 当前冻结 V2-S 在本地 video cohort 上重新评估，不能直接套用 Eigen 微调或服务器成绩。 |
| Video Depth Anything | 官方提取脚本读取 training/clean 和公开 DPT；每个场景完整提取，JSON 取前 100 帧，现有 Sintel 场景均不足 100 帧。 | 官方脚本读取 depth/val，配对 raw 左相机 image_02；逐序列按排序后的有 GT 帧取前 110 或 500 帧。论文长视频设置最多 500 帧，短 KITTI 最多 110 帧。 | 当前数据来源与取样规则相符；本地 KITTI 13 段、2,806 帧不是论文逐帧名单的已验证复现。公开脚本由传入目录决定序列集合，论文未提供可直接比对的完整 KITTI 视频 cohort。 |
| Depth Pro | 正式版表 18 明确 1,064 张、436×1024，深度范围 0.01–80 m；有米制深度和边界评估。官网 depth training GT 可支持该规模评估；1,064 张数量本身不能证明 RGB pass 与本地 clean 逐文件相同。 | 数据来源总表 15 标记 KITTI 为 Testing，但正式版定量米制评估表 1/4 和评估设置表 18 未提供 KITTI 对应列、样本数、划分。不能据总表擅自补成 Eigen 652 或官方隐藏 test 协议。 | 当前 Sintel 也是 1,064 帧，但使用 clean、上限 70 m，并有整视频对齐；与 Depth Pro 原论文 80 m 米制评估不同。 |

UniDepth V2 以 2025-12-18 的 v2 正文为准；Depth Pro 以 2025-04-21 的 v2 正文为准。VDA 同时有单图和视频评估表；本项目对应视频设置，不能将其单图表中的 DA-V2 数值作为当前视频设置的原论文对照值。

## KITTI 三种含义

- Eigen：研究者从 KITTI Raw 定义的评估划分，可用公开激光雷达/深度标注本地评分。其 train/test 序列边界与 depth 包 train/val 边界不等价。官方 DA-V2 的 652 帧列表本身同时引用 depth/train 与 depth/val。
- Depth train/val：官方公开深度标注包。当前训练使用 depth/train 后自行分 train/dev；当前 held-out KITTI 使用 depth/val。当前 VDA 风格 video benchmark 再选其中左相机前若干帧。
- 官方 Depth Prediction 隐藏测试：官网明确部分手工选择图像的深度不公开。需要官方服务器评分；当前项目本地结果不是该榜单结果。Sintel 光流评测服务器也不能当成深度预测提交入口。

使用公开 training GT 做外部深度评估，本身不构成泄漏。决定因素是被评估模型及其预训练组件是否接触过这些数据，以及是否按报告要求保留了训练/测试边界。

## 当前结果的可用范围与处理

- 可保留：当前 KITTI 公开 depth-val 视频测试，以及当前 fusion-head held-out 测试。对三种融合头，它们的 KITTI 序列与训练/dev 隔离。
- 必须修改解释：KITTI benchmark 与 held-out 是重叠的不同取样/处理视角，不能累计为两套独立泛化证据；Sintel 应写明公开 training/clean 上的外部评估。
- RAFT 的 Sintel 结果需披露光流预训练重合；Farnebäck 分支无该 RAFT 预训练问题。
- 若要求严格的 KITTI Eigen 实验，需要先按 Eigen 测试序列重新划分，排除训练和 dev 中对应的全部物理序列，再重新训练/选模。仅切换评估列表或删除 396 张测试图片不能变成完整 Eigen 复现。
- 当前共同评估可用于同一 cohort 下模型比较，但不等于四篇论文原始分数的逐项复现；范围、对齐单位、RGB pass、分辨率、统计聚合、权重版本均须匹配后才能如此称呼。

## 复核材料

- 机器可读计数和完整序列交集：`outputs/audits/sintel_kitti_20261105/overlap.json`
- DA-V2 的 652 个帧 ID 逐帧归属：`outputs/audits/sintel_kitti_20261105/dav2_kitti_652_membership.tsv`
- 实际 case 来源：`outputs/explore_a/shared_models_evaluation_v1/benchmark/{kitti,sintel}/*/cohort.json`
- 训练清单：`outputs/explore_a/mixed_h5_balanced/data.json`。其 SHA-256 与本次评估 session 固定记录一致。
- 比对 ID 为 `(drive, camera, frame stem)`，消除了 Resources/ResourcesLow 路径差异。抽查两处同 ID 的 RGB SHA-256 相同；不声称对全部镜像文件进行了内容哈希。

## 一手来源

- [Sintel depth 官方下载与单位说明](https://sintel.is.tue.mpg.de/depth)
- [KITTI Depth Prediction 官方说明与榜单](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [Depth Anything V2 正文及附录](https://arxiv.org/html/2406.09414v2)
- [DA-V2 官方 KITTI 652 帧列表](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/dataset/splits/kitti/val.txt)
- [UniDepth V2 正文](https://arxiv.org/html/2502.20110v2)
- [UniDepth 官方 Sintel 类](https://github.com/lpiccinelli-eth/UniDepth/blob/main/unidepth/datasets/sintel.py)
- [UniDepth 官方 KITTI 类](https://github.com/lpiccinelli-eth/UniDepth/blob/main/unidepth/datasets/kitti.py)
- [VDA 正文及补充说明](https://arxiv.org/html/2501.12375v1)
- [VDA 官方 Sintel 提取脚本](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/benchmark/dataset_extract/dataset_extract_sintel.py)
- [VDA 官方 KITTI 提取脚本](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/benchmark/dataset_extract/dataset_extract_kitti.py)
- [Depth Pro 正式版正文及附录](https://arxiv.org/html/2410.02073v2)
- [Torchvision RAFT 权重训练数据说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)
