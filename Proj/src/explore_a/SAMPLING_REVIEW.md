> 这是历史协议分析，不是当前执行结果或操作入口。
> 当前协议与命令见 [DA 操作指南](DA_TRAINING.md)。

# 固定采样基线、后续分层抽样与 oracle 差距

日期：2026-10-31。当前训练尚未结束；以下数值是固定 dev 的中途诊断，不是最终 test。
本文记录旧 fix 基线及当时的诊断。当前配置已支持 video_sqrt 和 3:3:2，见 FUSION_MODELS.md。
当时仅将已有策略命名为 `training.sampling_mode: fix`，不改变 RNG、
训练帧对、来源概率、结果目录、缓存、checkpoint 或进程。

## 已实现的 fix

```yaml
training:
  sampling_mode: fix
  source_weights:
    kitti: 0.5
    tum: 0.25
    nyuv2: 0.25
```

fix 表示概率固定：每个优化步选一个来源，再在该来源全部训练帧对中有放回均匀抽样。
不是固定 batch、固定训练子集、按顺序读完一个 epoch，也不是 cache/DGX 路由概率。
旧配置缺省也是 fix；显式标记不增加随机数调用。恢复和 microbatch 的样本决定不变。
fix 继续保留；新增 video_sqrt 显式选择视频分层采样，不会静默改变旧 fix 的行为。
正在运行的进程不热更新。添加该标签不会要求更换 run。

## 冻结模型是否见过 KITTI／NYUv2

原已废弃前端 论文附录 C.1 表13将真实 KITTI、NYUv2 列为 Testing；训练集包含
VirtualKITTI2、ScanNet、ARKitScenes 等。VirtualKITTI2 不是真实 KITTI，但存在场景风格
和内容来源的关联，不能将其理解为完全没有驾驶域先验。
官方仓库说明发布的是重新训练的参考实现，性能不与论文完全相同；因此上述说明是
论文协议证据，不是当前 checkpoint 的逐帧训练来源审计。

当前 farneback-rule-flow_refine 的光流／遮挡均不学习，不使用 RAFT 或 MFTIQ 结果。
因此 RAFT 的 KITTI 训练经历不能解释本组现象。其他 RAFT 实验须注明：当前
C_T_SKHT_V2 的 Sintel 微调混合包含 KittiFlow，即使名称没有末尾的 _K。

来源：
- https://arxiv.org/html/2410.02073v1#A3.SS1
- https://github.com/apple-aiml-research/ml-depth-pro
- https://docs.pytorch.org/vision/main/_modules/torchvision/models/optical_flow/raft.html

## 下一组固定采样对照：建议，尚未实现或启动

不根据这五个 dev 点宣称推导出了最优比例。先固定目标为室外与室内各占一半：
KITTI 50%、TUM 25%、NYUv2 25%，与原 fix 比较时只改变来源内抽样。
三源训练帧对保留率均为100%；可被选中不等于每个训练预算内都会遍历。
已有有放回采样在当前30k步预算下本来就不会覆盖全部帧对。

建议层次为：来源 -> 物理group -> 同group内序列 -> 时间位置 -> 原始完整帧对。
物理group沿用manifest，避免同一个NYU房间的多次采集被误当多个独立房间。

令 n_g 为group的帧对数，取 P(g|source) proportional to n_g^0.5；
同group内若有多个序列，采用相同的平方根温度权重。
这是温度采样 P proportional to n^(1/T) 在 T=2 时的工程适配，不是文献证明的深度
任务最优温度。T=1恢复长度权重，T趋向无穷接近等权。

视频内部按已有帧对的时间顺序分为最多8个等数量分层，轮换覆盖这些分层，层内由
独立固定seed抽取帧对。8是待对照的工程初值。NYU仅使用保留帧序号顺序，不伪造时间戳。
分层只决定哪些既有帧对被选中，不把抽出的两个远隔帧重新配对，不重建缓存。

KITTI和TUM的group等于当前序列，其平方根策略的精确manifest统计为：
- KITTI最长5段的来源内概率：49.94% -> 19.53%。
- TUM最长5段的来源内概率：40.31% -> 33.94%。
- NYU需按204个物理group计算，不能直接把224序列的温度统计作为房间统计。

保留全部候选，通过权重降低冗余；当前六锚点描述量不足以断言KITTI应删到25%、
TUM应删到50%等。若确实需要减小数据常驻量，再做固定、嵌套、按物理group和时间
分层的10/25/50/100%候选池，对比相同优化步数。仅比较子集指标不能证明训练等效。

研究依据和适配边界：
- Arivazhagan et al., 2019，温度采样；原论文是多语言翻译，本项目借鉴采样数学形式，
  不迁移其最佳温度或声称在深度任务有同样收益。
  https://arxiv.org/html/1907.05019
- Temporal Segment Networks, ECCV 2016，借鉴覆盖整个视频的分段抽样思想；原任务是
  动作识别，本项目保留既有深度帧对，不照搬视频分类的模型或训练协议。
  https://arxiv.org/abs/1608.00859

## 动态TUM配比

动态增加TUM仅在其问题主要是梯度曝光不足或多源竞争时可能有效。不能修复：
输入无法辨识的误差、GT误差、深度warp几何差异或纠偏输出表达限制。
TUM训练主损失也停滞，不是已经证实的少见数据源欠采样。

后续可先在固定训练probe上比较三源的归一化主深度损失进展、梯度范数和夹角，
再考虑受限的动态调度；不得仅按绝对loss或单次dev变差就提高权重。
保持KITTI=.5，TUM在[.15,.35]内变化，NYU=.5-TUM，可作为有界实验设计；边界是
本项目的工程选择，不是论文给出的最佳值。调度更新需平滑，并保存状态以支持恢复。
该调度当前没有实现，不影响正在训练的fix基线。

GradNorm调整的是任务损失权重／梯度，不等于直接改变样本概率；单源batch、Adam、
梯度裁剪情况下二者不严格等价。Group DRO优化最差组，与当前平均性能目标不同，
对高噪声的小组还需正则化，不能保证加大TUM就提高泛化。
- https://proceedings.mlr.press/v80/chen18a.html
- https://openreview.net/pdf?id=ryxGuJrFvS

## 同口径oracle对照

来自已完成固定10% dev oracle：3341帧对；与当前dev帧对ID一致。
主表统一使用10k，补充统一8k（当时的最佳总体选择步）；不按数据源分别挑最好步。
AbsRel越低越好。oracle逐像素使用GT，不是可部署或必然可学的性能目标。

| 来源 | 原始原已废弃前端 | 凸融合oracle | ±10%纠偏oracle | 模型8k | 模型10k |
|---|---:|---:|---:|---:|---:|
| KITTI | .185260 | .167377 | .101517 | .132512 | .127574 |
| TUM | .176451 | .156181 | .092421 | .169730 | .170245 |
| NYUv2 | .134367 | .118619 | .068079 | .125727 | .130418 |

改善空间兑现率=(raw-model)/(raw-repair_oracle)。10k时KITTI/TUM/NYU为
68.89%/7.38%/5.96%；8k时为62.99%/8.00%/13.03%。这些是描述性比例，不是学得会
的概率、GT正确率或模型能力百分比。oracle和dev均本机FP16，但batch形状可能不同。

KITTI已超出任何凸融合可达的AbsRel，说明纠偏路径提供了实际收益；TUM/NYU距离
凸融合oracle仍有差距，意味着不是仅靠突破凸区间就能获得理想性能。

原因按证据强弱区分：
1. 已观察：TUM总loss下降主要是辅助项，主深度loss只改善约3%；NYU主深度loss
   改善约11%，但在部分未见房间上回退；不是三源统一的训练完全失败。
2. 已观察：TUM纠偏RMS约.0951，接近.1上限，但oracle也使用同样的±10%范围，
   不能把大部分差距单独归为幅度不足。需检查修正符号／空间分配、gate和tanh梯度。
3. 已知协议：无历史处精确回退。FB覆盖KITTI/TUM/NYU约92.2%/86.4%/80.3%；
   oracle已经使用同一限制，因此它不能解释模型对这个oracle的剩余差距。
4. 待验证：三源梯度竞争、TUM有限训练场景、NYU训练/dev房间差异、网络容量和GT
   误差。当前日志没有梯度夹角、逐像素修正分布或同帧raw/fused训练收益，不能定因。
5. 信息限制：oracle直接知道GT，模型只见RGB/预测/几何。候选共同偏差的修复可能
   需要模型输入中没有的尺度线索。2D光流对齐也不自动补偿3D深度随视角的变化。

当前训练完成后，用共同best checkpoint重做这一表和heldout评价；开发集oracle不
作为test标签或推理输入。不要将当前10k中途结果写成最终结论。

本次计算表和图位于：outputs/explore_a/sampling_oracle_review_20261031/。
