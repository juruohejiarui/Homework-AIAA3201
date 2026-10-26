# 方向 A：相邻帧 confidence 融合

冻结Depth Pro；RAFT/Farneback × 规则/MFTIQ/遮挡学生 × CNN/U-Net/门控加残差，共18组。
支持online、严格cache和逐项缓存/即时计算的hybrid后端。双机只为不超过4,000 RGB帧的
视频新增缓存；长视频保留在训练和评价中。操作见[DGX_CACHE.md](DGX_CACHE.md)。

- [KITTI＋TUM立即开始](KITTI_TUM_START.md)
- [模块、数值、划分、恢复及评价说明](TRAINING_GUIDE.md)
- [模型下载](MODEL_DOWNLOADS.md)
- [各数据来源与全量Raw范围](DATASETS.md)
- [TUM下载](TUM_DOWNLOAD.md)、[NYUv2 Raw下载与官方MD5](NYUV2_DOWNLOAD.md)
- [ScanNet v1完整下载](SCANNET_V1_DOWNLOAD.md)
- [早期设计记录](EXPERIMENT_PLAN.md)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiaa3201
export XFORMERS_DISABLED=1
cfg=Proj/configs/explore_a/kitti_tum.yaml
# 已有固定manifest时直接训练；没有时先prepare --extract-kitti
bash Proj/explore_a.sh --config "$cfg" train --flow farneback --occlusion rule --fusion unet
```

默认mixed.yaml为KITTI＋TUM＋NYU；NYU未就绪时显式使用上述两来源配置。
训练、模型下载和正式评价均由用户启动。全流程入口all会检查权重、准备/验证清单、
训练两个学生及18个融合器，然后执行共用当前样本前端的evaluate-matrix与汇总。

数据集适配和隔离在data，冻结即时流水线在online，几何在geometry，模型在models，
训练在training，指标及异步结果保存位于evaluation；__main__仅做命令派发。
previous采用真实前一帧原始深度，不递归融合。GT不进入模型输入。
默认60,000步，dev每2,000步独立选择best.pt；训练期间不运行test，矩阵结束后评价一次。

输出相对融合权重，不视作校准的正确概率；遮挡学生蒸馏软标签，不报告真实遮挡准确率。

默认显示训练及各评价阶段的进度、ETA，导出Matplotlib曲线并记录TensorBoard。
日志补画使用`plot`命令；图表在各实验`plots/`，事件在`tensorboard/`。
启动服务与断点曲线说明见[KITTI_TUM_START.md](KITTI_TUM_START.md)。
