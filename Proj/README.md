# 视频深度与方向 A

## 课程 Part 1 / Part 2 完整实验

独立入口不依赖方向 A checkpoint。使用现有 `aiaa3201` 环境，从仓库根目录运行：

```bash
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
python -m depth_project.course run
python -m depth_project.course verify
```

完整结果位于 `outputs/course_part1_part2/`，成果包为 `outputs/part1_part2_results.zip`。
包括 BM/SGBM 全部15场景、UniDepthV2 全部654张 NYU 无对齐米制评价、
DA-V2-S/VDA-S 官方片段 FP16 演示、全部预测、效率测量及三个课程素材难例。
官方视频会从课程 PDF 原链接下载；选取前 `min(100,实际帧数)` 帧并保留原帧率，
不足100帧时如实记录源文件限制，不复制补帧。模型权重仍使用 `checkpoints/` 中的官方权重。
数据路径与精确协议见[独立模块指南](GUIDE_zh.md)。

## 方向 A 研究

项目代码位于 `src/depth_project` 和 `src/explore_a`。方向 A 统一使用冻结 DA-V2-S，
三模型为 `unet_guided`、`flow_refine`、`final_flow`，支持 Farnebäck／RAFT。
原前端专属代码、依赖、权重、预测缓存与旧执行结果已删除，训练划分保留。

- [完整训练、恢复、缓存和 DGX 命令](src/explore_a/DA_TRAINING.md)
- [三模型结构](src/explore_a/FUSION_MODELS.md)
- [基础数据、双目、独立模型工具](GUIDE_zh.md)
- [方法和评价协议](paper_reproduction_zh.md)
- [共享评估与 DGX 执行](UNIFIED_EXECUTION_zh.md)
- [模型列表与 ensemble 评估](ENSEMBLE_EVALUATION_zh.md)
- [实验结果分析与诊断图](reports/README.md)

训练数据位于 `~/Documents/Resources/Datasets`；KITTI／Sintel／Bonn benchmark
仍位于 `~/Documents/ResourcesLow/Datasets`，TUM 官方 validation 下载到前者。
四个视频集合之外，NYU labeled 可单独进行无历史单图评价。

使用现有 `aiaa3201` 环境，不创建额外虚拟环境。冻结模型包括 DA、RAFT、MFTIQ，
独立参考 UniDepthV2／VDA 保留。GT 始终留在本机；统一 hybrid 消费端只读预测缓存。

数据下载交接见 [下载文档](download_data_zh.md)；TUM 下载命令以 DA 指南为准。
