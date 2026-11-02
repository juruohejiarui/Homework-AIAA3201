# 视频深度与方向 A

项目代码位于 `src/depth_project` 和 `src/explore_a`。方向 A 统一使用冻结 DA-V2-S，
三模型为 `unet_guided`、`flow_refine`、`final_flow`，支持 Farnebäck／RAFT。
原前端专属代码、依赖、权重、预测缓存与旧执行结果已删除，训练划分保留。

- [完整训练、恢复、缓存和 DGX 命令](src/explore_a/DA_TRAINING.md)
- [三模型结构](src/explore_a/FUSION_MODELS.md)
- [基础数据、双目、独立模型工具](GUIDE_zh.md)
- [方法和评价协议](paper_reproduction_zh.md)

训练数据位于 `~/Documents/Resources/Datasets`；KITTI／Sintel／Bonn benchmark
仍位于 `~/Documents/ResourcesLow/Datasets`，TUM 官方 validation 下载到前者。
四个视频集合之外，NYU labeled 可单独进行无历史单图评价。

使用现有 `aiaa3201` 环境，不创建额外虚拟环境。冻结模型包括 DA、RAFT、MFTIQ，
独立参考 UniDepthV2／VDA 保留。GT 始终留在本机；统一 hybrid 消费端只读预测缓存。

数据下载交接见 [下载文档](download_data_zh.md)；TUM 下载命令以 DA 指南为准。
