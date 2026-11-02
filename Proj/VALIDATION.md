# DA 单前端验收记录（2026-11-02）

当前回归：206 passed、1 skipped、2 warnings。完整输出保存在本次运行的
`/tmp/da-accepted-tests.log`，不代表正式训练或 benchmark 精度结果。

验证覆盖三模型 DA 逆深度路径、两种光流入口、正尺度仿射对齐、纠偏边界、
历史无效精确回退、梯度、checkpoint、SSI 日志、共享缓存传输及安全清理、
阻塞 hybrid、RPC、阶段切换、原生缓存和 TUM 下载参数。

真实集成仅使用 KITTI／TUM／NYUv2 每源一对训练样本：
本机 DA＋Farnebäck、DGX 内存 DA＋RAFT，三个头均完成反向传播，冻结模型无梯度。
没有优化器更新，没有保存训练 checkpoint 或预测 NPZ。

删除前后训练 manifest 的 SHA-256 相同；共享光流／MFTIQ 数值身份一致。
本机共享缓存目录状态表重新统计了 478 个任务；未解码或哈希整个预测库。
原始数据、下载进度、DA 预测及共享组件保留，旧模型和执行结果已删除。

清理清单和真实集成 JSON 位于
`outputs/explore_a/da_only_migration_20261102/`。
启动、恢复、评价和下载命令见 [DA 指南](src/explore_a/DA_TRAINING.md)。
