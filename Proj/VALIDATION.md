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

## 连续 dev、双对齐与 oracle（2026-11-02）

原10%离散选择已由公共入口的确定性连续片段替代：3,341对，来源配额
KITTI/NYUv2/TUM=1,253/1,253/835，全部21/47/9个序列，46/47/18个片段。
manifest行顺序、模型、光流及重复验证不影响结果。

本次87项受影响检查分组通过，涵盖选择、双对齐、排序、GT无效区、oracle嵌套、
恢复复用、旧best归档与中断恢复、训练/heldout/观测、模型比较、checkpoint、
冻结pipeline及缓存。分组日志位于 `/tmp/dev-dual-*.log`；语法、100字符行宽及
`git diff --check`通过。命令参数解析使用aiaa3201环境验证。

真实完整oracle已结束：3,341对、111片段，Farnebäck + rule，采用配置中的
缓存/DGX/本机流水线。没有训练优化器更新，没有完整benchmark或预测缓存生产。
相同目录再以CPU模式恢复验证：全部结果直接复用，未创建冻结推理runtime。
临时推理staging已清空。消费者没有预测NPZ写入路径。

全部GT区域、逐图固定基线映射下，理想凸融合/有界纠偏的逆深度主损失下降分别为：
KITTI 11.22%/75.31%，NYUv2 7.30%/40.64%，TUM 15.19%/56.62%。
这是条件输出范围，不是学习模型精度。完整目标区分、双对齐深度指标、排序及覆盖见
`outputs/explore_a/da_continuous_dev_oracle_20261102/analysis.md` 和 `summary.json`。

## train/evaluation 分阶段调度调优

新增 `execution.routing.train/evaluation`，按提交ticket阶段决定路由；当前公共配置
分别为缓存/DGX/本机38/49/13及42/46/12。所有模型继承，生成hybrid配置也保留策略。
原run、模型参数、数据顺序、固定dev、缓存精度均不变。新增开销统计可与计算重叠，
不将累计RPC时间相加作为步时长。

46项受影响检查分组通过（`/tmp/phase-routing-complete.log` 44项及新增2项）；
覆盖路由边界、公式与可用率、独立RNG、配置继承、checkpoint、生成consumer、
解码乱序/共享端点/单名额顺序、失败及取消清理、阶段切换、NPZ只读、RPC和评价入口。
纯缓存真实测试发现并修复了完成结果占满名额的顺序死锁，没有放宽容量或增加fallback。

实测三条纯路径、公式计算、同样本train/dev/test对照和Guided训练复核均已完成。
临时训练头及优化器在独立进程中丢弃，没有写正式checkpoint或预测NPZ；
不根据test精度选择比例，精度评分只执行以计入性能开销，结果不参与选择。
本轮五个 `phase_routing_*` 测量输出目录已按用户要求删除；已采用的路由配置保留。
