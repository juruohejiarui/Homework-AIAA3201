# DGX Spark 缓存与本机训练

正式上传、生成、训练及最终评价均由用户启动。首次实施已创建DGX的独立环境并同步
项目及现有权重；后续改代码时再运行`sync-project`，不使用`--delete`。
不修改`fyp-train`或`vllm`。

## 从旧版升级

旧worker和协调器仍在运行时，不覆盖DGX源码，也不要同时启动新版协调器。
先由用户在本机请求有序停止（此命令同时兼容旧版worker）：

```bash
cd ~/Documents/Projects/Homework-AIAA3201
bash Proj/dgx_cache.sh stop --host all
```

等待DGX当前序列计算结束、worker及旧pipeline退出，再执行：

```bash
bash Proj/dgx_cache.sh sync-project
```

同步检测到DGX worker锁被占用时会拒绝覆盖；不自动终止进程。
之后按下面启动新版两端worker和唯一的本机协调器。首次双机协调会接管旧远端完成、
生成中、待生成及部分上传任务；已有本机缓存继续使用，不重算，不改变配方或checkpoint身份。

## 双机并行运行

在本机终端一启动本机4090 D worker：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiaa3201
cd ~/Documents/Projects/Homework-AIAA3201
bash Proj/dgx_cache.sh worker --host local
```

DGX终端启动另一个worker：

```bash
ssh Dgx-e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiaa3201
cd ~/Documents/Projects/Homework-AIAA3201
bash Proj/dgx_cache.sh worker --host dgx
```

本机终端二启动唯一的调度/传输协调器：

```bash
cd ~/Documents/Projects/Homework-AIAA3201
bash Proj/dgx_cache.sh pipeline \
  --config Proj/configs/explore_a/kitti_tum.yaml --workers local,dgx
```

本机直接读取原始RGB，直接写入完整缓存目录，不复制输入，也不建立上传符号链接。
DGX任务才上传RGB、回传预测并清理远端托管副本。两端逐序列动态分配，较快一端自然
取得更多任务；每端一个GPU worker。本机最多一个分配中任务，DGX最多两个在途任务，
传输最多并发两个。两端都使用四线程预读、八项预取及有界后台最高级别压缩。

每个任务持久保存owner.json，归属不随超时或断网改变。协调器和每端worker分别有
进程锁，进程退出后自动释放；重启复用归属和完整输出，不需要等待租约超时。
同一序列不会分配给两端。本机与DGX都使用同一份模型/预处理代码和固定划分。

默认SSH别名Dgx-e通过192.168.77.2:11451的2.5GbE连接。每个KITTI drive双相机合并为
一个任务，TUM按序列。固定170,360帧对及train/dev/test标签原样保留，不加入NYU。

## 单机模式

仅本机运行时，worker命令不变，协调器使用：

```bash
bash Proj/dgx_cache.sh pipeline \
  --config Proj/configs/explore_a/kitti_tum.yaml --workers local
bash Proj/dgx_cache.sh status --host local
```

本机模式不连接SSH。旧版DGX配方尚未接管过任务归属时，先通过双机模式接管一次；
本机模式会拒绝猜测旧远端任务归属。有未完成的DGX任务或待确认清理时，应启用DGX
完成这些任务，再切换到仅本机。已完成且清理确认的DGX结果可直接复用。

仅DGX运行继续支持旧命令（worker默认dgx，pipeline默认--workers dgx）：

```bash
bash Proj/dgx_cache.sh pipeline --config Proj/configs/explore_a/kitti_tum.yaml
```

仅DGX模式也不会接走已经归属本机的未完成任务。

## 路径和内容

| 内容 | 位置 |
|---|---|
| 本机完整缓存 | `~/Documents/ResourcesLow/cache/Homework-AIAA3201/explore_a/kitti_tum-fp32-flash-v1/` |
| DGX任务与临时缓存 | `~/Documents/Resources/cache/Homework-AIAA3201/explore_a/kitti_tum-fp32-flash-v1/` |
| DGX项目 | `~/Documents/Projects/Homework-AIAA3201/` |
| 混合消费配置 | 本机缓存根下的`consumer_hybrid.yaml` |
| 混合训练结果 | `Proj/outputs/explore_a/kitti_tum/runs_hybrid_60k/` |

两个端点采用相同的相对输出路径：

```text
frames/kitti/<日期>/<drive>/image_02/data/<帧号>.npz
flow/raft/kitti/<日期>/<drive>/image_02/data/<帧号>.npz
teacher/raft/kitti/<日期>/<drive>/image_02/data/<帧号>.npz
frames/tum/sequences/<序列>/rgb/<RGB时间戳>.npz
flow/farneback/tum/sequences/<序列>/rgb/<RGB时间戳>.npz
teacher/farneback/tum/sequences/<序列>/rgb/<RGB时间戳>.npz
```

frames仅含`depth`（米，1×H×W），flow含`forward/backward`（像素，2×H×W），
teacher仅含`visibility`（历史可用性，1×H×W）。所有数组为FP32，ZIP DEFLATE级别9，
不缓存RGB/GT，不量化成FP16。KITTI画布256×832，TUM为384×512。
元数据保存模型文件名/大小、预处理、精度；各任务完成记录另含生产端、GPU及实际软件版本、
注意力调用计数、回退原因、生成耗时、峰值显存和压缩后字节数。
不把跨硬件浮点结果声明为逐位相同。


## 4,000 RGB帧限制与视频记录表

cache_policy.max_rgb_frames: 4000只控制缓存生成。KITTI合计原始image_02/image_03
全部RGB文件，TUM统计原始序列全部RGB，不使用帧对数或manifest命中帧数。
超过4,000才跳过，等于4,000仍生成。198视频、170,360帧对及原有划分保持不变。
4个KITTI drive和2个TUM序列标为skipped_length；长视频仍完整参与训练和评价。

新worker、上传、回传和协调器均跳过超长任务，不为其补算或等待完成标记。
本机已有完整或部分NPZ保留。本次按用户选择删除DGX超长视频的托管预测及输入副本，
保留短视频未完成结果；未来刷新策略不会自动删除已有预测。

停止旧worker和协调器后，仅刷新策略与消费配置，无需重建manifest：

```bash
bash Proj/dgx_cache.sh policy --config Proj/configs/explore_a/kitti_tum.yaml
```

缓存根下cache_status.json/CSV记录总RGB数、缓存资格、本机none/partial/complete、
现存/预期文件数、远端状态及跳过原因，启动或status时刷新。只枚举文件，不全量解码
或哈希。该表是概览，实际训练逐项验证文件，新出现的有效缓存也可立即使用。

## 状态、恢复和空间

```bash
# 默认查询最近一次协调器启用的生产端；本机模式不会连接SSH
bash Proj/dgx_cache.sh status
bash Proj/dgx_cache.sh status --host local
bash Proj/dgx_cache.sh status --host dgx
# 停止对应端新派发，worker完成当前序列后退出
bash Proj/dgx_cache.sh stop --host local
bash Proj/dgx_cache.sh stop --host dgx
# 停止整个协调器及已启用的worker；旧stop命令等价
bash Proj/dgx_cache.sh stop --host all
```

状态包含任务归属、活动阶段、当前处理帧对数/速度、完成任务压缩字节数、总完成帧对、
生成耗时、失败原因和恢复路径。进度元数据最多每30秒更新；当前处理帧对数不等于
已发布完成缓存的数量。rsync另显示即时传输速度。旧worker无实时进度记录时不伪造数据。

本机任务经过pending → ready → generating → local_complete；DGX任务另经过
remote_complete → pull → local_complete → pruned。完成记录保留生产端、GPU、软件
版本和实际注意力后端信息；旧记录没有host字段时按原DGX来源解释。

停止后重启相应worker和pipeline即可续跑。若生成失败或空间不足，停止新派发并保留
任务归属及原子输出；在运行的其他任务可以完成。修复问题并等对应worker退出后重试：

```bash
# 重试失败任务，默认保持原生产端
bash Proj/dgx_cache.sh retry
# 本机离线重试一个任务；也可修复已完成但文件被误删的任务
bash Proj/dgx_cache.sh retry --host local --job tum__rgbd_dataset_freiburg1_xyz
# 原DGX任务同理；任务名请从status取得
bash Proj/dgx_cache.sh retry --host dgx --job kitti__2011_09_26_drive_0001_sync
```

重试要求协调器已退出。本机重试要求本机worker已退出；生成中或失败的DGX任务
同样要求DGX worker退出。
不自动跨端接管、不自动覆盖另一端的归属，不自动删除原始RGB/GT。
有单端stop请求时，重新启动协调器会清除所选端的派发暂停标记。

rsync使用部分文件目录续传，NPZ不重复压缩传输。后台写入全部成功并核对清单/大小后，
本机才发布local.complete.json；严格cache要求该标记，hybrid可复用已原子完成的单项NPZ，
不读取.tmp或半传输文件。DGX回传经本机确认后才清理该
任务托管的staging和预测NPZ，保留小型任务记录；本机生成的缓存不会被远端清理逻辑删除。
两端至少保留50GiB空闲，空间不足时退出并说明；释放空间后重试及重启即可。
不做全量哈希扫描，外部并发写盘仍可能改变可用空间。

### 关机或上传中断后的检查

`Conflicting upload link`表示托管上传暂存链接与固定sources.json不一致，不能据此判断
预测缓存损坏。新版上传会核对原始RGB大小并重建过期链接；保留真实文件和原始数据。
残留transfer.failed.json仍需要在协调器退出后用retry清除，再重新启动pipeline。
本机worker可以继续运行，重试DGX上传故障不会要求停止本机推理。

只检查已完成任务的文件清单和大小（不做哈希）：

```bash
PYTHONPATH=Proj/src python -m explore_a.cache.audit \\
  --output Proj/outputs/explore_a/kitti_tum/cache-inventory.json
```

对关机前后写入的指定任务增加NPZ解码、ZIP CRC、FP32数值和形状检查：

```bash
PYTHONPATH=Proj/src python -m explore_a.cache.audit --deep \\
  --job kitti__2011_09_30_drive_0020_sync \\
  --output Proj/outputs/explore_a/kitti_tum/cache-decode.json
```

检查命令只读，不改变完成标记或删除输出。complete=false任务的missing是尚待生成的
文件，不能当作已完成缓存丢失。需要重算时保留有效文件，仅移走确认损坏的NPZ，
清除对应完成状态并使用原生产端恢复；不要重新划分数据或删除整个缓存目录。

## 混合训练：不必等待全部缓存完成

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiaa3201
cd ~/Documents/Projects/Homework-AIAA3201
cfg="$HOME/Documents/ResourcesLow/cache/Homework-AIAA3201/explore_a/kitti_tum-fp32-flash-v1/consumer_hybrid.yaml"
# 先观察一组；无step 0 test，dev每2,000步
bash Proj/explore_a.sh --config "$cfg" train --flow farneback --occlusion rule --fusion unet
# 单组训练不自动test；训练完成后手动评价一次
bash Proj/explore_a.sh --config "$cfg" evaluate --flow farneback --occlusion rule --fusion unet
# 或两个学生和全部18组训练结束后，只执行一次共享test和汇总
bash Proj/explore_a.sh --config "$cfg" run-matrix
```

所有学生和融合器60,000步，有效batch 8，dev每2,000步，heldout_every=0。
仅dev决定best.pt，最终评价继续读取best.pt。训练日志/PNG/TensorBoard/CSV保留；
默认不产生周期test曲线，最终test保存在evaluation-test的frames.jsonl/metrics.json，
跨实验表为summary.csv，不补造测试曲线。旧在线结果仍在runs，不自动混接checkpoint。
严格cache模式不加载本机冻结大模型；hybrid按缺失项加载对应模型，本机直接读取RGB/GT。
混合评价分别报告缓存读取、即时前端及融合成本，历史缓存生成耗时仍在任务记录中。

```bash
tensorboard --logdir Proj/outputs/explore_a/kitti_tum/runs_hybrid_60k --host 127.0.0.1 --port 6006
bash Proj/explore_a.sh --config "$cfg" plot
```


execution.backend=hybrid逐项检查两帧深度、所选双向流和必要教师可用性；只运行缺失项
对应的模型，缺失或无效缓存会记录原因后即时计算，不写回磁盘。部分视频可逐项复用。
规则融合不加载教师；流已缓存而教师缺失时也不加载RAFT。模型首次需要时加载并复用。

首次初始化保护Python/NumPy/PyTorch随机状态。Depth Pro保持FP16和Flash优先，
RAFT/MFTIQ保持FP32。模型与训练头共享本机GPU，峰值由日志记录；原4×累积2的OOM
恢复保留。顺序评价只复用紧邻前帧的原始深度，切换序列或相机清空，不递归融合。

consumer.yaml仍是严格cache模式。混合配置使用独立runs_hybrid_60k，不自动混接旧
checkpoint；缓存覆盖率变化不会改变混合实验身份。train/dev/test/矩阵评价均支持混合。

training.jsonl、validation.jsonl、scalars.csv及TensorBoard的train_inputs/dev_inputs
记录缓存命中、即时计算、无效缓存计数及耗时。训练是该更新的增量，dev是整轮增量。
最终metrics.json的latency.input_stats给出整次累计；矩阵计数属于共享前端。
在线分项耗时包含首次加载，model_load_s另列说明，不重复相加。不将缓存历史生成
成本算成本机单次评价时间。即时计算结果不会产生新的预测文件。

## 环境与注意力

环境实测记录见[DGX_ENVIRONMENT.md](DGX_ENVIRONMENT.md)。复建脚本为
`bash Proj/setup_dgx_env.sh`，仅在DGX运行；ARM依赖与完整freeze锁文件独立保存。
官方GPU wheel固定[torch 2.12/cu130](https://download.pytorch.org/whl/cu130/torch/)和
[torchvision 0.27/cu130](https://download.pytorch.org/whl/cu130/torchvision/)。
FlashAttention使用[社区v0.9.22资产](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/expanded_assets/v0.9.22)
中`flash_attn-2.8.3+cu130torch2.12-cp313-cp313-manylinux_2_34_aarch64.whl`。

Depth Pro保持FP16：兼容时调用外部FlashAttention，随后尝试PyTorch SDPA Flash，
均不兼容则保留SDPA。实际调用计数而非安装状态写入每序列完成报告。
RAFT、MFTIQ及DINOv2保留FP32，不为Flash改变精度；卷积和相关性不强行替换。
[FlashAttention dtype要求](https://github.com/Dao-AILab/flash-attention#installation-and-features)。
