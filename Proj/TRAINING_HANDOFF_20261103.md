# 训练实施：对话交接（2026-11-03）

本文件汇总原长对话中仍有效的训练决定、实现、实验和操作约束。它是工作交接，不是课程报告。
后续修改以用户新要求及实际代码/config为准，不把历史设想当作尚需执行的命令。

## 工作空间与要求

- 仓库 `/home/hjr/Documents/Projects/Homework-AIAA3201`，Python环境 `aiaa3201`。
- Python `/home/hjr/miniconda3/envs/aiaa3201/bin/python`；代码 `Proj/src/explore_a`。
- 阅读仓库 AGENTS.md。课程权威文件为
  `/home/hjr/Downloads/Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency (1).pdf`，
  SHA256 `ea65076917e87febb0c3c67773b9fc6edf304e2821a74266a03f22ef33abc226`。
  不使用历史六页附件推翻它。方向A保留光流warp、FB遮挡检查、confidence权重融合。
- 严格两帧是用户选定的实验约束，不应声称PDF明确禁止更多帧。
- 课程项目预算有限；优先完成既定模型，不自动增加矩阵实验、参数搜索、GradNorm或下载。
- 用户允许常规实现与受影响验证；正式训练、完整benchmark、全量缓存生产应由用户明确启动。
  完整dev oracle是此前已明确授权且已执行的例外。
- 模块职责清晰、命令入口薄、Python行宽≤100；必要注释解释单位、帧对、兼容判断。
  不做全量哈希/重复解码/缓存重算；不为普通config改动强制new run。
- 工作树已有大量未提交实现；不要reset、clean、覆盖用户修改或自动commit。
  `Proj/third_party/UniDepth/`是独立第三方资源，不能当垃圾删除。

## 有效实验范围与三个模型

已转为DA-V2-S单前端。Depth Pro代码、权重、专用预测缓存及旧执行结果此前按用户要求删除。
不要复活Depth Pro双前端方案、旧cnn/residual、遮挡student、原始unet入口。
保留冻结RAFT/Farnebäck与可选冻结MFTIQ。主遮挡策略rule，正式最终设计以Farnebäck为主，
三个模型的两种光流命令均可用。默认候选：

1. `unet_guided`：SmallUNet/GroupNorm，融合权重和历史reset门，凸融合基线。
2. `flow_refine`：双帧共享16/32/64编码器、光流对齐特征、ConvGRU、解码、全局/局部纠偏。
   已采用分解监督、轻量全局上下文和FiLM；不是早期共享tanh(global+local)结构。
3. `final_flow`：当前帧独立S纠偏分支＋1/4尺度显式T时序分支，不跨帧保留隐藏状态。
   T对真实帧对和自身参照用共享权重，差值产生全局/局部增量，由独立可靠性门控制。
   S不读取历史；自身参照使用当前特征、零流/零差分、相同有效区域。
   增量头零初始化，可靠性门初始0.5。训练双向、推理只正常方向。

所有模型先对历史预测做光流warp，并做FB拒绝，再做无GT的历史逆深度标尺对齐，
然后confidence融合；并非只把原始flow/depth拼入网络而不warp。
有效历史系数包含hard几何、FB和可见性；MFTIQ不能重新启用FB拒绝像素。
历史无效处三个模型均精确返回当前原始DA输出，纠偏也不能放宽该规则。

## DA表示、对齐与训练目标

- DA输出相对逆深度，数值越大通常越近；不是Depth Pro的米制深度，也不是保证视频统一的1/米。
  概念上 r_t≈s_t/Z+b_t；s_t、b_t不是显式网络标量输出，表示存在尺度/偏移自由度。
  光流只有像素对应，不能自行解决标尺差异或相机运动造成的Z变化。
- 历史warp后在FB有效对应点上以三次Huber重加权拟合正尺度仿射到当前预测。
  有效点<64、退化或非正尺度时禁用历史，不用GT补救。
- 特征归一化使用当前预测中位数和Q90−Q10；S统计仅来自当前帧。
- 两个refine模型采用加性逆深度纠偏，范围±0.1倍当前Q90−Q10，不用旧米制乘性修正解释DA。
  全局g=f*tanh(a)、局部q=tanh(b)，按剩余正负空间组合，始终有界，局部可抵消全局。
  全局/局部头零初始化；FiLM初始恒等。
- 当前主训练确实是逐图目标：从冻结原始DA和GT逆深度拟合每帧固定a,b并detach，
  所有预测分支复用同一映射。归一化逆深度SmoothL1，beta=0.1。
  不对改进预测重新拟合训练映射；退化使用正中位数比例，空GT返回零并记录覆盖率。
- blend/reset、全局/局部辅助目标只在loss构造，不能进入FusionInputs、缓存、DGX请求。
  blend候选相对差至少1%；reset改善阈值0.001；全局/局部以融合预测detach后的目标分解监督。
- 系数：blend 0.10、reset 0.05、纠偏正则0.01、全局/局部各0.01。
  f=0关闭纠偏及其分解监督。
- final_flow严格只交换同一对的RGB、预测、GT、双向流、有效mask，不读取第三帧。
  两方向现有损失平均，缺GT端点不稀释；双向GT变化匹配SmoothL1 beta0.1，权重0.05，
  前500步线性启用。当前这个时间目标也对两个端点分别做GT gauge，
  因而不能保证原生DA视频标尺漂移被直接惩罚。此问题已解释，尚未授权改变损失。
- 冻结前端/光流/teacher不训练，梯度只进入融合头。
- aux_depth_ratio下降而depth_loss平稳不等于主任务改善；同时看分源主损失、AbsRel及分项。
  辅助总loss不能直接在不同结构间比较。GradNorm只讨论/诊断过，未接入正式优化器。

## 数据、划分与采样

训练数据根 `~/Documents/Resources/Datasets/`；benchmark其他资源仍在
`~/Documents/ResourcesLow/Datasets/`。不要全局替换成同一个根。
主manifest `Proj/outputs/explore_a/mixed_h5_balanced/data.json`；
历史基础manifest `Proj/outputs/explore_a/kitti_tum/data.json`。
保留既有固定划分、物理房间隔离、官方测试房间及泄漏核查；不要重新划分train/test。

NYUv2 H5数据每文件含RGB+米制FP32 GT，无需为RGB再下Raw。
根 `~/Documents/Resources/Datasets/nyu_depth_v2/h5/{train,val}`。
同场景按数字编号排序，相邻保留帧编号通常差5，无时间戳；不宣称原始连续帧或0.1s。
H5已对齐，不执行Raw传感器投影。val/official 654没有可靠视频映射，不能按编号拼时序。
训练NYU H5是额外视频实验，课程NYU Labeled静态评价独立；不能混淆二者。
已有泄漏核查保留在outputs相关目录；更改数据时读取记录，不能仅凭同名/同场景推断同一视频。
TUM官方validation独立benchmark manifest，不进入训练。

当前源概率 KITTI:NYUv2:TUM=3:3:2，`sampling_mode: video_sqrt`。
`training/sampling.py::StepBatches`每优化步用seed+step建立独立随机流，先选一次数据源，
再按P(video|source)∝sqrt(该video帧对数)选视频、视频内均匀选一对。
一个effective batch内优先不同视频；候选不足才重新填充。
KITTI左右相机归同一物理录制。不是全库逐帧均匀，也不是一个microbatch换一次源。
抽样/恢复不受microbatch切分和输入预取影响。

全量split历史统计（pairs）：KITTI train68470/dev17428/test6852；
TUM47554/8514/21542；NYU38420/7466/1418。以实际manifest复核新变化。

## 公共固定dev：替代旧10%离散抽样

公共位置 `training/dev_subset.py::select_dev`，无需新增训练前准备命令。
协议 `continuous-sqrt-332-v1`，seed2026；当前3341对：KITTI1253、NYU1253、TUM835。
覆盖全部21/47/9序列，分别46/47/18片段。序列sqrt配额、至少一对、封顶、最大余数。
序列内通常一段，配额≥64优先时间分散两段；按精确RGB端点链确保连续，
不跨相机、场景、原匹配缺口。NYU仍只是保留H5帧的连续链。
模型/光流/轮次/resume/manifest输入行顺序不改变选中ID及顺序。
摘要 `9813043ebd074c37f00a35a5f0d92f6d3f43c152b8c3df080466b83555cd130b`。
训练写dev_selection.json；评价/oracle也记录ID和边界。
`dev_sample_fraction: .1`表示预算；显式full-dev可绕过子集。
旧checkpoint切换新dev时归档旧best、重新评价当前权重、建立新best基准，不删历史。

## 评价与模型选择

- img：每图对GT逆深度拟合尺度/偏移，报告absrel_img/rmse_img/delta1_img。
- video：完整评价视频共用映射；公共dev按选中连续片段共用映射，
  报告absrel_video/rmse_video/delta1_video。不能把片段当整部原视频。
- 常规各方法独立拟合；另保留基线固定映射诊断，二者不能混称严格oracle界。
- ordinal_accuracy不拟合，固定帧ID种子生成4096候选点对，GT深度比≥1.1最多保留1024，
  同预测值计错、无合格点对记NA。这是传感器GT排序诊断，不是官方DA-2K成绩。
- best按absrel_img，先物理序列再数据源等权；不是按训练总loss或test选模。
- train/dev/heldout记录相同可计算指标与分项loss，JSONL/TB/CSV保留；
  tqdm仅loss/lr，速度与ETA由进度条显示，训练每步日志。
- 视频benchmark KITTI、Sintel、Bonn沿用核对过的VDA long全范围，
  TUM官方validation全部33录制、全部有效匹配帧。禁止擅自前110帧或“首轮”截断。
  可选性能探针的小集合必须与正式精度benchmark明确隔离。
- NYU Labeled 654课程静态米制评价用UniDepthV2原始米制不对齐；DA相对分数不替代课程要求。
- 视频可视化整段固定归一化，改进结果复用基线范围，不做逐帧minmax掩盖闪烁。
- final_flow同best消融 full/no-temporal/global-only/fusion-only/shuffled-history。
  扰动只作用T历史特征，保留正常融合与有效区。另保留refine投影诊断入口。
  需用同checkpoint消融辨别单帧residual和时序贡献，oracle本身不能证明时序学习。

## 已完成完整DA oracle（不要误当待执行）

目录 `Proj/outputs/explore_a/da_continuous_dev_oracle_20261102/`，
阅读analysis.md、summary.json、protocol.json及clips/*.json；完整3341对/111片段已执行，
恢复检查通过，不依赖训练checkpoint，不根据test调参，未写共享预测NPZ。
冻结DA+Farnebäck+rule一次获取，复用8分支：raw、未对齐/已对齐half、
未对齐/已对齐凸融合oracle、additive10、权重[-1,2]外推、GT辅助历史对齐诊断。
分别使用原始DA逐图固定映射与片段固定映射，在共同历史有效区和全部GT区评分。
验证两组fixed gauge下 refine≤fusion≤raw；重新拟合SSI不声称严格嵌套。

全部GT区归一化逆深度主损失的下降（凸融合 / ±0.1分位差纠偏）：

| 来源 | 逐图固定映射 | 片段固定映射 |
|---|---|---|
| KITTI | 11.22% / 75.31% | 9.54% / 68.97% |
| NYU | 7.30% / 40.64% | 6.53% / 39.67% |
| TUM | 15.19% / 56.62% | 7.76% / 46.62% |

历史有效覆盖92.29/80.63/85.74%；GT落入候选区间8.12/7.20/10.32%；
共同偏高或偏低约90–93%；纠偏触界9.98/31.65/17.98%；历史仿射接受率100%。
说明凸融合空间有限、有界refine扩大输出范围，但不证明轻量模型可学会或时序有效。
TUM原始AbsRel img .063840/video .114276；逐图oracle refine .027197/.100082，
片段目标refine video .063430。逐图改进不能等同视频共同标尺改善。
NYU间隔与较低覆盖/较多触界可能有关，统计不能单独证明因果。
旧DepthPro/旧结构实验只作历史启发，不能复用其best代表当前模型。

## 统一hybrid与本轮调优

`online/runtime.py`统一训练、dev/test、视频benchmark、NYU/oracle冻结推理。
组件级cache/remote/local任务，端点与多头去重；缓存metadata兼容且存在才提交，
缺项提交前按远端:本机相对概率再分配。提交后损坏/身份冲突/RPC错误直接报错。
没有50ms期限、迟到丢弃、冷却或失败fallback；队列满背压，完成后消费。
本机CUDA主线程执行；等待外部结果优先执行待就绪本机任务。
DGX持久SSH，JSON长度分帧+原始FP32数组，stdout隔离日志，无pickle/临时RGB文件/GT。
FP16冻结推理、FP32输出、注意力策略保持；不同硬件不承诺逐位相同。
cache最多4、DGX最多12、send_ahead2、输入prefetch32/read_workers8。
阶段切换排空并有界保留训练任务；退出只关本次服务，不停止已有缓存worker。

已增加execution.routing.train/evaluation，阶段按提交ticket记录，不随全局切换误路由。
性能模型 t=max(H+p_host L,p_cache C,p_dgx R)，以纯路径吞吐估计，再做相同样本混合验证。
此前测得推荐train38/49/13、eval42/46/12，FinalFlow慢轮+39.6%、Guided+57.8%，
dev+test双轮中位+51.3%；冷热页缓存影响很大。仅Farnebäck/rule特定预算，不代表RAFT最优。
用户随后要求删除本轮执行结果，五个phase_routing_*输出已删除，oracle/训练结果保留。
性能工具routing_benchmark/routing_balance/routing_samples代码仍在，文档记录仍在。
调优发现并修复解码乱序+完成结果占满名额死锁：提交前注册解码ticket、按消费顺序frontier
激活、有scope取消、组件position顺序；不通过扩队列或fallback掩盖。

### 2026-11-03交接时实际config（优先于历史文档）

用户在后续交接前已进一步修改配置，不能覆盖回历史推荐：

```yaml
execution:
  routing:
    train: {cache: 0.25, remote: 0.40}       # host .35
    evaluation: {cache: 0.35, remote: 0.43}  # host .22
training:
  effective_batch: 16
  microbatch: 16
```

三个模型都继承公共设置。普通配置变化记history不强制new run；执行中不热更新。
缓存probability是尝试占比，缺项会改变实际来源，须观察实际统计。
remote_online.enabled=false全局禁远端；execution.backend=online纯本机；
不要使用已删除的旧配置/consumer或旧50ms参数。

## batch/step精确语义

当前16/16每步一个microbatch；之前16/8每步累积两次。
accumulation=effective_batch/microbatch，必须正整数整除。
每步zero_grad一次，每microbatch反向loss/accumulation，累计后clip、optimizer、scheduler各一次。
同effective batch始终一个数据源，不会因切microbatch变成混合源梯度。
减microbatch省头训练显存，保持effective batch维持样本批预算；GroupNorm不依赖batch统计。
增effective batch而不减steps会增加总样本预算；30k×16=480k帧对抽取（可能重复）。
final_flow两方向不等于新增独立视频对。冻结前端吞吐比例可能随batch改变。
OOM自动重试仅现有microbatch8→4路径，不要假设16会自动降档；当前日志里累积次数提示
存在旧常量，若修此处应按effective/micro动态计算，不改变训练数学。
更改batch后可同run恢复；不同模型/flow/occlusion/表示不能跨任务恢复。

## 位置、缓存与结果保护

公共config `Proj/configs/explore_a/mixed_h5_da2.yaml`；
refine/final为同目录mixed_h5_da2_refine.yaml、mixed_h5_da2_final.yaml，仅extends及runs覆盖。
run根 `Proj/outputs/explore_a/mixed_h5_balanced/` 下：
- runs_da2_guided_30k/<flow>-rule-unet_guided
- runs_da2_refine_30k/<flow>-rule-flow_refine
- runs_da2_final_30k/<flow>-rule-final_flow
交接检查时guided/Farnebäck已有last.pt，未加载核对step；不要声称没有可恢复结果。
旧配置和文档可能还写batch8及旧路由，后续实施应以当前YAML为准并按需修正文档。

本机cache `~/Documents/ResourcesLow/cache/Homework-AIAA3201`；DGX
`/home/hjr/Documents/Resources/cache/Homework-AIAA3201`；规范glob。
DA画布glob/frontends/da2/frames，原生glob/frontends/da2/native；共享glob/flow、glob/teacher。
缓存身份按组件、输入裁剪/画布/尺寸/精度区分，不能跨不同输入误复用。
不缓存标尺对齐/归一化/GT映射。原生可派生画布，反向不行。
NPZ FP32/DEFLATE9；生产端原子写、预分配；消费者不写回。
超过4000RGB序列只跳过缓存制作，仍训练online。NYU按H5帧数计。
原始数据与预测缓存应用层只读；last/best/identity、JSONL/session、TB/PNG/CSV是保留训练输出。
不要擅删训练输出以优化机械盘；用户刚删除的仅五个性能测量目录。
路径旧“新加卷1”不应恢复；当前训练根已迁移Documents/Resources。

## 实际操作命令（仓库根目录）

```bash
conda activate aiaa3201
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
fusion=unet_guided                    # unet_guided / flow_refine / final_flow
flow=farneback                        # farneback / raft
case "$fusion" in
  unet_guided) file=mixed_h5_da2.yaml; run=runs_da2_guided_30k ;;
  flow_refine) file=mixed_h5_da2_refine.yaml; run=runs_da2_refine_30k ;;
  final_flow) file=mixed_h5_da2_final.yaml; run=runs_da2_final_30k ;;
esac
cfg="$PWD/Proj/configs/explore_a/$file"
bash "$PWD/Proj/explore_a.sh" --config "$cfg" check-weights
python -m explore_a.remote_online.sync --config "$cfg"
bash "$PWD/Proj/explore_a.sh" --config "$cfg" train \
  --flow "$flow" --occlusion rule --fusion "$fusion"
```

只一次运行一组，避免争用本机/DGX。每1000步原子last.pt，每2000步dev；
lr1e-4/weight_decay1e-4/warmup500/clip1/AMPtrue，30k步、log_every1、heldout_every0。
停止前台Ctrl+C一次；后台kill -INT <训练主进程PID>。中断中的step不保证保存，
可恢复最近完整checkpoint。原命令重启自动last.pt，无需--resume；best.pt用于评分。
不要对cache worker/所有python进程宽泛pkill。observer持有run时不要并发重建TB。

```bash
# 消费端不要求worker同时运行；缺项可预分配online。
bash "$PWD/Proj/dgx_cache.sh" pipeline --config "$cfg" --recipe glob-da2 --workers local,dgx
# 需要补共享flow/MFTIQ才运行：
bash "$PWD/Proj/dgx_cache.sh" pipeline --config "$cfg" --recipe glob --workers local,dgx
bash "$PWD/Proj/dgx_cache.sh" status --recipe glob-da2 --host all
bash "$PWD/Proj/dgx_cache.sh" retry --recipe glob-da2 --host all
bash "$PWD/Proj/dgx_cache.sh" stop --recipe glob-da2 --host all
# 原生独立增量生产（不是启动训练前必需）：
bash "$PWD/Proj/dgx_cache.sh" native --config "$cfg" --scope manifest --split all
```

```bash
runs="$PWD/Proj/outputs/explore_a/mixed_h5_balanced"
ckpt="$runs/$run/$flow-rule-$fusion/best.pt"
out="$PWD/Proj/outputs/explore_a/da2_${fusion}_${flow}_comparison"
for scope in dev test benchmark tum-validation; do
  bash "$PWD/Proj/compare_models.sh" --config "$cfg" --checkpoint "$ckpt" \
    --output "$out" --scope "$scope" --model fusion \
    --benchmark-root "$HOME/Documents/ResourcesLow/Datasets" \
    --tum-root "$HOME/Documents/Resources/Datasets/TUM-RGBD" || break
done
# 完整dev oracle此前已经完成；如用户要求另跑才执行新输出目录：
python -m explore_a.evaluation.relative_oracle --config "$cfg" \
  --output "$PWD/Proj/outputs/explore_a/da2_dev_oracle"
```

多头evaluate-final需显式三个checkpoint，参阅DA_TRAINING.md；final_flow消融追加--ablations。
TUM下载脚本download_tum_validation.sh已准备，固定清单Proj/storage/tum/validation_plan.json，
33包共27,254,725,677字节压缩包，目标
~/Documents/Resources/Datasets/TUM-RGBD/official_validation；plan/download/extract/prepare/status
详见DA_TRAINING.md。不因交接自动下载或跑评价。

## 验证与接续方式

最新连续dev/双评分/oracle受影响回归此前87项分组通过；
路由/背压/解码死锁/来源去重/失败清理/恢复等46项分组通过，语法、100列、diff检查通过。
实际完成完整3341oracle和隔离性能实验；未因此启动正式训练或完整视频benchmark。
最终路由改动已同步DGX；交接前用户新增的纯config值不等于运行中已生效。
后续改代码后重新同步DGX服务，既有权重/数据不随同步传输。
测试环境PYTHONPATH=Proj/src、MPLCONFIGDIR=/tmp/aiaa3201-mpl、OMP_NUM_THREADS=1。
合成/替身验证应覆盖表示、无历史精确回退、GT隔离、范围/梯度、双向交换、冻结无梯度、
缓存只读、组件去重、错误报错、阶段切换、checkpoint恢复。真实GPU仅受影响短程验证。

研究启发：CODD的reset/融合分工、PatchFusion全局上下文、LapDepth结构/残差分工、
时序对齐与可靠性门；均为项目适配，不声称完整复现其架构。
模型容量/源梯度竞争是历史讨论问题，尚无证据可把所有室内学习弱归因于容量。
不得根据test调采样、loss、结构；多源优化步同源这一事实影响GradNorm可行性。

新对话职责：继续构建、修改、排错和按用户指令执行训练；先读取本文件与实际config，
概括接续状态后等待用户具体任务。仅创建交接不授权重新调参、训练、下载或删除文件。
