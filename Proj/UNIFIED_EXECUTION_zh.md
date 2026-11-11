# 统一训练、dev 与多模型评估

2026-11-04。模型列表的唯一维护入口是 `maintenance/evaluate_models.json`。
旧名称 `evaluation_models.json` 是指向该文件的符号链接。

## 现在如何运行

在仓库根目录运行；暂时排除尚未下载完成的 TUM：

```bash
cd /home/hjr/Documents/Projects/Homework-AIAA3201
/home/hjr/miniconda3/envs/aiaa3201/bin/python \
  Proj/maintenance/evaluate_trained_models.py run \
  --scopes benchmark test \
  --output Proj/outputs/explore_a/shared_models_evaluation_v2
```

TUM 准备完成后，在同一会话追加该 scope：

```bash
/home/hjr/miniconda3/envs/aiaa3201/bin/python \
  Proj/maintenance/evaluate_trained_models.py run \
  --scopes tum-validation \
  --output Proj/outputs/explore_a/shared_models_evaluation_v2
```

只重新生成汇总表，不运行模型：

```bash
/home/hjr/miniconda3/envs/aiaa3201/bin/python \
  Proj/maintenance/evaluate_trained_models.py report \
  --output Proj/outputs/explore_a/shared_models_evaluation_v2
```

产物包括 `comparison.md`、`comparison.csv`、`comparison.json`；
`scope/dataset/case/model/metrics.json` 是每个模型在该 case 上的原子提交记录。
`cohort.json` 记录实际输入，`session.json` 记录评估协议和模型实现/权重身份。
`--models NAME ...` 选择注册模型，`--scopes dev` 使用固定 dev 子集，
`--full-dev` 使用完整 dev。benchmark 默认 long；TUM 始终使用完整匹配验证序列。
benchmark 和 TUM 的默认数据根目录分别为 ResourcesLow 与 Resources 中的现有目录。

当前列表包含两个 guided checkpoint、已完成的 final_flow checkpoint 和DA2 FP32 论文参考项和一个 UniDepth V2 参考模型。
DA2 基线自动产生：`raw_canvas` 对应融合头实际使用的输入，
视频 benchmark 另有 `raw_native`，对应原始输入分辨率的 DA2。
两者输入尺度不同，因此保留两个有明确名称的基线；各自均只提交一次。

每个 checkpoint 首次运行时固定到会话中的小型权重快照；原始 `best.pt` 后续更新
不会悄悄替换已经进行中的比较。评估更新后的 best 时，使用新的模型名称或新输出目录。
添加模型后重复同一命令：已完成模型与公共基线继续复用提交记录，新增模型单独补齐。
原有逐模型输出与本次新协议分开存放，不能混入同一输出目录。

## 计算与调度

执行顺序是 scope → case → 多个模型。case 是物理序列、相机与已选连续 clip 的组合。
同一输入的 DA2、同一光流设置的双向 flow、所需的 MFTIQ，以及相应几何预处理
在运行中共享；不同 head 顺序前向。GT 解码与有效 mask 也共用。
基础 SSI 统计量写入指标记录，之后新增模型可以复用基线固定 gauge。
结果记录保存指标，不保存新的持久化预测缓存。

训练仍由 host 执行 head、loss、反向传播和优化器更新，cache/DGX 只供应冻结组件。
训练的 sampler、loss、有效 batch、优化器与 checkpoint 参数名称不变。
dev/held-out 共用 collector 与模型接口；dev 的选模分数仍读取既有 `ssi_abs_rel`，
沿用原来的 NumPy 平均计算，先按物理序列、再按数据集平均。
补充报告的 `absrel_img`、`absrel_video` 不替代既有选模计算。
dev 在临时评估上下文运行，恢复训练模式和随机数状态，并复用 host 已加载的冻结模型。

完整推理时，host 与 DGX 各有独立调度线程。host 可以从 cache 读已有组件；
DGX 可以执行 DA2、flow、teacher 和多个训练 head。
调度先观察该 scope 的 case 大小，再按固定 seed 和 case ID 分配；DGX 的总输入帧数
不超过配置预算。完整 case 的粒度可能使实际 DGX 比例低于设置值。
执行归属与模型列表顺序、解码线程完成顺序无关。
两端独立消费任务，RGB/结果只保留有界在途窗口，不把整个视频载入 RAM。
序列参考模型使用临时磁盘 spool；VDA 保留上游 32 帧窗口、关键帧与对齐算法。

DGX 启动前自动同步项目 Python 源码并检查身份；传输小型 head 快照，不传优化器或 GT。
注册的冻结参考模型权重和适配器声明的 `runtime_paths` 也按需同步。
依赖包须在目标架构安装，不能把 x86 二进制依赖直接复制到 ARM。
这次已在 DGX 的 `Proj/.runtime/unidepth-deps` 隔离安装 UniDepth 导入所需依赖，
没有替换训练环境已有包。

## 中断训练与恢复

原来的训练启动指令继续使用，仍从对应运行目录的 `last.pt` 恢复；
模型、优化器、scheduler、AMP scaler、步数以及 Python/NumPy/Torch/CUDA RNG 均保留。
`Ctrl+C` 恢复到最近成功保存的 `last.pt`，保存间隔之后的步数重跑。
临时文件写入成功后才原子替换正式 checkpoint。

首次从旧执行记录恢复时，会备份历史 best，在当前执行配置下分别重评恢复权重和历史 best，
重新确定 best 分数后继续训练。备份是复制，重评期间保留可用的 `best.pt`；
重评中断可以重试，也能恢复旧实现先 rename、随后中断留下的 best 备份。
此过程不改变当前训练模型或优化器状态。
这里的重评针对执行归属、GPU/批次等变化带来的数值差异，不代表更换评估指标。

`farneback + final_flow` 的回归检查覆盖：中途异常、旧格式 checkpoint、
评估迁移再次中断、旧版 rename 中断遗留状态，以及最终恢复后的权重、优化器与 scheduler
与不中断训练逐项一致。

## 指标口径

| 字段 | 含义 |
| --- | --- |
| `absrel_img` / `rmse_img` / `delta1_img` | 每张图拟合逆深度 scale/shift 后评分 |
| `absrel_video` / `rmse_video` / `delta1_video` | 整个序列/已选 clip 共用一个逆深度 scale/shift |
| `*_baseline_fixed_video` | 融合分支使用 raw_canvas 的整段 gauge，不为融合结果重新拟合 |
| `ordinal_accuracy` | 同一帧 ID 固定抽样的 GT 深度排序正确率 |
| `metric_*` | 仅对输出米制深度的参考模型报告的无对齐指标 |

前两组在所有 scope 使用相同实现；基础相对指标的拟合使用 GT，
不能解释为模型已经恢复了米制尺度。该表统一跨数据集比较口径，不冒用论文原表的数值。
每个物理序列内部按帧统计，再等权汇总序列；分别报告各数据集。
固定基线 gauge 只应用于共用 DA2 表示空间的融合分支。

## 新增模型

新增训练结果只需追加配置、checkpoint 和唯一名称。例如：

```json
{
  "name": "farneback_final_flow_lr1e-4",
  "config": "configs/explore_a/mixed_h5_da2_final.yaml",
  "checkpoint": "outputs/explore_a/mixed_h5_balanced/runs_da2_final_30k/farneback-rule-final_flow/best.pt"
}
```

共同评测使用相同的数据、canvas、冻结前端与几何协议；配置不一致会明确报错，
避免用第一个模型的输入设置覆盖其他模型。

冻结参考模型使用 `kind: "frozen"`，内建 adapter 有 `da2`、`unidepthv2`、`vda`。
示例（按需要追加到 JSON 列表）：

```json
{
  "name": "vda_s_fp32",
  "kind": "frozen",
  "adapter": "vda",
  "weights": "checkpoints/video_depth_anything_vits.pth",
  "load_param": {"fp32": true, "input_size": 518},
  "specific_in": {},
  "specific_param": {"fps": 1.0}
}
```

自定义 adapter 用 `package.module:Class`，不需要修改评估模型枚举。
构造器接受 `weights` 和所需加载参数；`load_param` 中的参数传给构造器。
`capability` 声明 `image` 或 `sequence`；可声明项目内的 `runtime_paths`，
也可在注册项中提供 `sync_paths`，以同步额外源码。
输入是 `ModelInput(rgb, frame_ids, timestamps, reset_history, specific_in)`；
模型实现 `forward(request, **specific_param)`，返回 `ModelOutput`，包含
Tensor `prediction`、`representation`、`frame_ids` 和 `specific_out` 字典。
冻结参考 adapter 应为每个 RGB 返回一张 HW 深度图；sequence adapter 返回 N×H×W。
表示类型是 `relative_inverse` 或 `metric_m`。
训练前向不 detach；传输与落盘边界才 detach。
字典传输支持嵌套 dict/list/tuple、标量、NumPy 数组及 Tensor，保留 Tensor dtype
（包括 BF16）和标量/空数组形状；不会使用 pickle 传输。

## 比例的测量依据

仅使用固定的 RAFT guided best 进行吞吐调优。
三个独立随机轮次，每轮 128 对，分别测 local、cache、DGX 冻结组件和 DGX 完整推理。
预热不计入稳态计时；cache 在每个轮次先测，轮次之间使用不重复的文件。
cache 三轮物理读取约 454 / 454 / 446 MB。
随机抽样和物理读取记录降低热缓存偏差，不等同于强制清空系统 page cache。

测得冻结 local 约 50.2 ms/对，cache 保守约 92.6 ms/对，
DGX 冻结组件约 116.5 ms/对，DGX 完整推理约 140.6 ms/对。
host head 纯推理约 5.2 ms/对，训练 forward/backward/update 约 6.1 ms/对。

容量模型令 local/cache/DGX 比例为 l/c/r，且 l+c+r=1。
训练 host 约束为 `l*local_frozen + head_train`，DGX 仅计算冻结组件；
完整推理的 host 约束为 `l*(local_frozen+head_eval)+c*head_eval`，DGX 计算完整 head。
cache 与 DGX 各有独立吞吐约束，求最小瓶颈时间，再将 DGX 比例向下取整到 0.1。
理论训练 DGX 比例约 0.245，推理约 0.203，因此都保守设为 0.2。
这是不同成本模型得到的结果，两个阶段没有共用一套 head 成本公式。

默认训练组件份额为 cache 0.32、DGX 0.20、local 0.48。
完整推理的 DGX case 帧数预算为 0.20；host 内部 cache 份额为 0.38，
对应全局约 0.30，其余由 host 在线计算。cache 缺失按既有规则分配到在线端。

最后一轮使用另外两组各 256 对的样本，避开前面测量与复测使用过的文件：

| 工作 | 纯 local | 混合执行 | 实测速率提升 |
| --- | ---: | ---: | ---: |
| 训练（配置中的 batch、AMP、backward、clip、零 LR 临时优化器） | 56.1 ms/对 | 36.7 ms/对 | 约 1.53× |
| 完整推理（单个 RAFT guided head） | 54.5 ms/对 | 47.7 ms/对 | 约 1.14× |

最后一轮训练/推理混合测量仍分别产生约 292 / 275 MB 物理读取。
推理实分 DGX 48/256 帧（18.75%），符合完整 case 粒度和 20% 上限。
这里测量输入准备与模型计算，不包含正式全量评分、所有参考模型或结果制图。
其他 head/flow 的吞吐会随计算量变化；这组数据用于当前共享配置的保守默认值。

记录保存在 `outputs/explore_a/execution_tuning_20261104/`、
`execution_tuning_20261104_recheck/` 和 `execution_tuning_20261104_final/`。
后两者保留调度修正后的独立样本复测，最终采用 `final` 的记录。
所有测量使用临时模型/零 LR 优化器，没有覆盖正式训练 checkpoint。

重新调优的入口：

```bash
PYTHONPATH=Proj/src OMP_NUM_THREADS=1 \
/home/hjr/miniconda3/envs/aiaa3201/bin/python \
  -m explore_a.audits.execution_tuning \
  --pairs 256 --rounds 3 --seed 20261107 \
  --output Proj/outputs/explore_a/execution_tuning_new
```

`--reuse-costs PATH/balance.json` 可以复用单端成本、只验证调度；
`--exclude-protocol PATH/protocol.json` 可重复指定，排除已经测过的文件。
新测量会保存预热、单端、混合验证的全部 sample ID。

## `_raw` 差异的结论与验证

旧调度由最早完成解码的 occurrence 选择执行端，当前/上一帧复用又采用先提交者，
因此同样输入可能落在不同 GPU。现改为冻结输入身份固定路由，完整推理再固定 case 归属。
真实权重检查中，同一端重复计算逐元素一致；RTX 4090 D 与 GB10 的 FP16 输出存在差异。
融合中的阈值/可见性判断还可能放大个别像素差异，因此没有假设跨 GPU 逐元素相等。

`model_contract.json` 确认 DA2、UniDepth V2 和跨窗口 VDA 的统一接口与原路径最大误差均为 0。
`shared_smoke/results.json` 记录两训练模型在三个来源的端内重复一致性、
UniDepth 相机输入与 confidence/intrinsics/points/rays 等输出字典的 RPC 检查，
以及 live dev 实际 host 7 帧、DGX 1 帧，训练模式、梯度开关、RNG 保持不变的检查。

## 2026-11-05：Sintel 使用已复现的 AC 标准

后续 benchmark 的 Sintel 一律使用 training/final 的 23 段全部 1064 帧，缺失 final 输入会明确报错，不回退到 clean。逐图 AbsRel/δ1/RMSE：GT 先截断至 72 m，逐图拟合逆深度 scale/shift，允许负 scale；对齐后的逆深度下限为 1/80，转回深度并截断至 [1e-5,80] m，全部 1064 帧等权平均。最远处也参与逐图评分，不能沿用视频 GT<70 m 的排除规则。

整视频指标仍在 final 连续输入上按 GT<70 m、预测截断至70 m、整段逆深度统一对齐计算，再按段等权平均；它不能与历史 clean 结果直接比较。KITTI 继续使用原连续视频 cohort。训练/dev 的既有选模指标和 checkpoint 恢复规则不改变。

共享评估、独立 benchmark 和参考模型入口使用同一 Sintel 评分函数。现有模型推理精度与训练前端保持一致；AC 论文基线核验另采用 FP32 和 align_corners=False，见审计报告，不把运行精度差异隐藏为完全相同的推理设置。

默认输出目录为 shared_models_evaluation_v2。旧版结果不迁移、不合并；显式指向旧输出目录会提示换目录。结果中的 image_standard、scoring_protocols.json 和 session.json 固定实际标准；DGX 在下次启动评估前自动同步源码，新 final RGB 路径有独立缓存身份。

`da2_fp32` 是明确注册的论文参考项：FP32、短边518、输出插值 align_corners=False，与本次 AC 复现配置一致；它不与训练前端的 FP16 宽屏缩放输出混称。`raw_canvas` / `raw_native` 仍描述训练前端在两个空间分辨率下的实际输出。不同数值设置不能共用一份预测，同一注册项仍每个输入只推理一次，所有融合头继续共享其训练前端计算。

验证：生产评分函数用全部1064帧保存预测复算，得到 AbsRel=0.23951921997621878、δ1=0.7122263998181737，与未修改的 AC 评分源码一致。实际注册项还通过一帧端到端推理与已有复现预测的逐位比较。核验记录位于 outputs/audits/sintel_kitti_20261105/production_ac_validation.json 与 production_ac_inference_validation.json。
