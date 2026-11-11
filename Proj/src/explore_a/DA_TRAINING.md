# DA-V2-S：三模型与视频评价

方向 A 的冻结前端统一为 Depth Anything V2 Small，输出相对逆深度。训练头为
`unet_guided`、`flow_refine`、`final_flow`；两种光流为 `farneback`、`raft`。
默认遮挡使用 `rule`，可显式选择冻结 `mftiq`。原有训练和评价结果已作废并删除。

`unet_guided` 在对齐后的逆深度候选之间凸融合；`flow_refine` 使用双帧编码、
ConvGRU、全局上下文与局部纠偏；`final_flow` 使用当前帧 S 分支、显式时序增量 T
和严格两帧双向监督。两种纠偏均限制在 ±0.1 倍当前帧 10%–90% 分位差，
历史无效时精确返回原始当前预测。GT 仅用于损失和评分，不进入前向或远端请求。

训练保持 KITTI:NYUv2:TUM=3:3:2，先按视频帧对数平方根抽序列，再抽帧对；
30,000 步，有效 batch 8，固定 3,341 对连续片段 dev，每 2,000 步 dev，best 按逐图对齐 AbsRel。
开发、测试的 AbsRel/RMSE/δ 是相对逆深度 SSI 对齐后的深度指标，不是未对齐米制结果。

三个配置文件是同一实验设置的三个入口：`mixed_h5_da2.yaml` 包含公共设置，
`mixed_h5_da2_refine.yaml` 和 `mixed_h5_da2_final.yaml` 用 `extends` 继承它，
仅覆盖各自的 `paths.runs`。共享参数只需在公共文件修改一次；checkpoint、
原始 manifest 和缓存身份保持原语义。旧的双前端／两源／online-control 配置已移除。
需要纯本机对照时，在公共配置设置 `execution.backend: online`，不另建配置副本。
配置加载后会补齐固定的 DA、缓存模式、可观测性默认项，生成的 worker YAML 则完全展开。

## 环境与同步

以下命令从仓库根目录执行：

```bash
conda activate aiaa3201
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
cfg="$PWD/Proj/configs/explore_a/mixed_h5_da2.yaml"
bash "$PWD/Proj/explore_a.sh" --config "$cfg" check-weights
python -m explore_a.remote_online.sync --config "$cfg"
```

冻结组件统一经阻塞式 hybrid 分配：训练缓存/DGX/本机=38%/49%/13%，
评价=42%/46%/12%。缓存提交前缺项按该阶段 DGX:本机比例重新分配；
提交后损坏、身份冲突或 RPC 失败直接报错。所有阶段复用
相同流水线；本机 CUDA 由主线程执行，DGX 仅传内存 RGB／FP32 预测，不传 GT。
消费端不写预测缓存。更改配置和代码后需要重启，运行中不热更新。

## 训练、停止和恢复

一次选择一个模型和光流，避免同时启动六组争用同一 GPU：

```bash
fusion=unet_guided                    # unet_guided / flow_refine / final_flow
flow=farneback                        # farneback / raft
case "$fusion" in
  unet_guided) file=mixed_h5_da2.yaml ;;
  flow_refine) file=mixed_h5_da2_refine.yaml ;;
  final_flow) file=mixed_h5_da2_final.yaml ;;
esac
cfg="$PWD/Proj/configs/explore_a/$file"
bash "$PWD/Proj/explore_a.sh" --config "$cfg" train \
  --flow "$flow" --occlusion rule --fusion "$fusion"
```

每 1,000 步原子保存 `last.pt`。终端按一次 Ctrl+C 停止；可以恢复到最近已完成的
checkpoint，不承诺保存中断中的优化步。后台进程用 `kill -INT <训练主进程PID>`。
中断时先关闭本次冻结/评价 RPC，再取消排队读取；CPU 读取线程不阻塞进程退出，
不在中断清理时重绘曲线。正常运行仍完整消费任务，不增加推理超时或失败回退。
`explore_a.sh` 默认设置 `OMP_NUM_THREADS=1`，保留用户显式设置；冻结 DGX 服务沿用该值。
恢复重新执行相同命令，自动加载该模型目录的 `last.pt`，不需要 `--resume`。
首次训练没有旧结果可恢复。`best.pt` 用于评价，`last.pt` 用于续训。

如确实需要串行运行全部六组，由用户执行：

```bash
bash "$PWD/Proj/explore_a.sh" --config "$cfg" run-matrix --train-only
```

## 预测缓存与共享光流

DA 画布预测位于 `glob/frontends/da2/frames`，原生预测位于
`glob/frontends/da2/native`。`glob/flow` 和 `glob/teacher` 为共享组件。
`glob` 生产任务仅生成光流和 MFTIQ；`glob-da2` 任务仅生成 DA 预测。
已有有效共享文件、生产归属与完成记录保留，不重压缩。

```bash
# 增量 DA 画布预测；本机与 DGX 两端 worker 由 pipeline 管理。
bash "$PWD/Proj/dgx_cache.sh" pipeline --config "$cfg" \
  --recipe glob-da2 --workers local,dgx
# 仅在需要补充共享光流／MFTIQ 时执行；不会重新生成已删除的前端预测。
bash "$PWD/Proj/dgx_cache.sh" pipeline --config "$cfg" \
  --recipe glob --workers local,dgx
# 状态、显式 worker 与重试入口保持可用。
bash "$PWD/Proj/dgx_cache.sh" status --recipe glob-da2 --host all
bash "$PWD/Proj/dgx_cache.sh" worker --host local
bash "$PWD/Proj/dgx_cache.sh" retry --recipe glob-da2 --host all
bash "$PWD/Proj/dgx_cache.sh" stop --recipe glob-da2 --host all
# 在 DGX 激活 aiaa3201 后，仓库根目录手动 worker：
# bash "$PWD/Proj/dgx_cache.sh" worker --host dgx
# 独立生产原生预测，消费时可派生画布；首次运行不是必要前提。
bash "$PWD/Proj/dgx_cache.sh" native --config "$cfg" --scope manifest --split all
bash "$PWD/Proj/dgx_cache.sh" native --config "$cfg" --scope benchmark --setting long \
  --benchmark-root "$HOME/Documents/ResourcesLow/Datasets" \
  --datasets kitti sintel bonn tum-validation
```

缓存 pipeline 的任务传输沿用原确认与临时 RGB 清理流程。停止使用原 `stop` 入口；
训练只读缓存，不依赖缓存 worker 同时运行。DGX 内存服务同步不传权重或数据。

## TUM 官方 validation 下载与准备

已提供全部 33 段的固定计划，压缩包合计 27,254,725,677 字节；不是解压后大小。
目标为 `~/Documents/Resources/Datasets/TUM-RGBD/official_validation`。
隐藏真值为轨迹，包内 RGB／传感器深度用于本项目深度评价，不提交轨迹。

```bash
plan="$PWD/Proj/storage/tum/validation_plan.json"
bash "$PWD/Proj/download_tum_validation.sh" status --manifest "$plan"
bash "$PWD/Proj/download_tum_validation.sh" download --manifest "$plan" --jobs 2
bash "$PWD/Proj/download_tum_validation.sh" extract --manifest "$plan"
python -m explore_a.data.tum_validation --plan "$plan" \
  --output "$HOME/Documents/Resources/Datasets/TUM-RGBD/official_validation/prepared/manifest.json"
```

下载中断后重跑 `download`；已完成压缩包按清单大小复用。若需从官方重新生成计划，
指定一个不存在的新 manifest，避免覆盖固定计划：

```bash
bash "$PWD/Proj/download_tum_validation.sh" plan \
  --manifest "$PWD/Proj/storage/tum/validation_new_plan.json" \
  --output "$HOME/Documents/Resources/Datasets/TUM-RGBD/official_validation"
```

## dev、held-out 和四个视频 benchmark

下列示例选择 guided/Farnebäck 的 best；更换模型时同时更换 cfg、fusion 和 checkpoint。
另外两组 run 名为 `runs_da2_refine_30k`、`runs_da2_final_30k`，模型名各自一致。

```bash
fusion=unet_guided                    # unet_guided / flow_refine / final_flow
flow=farneback                        # farneback / raft
case "$fusion" in
  unet_guided) file=mixed_h5_da2.yaml; run=runs_da2_guided_30k ;;
  flow_refine) file=mixed_h5_da2_refine.yaml; run=runs_da2_refine_30k ;;
  final_flow) file=mixed_h5_da2_final.yaml; run=runs_da2_final_30k ;;
esac
cfg="$PWD/Proj/configs/explore_a/$file"
runs="$PWD/Proj/outputs/explore_a/mixed_h5_balanced"
ckpt="$runs/$run/$flow-rule-$fusion/best.pt"
out="$PWD/Proj/outputs/explore_a/da2_${fusion}_${flow}_comparison"
for scope in dev test benchmark tum-validation; do
  bash "$PWD/Proj/compare_models.sh" --config "$cfg" --checkpoint "$ckpt" \
    --output "$out" --scope "$scope" --model fusion \
    --benchmark-root "$HOME/Documents/ResourcesLow/Datasets" \
    --tum-root "$HOME/Documents/Resources/Datasets/TUM-RGBD" || break
done
```

`benchmark` 为 KITTI／Sintel／Bonn 的既定 VDA long；`tum-validation` 使用全部录制、
全部有效匹配帧，没有额外截断。TUM 不加入训练 manifest。视频报告逐分支整段 SSI，
另报告复用基线整段映射的指标与时序误差，不用逐帧对齐掩盖闪烁。
`baseline-<flow>-raw` 指同画布原始 DA，视频 `raw_native` 为原生 DA。

如需额外 UniDepthV2 和独立 DA 参考，把 `--model fusion` 改为 `--model all`。
这会增加参考模型计算；独立参考路径及精度保持原实现。

`evaluate-part2` 也支持两种光流和四个数据集：

```bash
bash "$PWD/Proj/explore_a.sh" --config "$cfg" evaluate-part2 \
  --checkpoint "$ckpt" --output "$out/video-fusion" --setting long \
  --datasets kitti sintel bonn tum-validation \
  --benchmark-root "$HOME/Documents/ResourcesLow/Datasets" \
  --tum-root "$HOME/Documents/Resources/Datasets/TUM-RGBD"
```

`final_flow` 的同 checkpoint 五项消融在上述命令追加 `--ablations`。
NYU labeled 单图不属于四个视频集合；可单独 `compare_models.sh --scope nyu`，
明确无历史精确回退。DA oracle 使用完整固定 dev：

```bash
python -m explore_a.evaluation.relative_oracle --config "$cfg" \
  --output "$PWD/Proj/outputs/explore_a/da2_dev_oracle"
```

三个显式 checkpoint 的 held-out 共用冻结前端比较（含 FinalFlow 五项模式）：

```bash
flow=farneback
runs="$PWD/Proj/outputs/explore_a/mixed_h5_balanced"
bash "$PWD/Proj/explore_a.sh" --config "$cfg" evaluate-final --split test \
  --guided-checkpoint "$runs/runs_da2_guided_30k/$flow-rule-unet_guided/best.pt" \
  --refine-checkpoint "$runs/runs_da2_refine_30k/$flow-rule-flow_refine/best.pt" \
  --final-checkpoint "$runs/runs_da2_final_30k/$flow-rule-final_flow/best.pt" \
  --output "$PWD/Proj/outputs/explore_a/da2_three_heads_$flow"
```

## Continuous development cohort and quantitative video evaluation

The shared `select_dev` entry now uses protocol `continuous-sqrt-332-v1` instead of
scattered 10% pairs. No preparation command is required. With the current manifest it
selects 3,341 pairs: KITTI 1,253, NYUv2 H5 1,253, TUM 835; all 21/47/9 sequences remain.
Quotas follow square-root sequence size. Exact existing endpoints form clips; camera,
scene and original matching gaps are never bridged. The seed is 2026. Model, flow,
validation iteration and manifest row order do not change the selection.
Training writes `dev_selection.json`; comparison and oracle also record IDs and clip
boundaries. Current cohort SHA-256:
`9813043ebd074c37f00a35a5f0d92f6d3f43c152b8c3df080466b83555cd130b`.
Full-dev comparison remains available with `--full-dev`.

`absrel_img`, `rmse_img`, `delta1_img` fit each image to GT inverse depth.
`absrel_video`, `rmse_video`, `delta1_video` fit one gauge per whole evaluated video,
or per selected dev clip. Disconnected dev clips never share a gauge. These are
GT-assisted relative scores, not raw metric accuracy. Existing `ssi_*` image fields
remain available. Best selection retains image AbsRel, equal sequence then source
weighting. On resume with a changed cohort, old best is archived, current weights are
re-evaluated, and the new best comparison is initialized on that cohort.

`ordinal_accuracy` uses raw inverse-depth ordering with deterministic frame-ID point
pairs: at most 4,096 candidates, GT depth ratio at least 1.1, at most 1,024 retained.
Ties are incorrect; absent eligible pairs yield no score. It is a project sensor-GT
ranking diagnostic, not the DA-2K annotation benchmark. All branches use the same pairs.

NYUv2 H5 is an additional video experiment like KITTI/TUM. The course NYUv2 Labeled
654-image metric evaluation stays separate: UniDepthV2 raw meters, no test alignment.
The DA video metrics do not replace that requirement. Native/canvas differences and
existing benchmark masks/crops remain explicit. Video visualizations still use fixed
clip-wide normalization anchored to the unchanged baseline.

### Complete development oracle

```bash
conda activate aiaa3201
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
cfg="$PWD/Proj/configs/explore_a/mixed_h5_da2.yaml"
python -m explore_a.evaluation.relative_oracle --config "$cfg" \
  --output "$PWD/Proj/outputs/explore_a/da_continuous_dev_oracle_20261102"
# Separate RAFT run; same fixed cohort and eight oracle candidates:
python -m explore_a.evaluation.relative_oracle --config "$cfg" --flow raft \
  --output "$PWD/Proj/outputs/explore_a/da_continuous_dev_oracle_raft"
```

The same command resumes. Complete clips commit atomically; partial CPU staging is
reused. The run evaluates every selected pair with `--flow farneback` (default) or
`--flow raft`, both with rule occlusion and the configured cache/DGX/local pipeline.
Use a separate output directory for each flow. Resume checks the flow and, for RAFT,
its weight identity; existing Farneback results remain resumable with the default command.
GT stays local; shared prediction NPZ files are read-only.
Temporary `.npy` inference staging is under the oracle output on the system filesystem,
not in the shared cache; completed clips remove it. Allow about 20 GB staging space.
Other evaluation paths use automatically cleaned system temporary files for bounded-RAM
video scoring; this output I/O is separate from frozen prediction cache production.

Image-fixed and clip-fixed baseline gauges are two distinct oracle objectives.
Within each objective all candidates share the original DA gauge and common mask.
The primary inverse-depth Smooth L1 checks the nested bound:
refine oracle <= convex fusion oracle <= raw DA. Full-GT scores expose limited history
coverage. Independently refitted image/video scores and ranking are secondary diagnostics;
they are not strict bounds. A per-pixel conditional oracle does not prove that the
lightweight model can learn the solution.

Training and resume retain the same commands (Ctrl+C stops; rerun restores `last.pt`):

```bash
bash "$PWD/Proj/explore_a.sh" --config "$cfg" \
  train --flow farneback --occlusion rule --fusion unet_guided
```

Use `mixed_h5_da2_refine.yaml` with `--fusion flow_refine`, or
`mixed_h5_da2_final.yaml` with `--fusion final_flow`; replace flow with `raft` as needed.
Comparison commands are unchanged; a fresh output directory separates the new score
protocol from previous runs:

```bash
bash "$PWD/Proj/compare_models.sh" --config "$cfg" --checkpoint "$ckpt" \
  --output "$out" --scope dev --model fusion
bash "$PWD/Proj/compare_models.sh" --config "$cfg" --checkpoint "$ckpt" \
  --output "$out" --scope benchmark --model fusion \
  --benchmark-root "$HOME/Documents/ResourcesLow/Datasets"
```


## 按阶段的实测调度比例

公共配置在 `execution.routing` 下分别控制 `train` 与 `evaluation`：

```yaml
execution:
  backend: hybrid
  routing:
    train: {cache: 0.38, remote: 0.49}
    evaluation: {cache: 0.42, remote: 0.46}
```

本机比例为剩余部分。`evaluation` 覆盖 dev/test/benchmark/NYU/oracle；它不改变
指标协议。比例按任务提交时的阶段决定，不会被嵌套dev或异步解码时的全局阶段覆盖。
缓存比例是尝试分配比例；缺项在提交前按剩余远端/本机比例分配，实际命中可能更低。
`execution.backend: online` 强制全部本机；`cache` 强制缓存。`remote_online.enabled`
仍是DGX总开关。全局旧概率可作未配置阶段策略的默认值，当前三份配置只通过上述
公共配置管理比例，避免重复编辑。生成的hybrid consumer也保留阶段策略。

这些策略只记录执行历史，保持原run与checkpoint，不重新训练或重划dev。
运行中的进程不热更新；停止后用原命令恢复即可。冻结预测精度、模型与注意力不变。
同时修复了解码乱序时后续完成结果占满外部名额的死锁：解码前登记顺序边界，
按帧对及组件消费顺序提交，取消未启动的解码ticket；不增加队列容量或丢弃结果。

### 公式与实测口径

每帧对本机冻结成本L、缓存有效成本C、DGX有效成本R、头与评分等其余成本H：

`t = max(H + p_host*L, p_cache*C, p_dgx*R)`，三者概率之和为1。

C/R通过纯路径的饱和吞吐测量，含既有并发、解压/传输；不把重叠RPC耗时相加。
计算器用容量约束求最小t，再根据缓存可用率把实际使用份额换成配置中的尝试比例。
随机训练使用较慢缓存轮次，dev/test使用首次及重复读取的合并成本。
这是流体吞吐近似；有界队列、逐对消费、batch屏障及冷热页缓存使理论预测偏乐观，
必须用相同固定样本验证。实测没有更改模型选择或利用test精度调参。

当前验证使用Farnebäck+rule，本机RTX4090D及DGX，FP16冻结/FP32输出，batch8；
临时final_flow包含双向监督、反传、AdamW与训练指标；dev/test包含逐图与片段评分。
排除模型加载/预热、正式TensorBoard/checkpoint/PNG写入，不作为论文单机模型延迟。
RAFT、MFTIQ及原生视频benchmark并未分别调优；可用相同工具另测，不声称此比例全局最优。
测速本机环境 `OMP_NUM_THREADS=1`，使用这一设置可更接近记录条件。

### 无网格的重新测量

以下命令仅写独立性能输出与临时CPU文件，不保存训练权重、不写预测缓存。
所有比例最终仍须人工或经对照确认后写入配置，工具不会自动修改正式配置。

```bash
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
cfg="$PWD/Proj/configs/explore_a/mixed_h5_da2.yaml"
perf="$PWD/Proj/outputs/explore_a/my_routing_measurement"
# 完整纯本机任务，提供头/评分等开销H。
python -m explore_a.audits.routing_benchmark --config "$cfg" \
  --output "$perf/host_full" --candidates 0:0 --steps 16 --rounds 2 --balanced-sources
# 新的固定性能样本；先检查缓存齐全子集，三条纯路径使用完全相同输入。
python -m explore_a.audits.routing_benchmark --config "$cfg" \
  --output "$perf/paths" --candidates 0:0 1:0 0:1 --steps 16 --rounds 2 \
  --frozen-only --cache-available-only --sample-seed 2027 --balanced-sources
python -m explore_a.audits.routing_balance --coarse "$perf/host_full/summary.json" \
  --paths "$perf/paths/summary.json" --output "$perf/balance.json"
# 对照当前示例计算出的训练比例；新机器按balance.json替换候选值。
python -m explore_a.audits.routing_benchmark --config "$cfg" \
  --output "$perf/confirm_train" --candidates 0.2:0.28 0.38:0.49 \
  --phases train --steps 16 --rounds 2 --balanced-sources --sample-seed 2028
python -m explore_a.audits.routing_benchmark --config "$cfg" \
  --output "$perf/confirm_eval" --candidates 0.2:0.28 0.42:0.46 \
  --phases dev test --steps 16 --rounds 2 --balanced-sources --sample-seed 2028
```

`--sample-seed`只选择性能样本，正式固定dev与训练采样均不改变。新dev性能样本
始终来自公共固定dev内部。test只测吞吐，精度结果丢弃，不据此选择模型或损失。
同一输出目录可恢复已提交完成的trial；改变配置/样本/协议应使用新的性能目录。
