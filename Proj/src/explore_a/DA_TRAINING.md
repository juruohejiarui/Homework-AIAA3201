# DA-V2-S：三模型与视频评价

方向 A 的冻结前端统一为 Depth Anything V2 Small，输出相对逆深度。训练头为
`unet_guided`、`flow_refine`、`final_flow`；两种光流为 `farneback`、`raft`。
默认遮挡使用 `rule`，可显式选择冻结 `mftiq`。原有训练和评价结果已作废并删除。

`unet_guided` 在对齐后的逆深度候选之间凸融合；`flow_refine` 使用双帧编码、
ConvGRU、全局上下文与局部纠偏；`final_flow` 使用当前帧 S 分支、显式时序增量 T
和严格两帧双向监督。两种纠偏均限制在 ±0.1 倍当前帧 10%–90% 分位差，
历史无效时精确返回原始当前预测。GT 仅用于损失和评分，不进入前向或远端请求。

训练保持 KITTI:NYUv2:TUM=3:3:2，先按视频帧对数平方根抽序列，再抽帧对；
30,000 步，有效 batch 8，固定 10% dev，每 2,000 步 dev，best 按 dev SSI AbsRel。
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

冻结组件统一经阻塞式 hybrid 分配：缓存 20%、DGX 28%、本机 52%。缓存提交前缺项
按 DGX:本机=28:52 分配；提交后损坏、身份冲突或 RPC 失败直接报错。所有阶段复用
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
