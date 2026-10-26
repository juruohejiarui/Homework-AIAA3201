# KITTI＋TUM：即时计算启动

使用`Proj/configs/explore_a/kitti_tum.yaml`，只启用KITTI和TUM，不等待NYUv2。
每组60,000步、有效batch 8，候选池包含全部合格训练帧对。每次抽样即时运行冻结模型，
不保存中间深度、光流或教师输出。结果位于`Proj/outputs/explore_a/kitti_tum/runs/`。

在仓库根目录执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiaa3201
cd /home/hjr/Documents/Projects/Homework-AIAA3201
export XFORMERS_DISABLED=1
cfg=Proj/configs/explore_a/kitti_tum.yaml

bash Proj/explore_a.sh --config "$cfg" status
```

**已有`data.json`时不要再次prepare。**若尚未准备划分，执行一次：

```bash
bash Proj/explore_a.sh --config "$cfg" prepare --extract-kitti
```

先训练Farneback＋规则遮挡＋U-Net：

```bash
bash Proj/explore_a.sh --config "$cfg" train --flow farneback --occlusion rule --fusion unet
```

这组仅需Depth Pro，不加载RAFT或MFTIQ。每2,000步及结束评价dev，
仅dev AbsRel选择best.pt。训练期间默认不评价test（heldout_every=0），无step 0 test。
训练日志为training.jsonl，权重为best.pt、last.pt；旧30,000步协议需使用新runs目录，
不能自动混接。DGX缓存训练使用独立runs_cached_60k，见[DGX_CACHE.md](DGX_CACHE.md)。

训练后比较：

```bash
bash Proj/explore_a.sh --config "$cfg" evaluate --flow farneback --baseline raw
bash Proj/explore_a.sh --config "$cfg" evaluate --flow farneback --baseline rule-fixed
bash Proj/explore_a.sh --config "$cfg" evaluate --flow farneback --occlusion rule --fusion unet
bash Proj/explore_a.sh --config "$cfg" summarize
```

扩展到全部18组：

```bash
# 两个遮挡学生、18组融合器，然后一次共享前端的完整评价
bash Proj/explore_a.sh --config "$cfg" run-matrix
# 或只有训练
bash Proj/explore_a.sh --config "$cfg" run-matrix --train-only
# 全部所需best.pt就绪后，单独评价23个分支（18组＋5个基线）
bash Proj/explore_a.sh --config "$cfg" evaluate-matrix
bash Proj/explore_a.sh --config "$cfg" summarize
```

共享评价仅在当前样本上共用冻结输出，各分支时序状态独立，处理完释放。
独立evaluate报告在线预测耗时；共享评价的整体吞吐、前端和融合器耗时分别报告，
其独立端到端延迟为空，不将共享成本误称为单模型延迟。

CPU原始帧预读由`online.read_workers: 4`、`online.prefetch: 8`控制。
评价结构化日志单写入者保序，图片由`result_io.write_workers: 2`后台保存，
写入队列默认8项；写入全部成功后发布metrics.json。
不保存稠密预测文件，仍保留逐帧指标、逐序列统计和每序列少量可视化。
正式训练及评价由你启动。

## 进度、曲线和TensorBoard

新版默认显示训练优化步及dev/test帧对进度，当前默认不执行训练中的test。
终端使用进度条，重定向日志时每30秒输出进度、速度与ETA；阶段开始、结束和错误另行显示。
训练条附loss、学习率、当前来源/序列、dev/test摘要及CUDA显存。
最终evaluate和evaluate-matrix也有进度；矩阵总量仍是帧对数，不乘23。

各实验目录中的`plots/`默认每1,000步及结束更新；有相应日志才生成对应图表：

- `training.png`：loss、学习率、优化更新吞吐与峰值显存。
- `development.png`：dev总体/各来源分数及历史最佳值。
- `test_depth_macro.png`、`test_depth_<数据源>.png`：七项深度指标和原始Depth Pro基线。
- `test_delta_macro.png`：相对基线变化，误差下降/精度上升表示改善。
- 学生实验使用`distillation.png`记录MFTIQ软标签BCE，不标作真实遮挡准确率。

TensorBoard 2.21.0已安装在aiaa3201中，每个实验的事件在`tensorboard/`。
由你另开终端手动启动服务，不需要停止训练：

```bash
conda activate aiaa3201
cd /home/hjr/Documents/Projects/Homework-AIAA3201
tensorboard --logdir Proj/outputs/explore_a/kitti_tum/runs --host 127.0.0.1 --port 6006
```

浏览器打开`http://127.0.0.1:6006`。在Scalars中查看train、dev、test、test_delta、
baseline、test_counts及time标签；按不同实验比较。TensorBoard事件异步刷新，默认10秒。

已有日志可以单独补画，不读取原始数据、不加载模型、不重新推理：

```bash
cfg=Proj/configs/explore_a/kitti_tum.yaml
bash Proj/explore_a.sh --config "$cfg" plot --run farneback-rule-unet
# 可选：同时重建TensorBoard；该实验的训练进程必须已停止
bash Proj/explore_a.sh --config "$cfg" plot --run farneback-rule-unet --tensorboard
# 不指定--run则处理此配置下所有有标量记录的实验
bash Proj/explore_a.sh --config "$cfg" plot
```

未完成第一次test、也没有训练标量时，没有可画的数据，不生成伪造的零值曲线。
已运行的旧进程不会自动加载新代码，需要你停止后重跑原命令；last.pt恢复的步数不变。
默认只有dev曲线；最终test保存在最终评价目录，不补造训练test曲线。
恢复时保留原始JSONL，通过`observability/sessions.jsonl`记录会话边界和checkpoint步数，
图表与TensorBoard只展示有效训练分支，排除回退后失效的未来步数。
损坏的最后一行先保留为`.partial-*`文件再修复；不删有效历史记录。

可独立控制显示与导出，不改变训练/模型选择身份：

```yaml
observability:
  progress: auto          # auto/bar/text/off
  tensorboard: true
  plots: true
  plot_every: 1000
  refresh_seconds: 1.0
  non_tty_every_s: 30.0
  tb_flush_s: 10
```

不增加模型前向次数，不改变随机状态、dev/test频率或最佳checkpoint选择。
PNG覆盖更新，TensorBoard只记录标量，不额外保存稠密图像或中间预测。

### 报告重画图的数据

原始数值保存在JSONL中，PNG是可重建的派生文件。每次刷新图表和退出训练时，另在
实验目录导出`scalars.csv`长表，包含`step, phase, scope, dataset, sequence, camera,
metric, value, tag`，覆盖实际已记录的训练、dev、test、基线、变化值、计数及test逐序列指标。
CSV不使用TensorBoard的float32事件值回填，不额外舍入原始日志数值。
默认训练每50步一条，dev每2,000步一条，默认训练中不记录test；没有记录的训练步不会补造数据。
最终评价逐样本指标仍在各`evaluation-test/frames.jsonl`，完整汇总在`metrics.json`，
不把所有逐帧数据重复塞进训练曲线CSV。`plot`会重新生成CSV及PNG，不需要重新训练。

```python
import pandas as pd
from pathlib import Path
run = Path("Proj/outputs/explore_a/kitti_tum/runs/farneback-rule-unet")
table = pd.read_csv(run / "scalars.csv")
curve = table[table["tag"] == "test/macro/abs_rel"]
curve.plot(x="step", y="value")
# 也可直接读取原始日志，自定义聚合或图像样式
train = pd.read_json(run / "training.jsonl", lines=True)
```
