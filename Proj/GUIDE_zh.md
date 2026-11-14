> 方向 A 的最新三模型训练、hybrid 和四视频评价见
> [DA 操作指南](src/explore_a/DA_TRAINING.md)。以下为独立基础工具。

# 独立模块运行指南

## 正式课程实验入口

课程使用 `depth_project.course` 完整执行 Part 1 / Part 2。以下在仓库根目录、
现有 `aiaa3201` 环境运行：

```bash
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
python -m depth_project.course run
python -m depth_project.course verify
python -m depth_project.course report
```

`run` 执行所有正式数据、预测保存、效率测量和交付生成；`verify` 从保存数组重新评分，
检查654张覆盖、30组立体输出、视频对应关系和ZIP完整性；`report` 重新生成可视化和成果包。
默认输出 `Proj/outputs/course_part1_part2/`，可通过 `--output` 指定新的目录。
正式 NYU 使用 UniDepthV2 原始米制结果，禁止 GT 尺度对齐；本指南下方的 SSI 工具仅用于额外研究。

实际数据位置：`Proj/data/middlebury/MiddEval3/trainingQ`（15个带GT场景），
`Proj/data/nyuv2/nyu_depth_v2_labeled.mat`、`splits.mat` 和 `test/`。
模型使用 `Proj/checkpoints/depth_anything_v2_vits.pth`、`video_depth_anything_vits.pth`
和 `unidepth-v2-vits14/`。官方权重来源：
[DA-V2-S](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)、
[VDA-S](https://huggingface.co/depth-anything/Video-Depth-Anything-Small)、
[UniDepthV2 ViT-S](https://huggingface.co/lpiccinelli/unidepth-v2-vits14)。
现有数据或权重缺失时报错，不擅自替换数据集或模型。

以下命令在仓库根目录执行，环境使用现有 `aiaa3201`。完整下载步骤见 [下载交接](download_data_zh.md)。本指南提供运行方式，不表示所有课程实验已经完成。

## 1. 数据准备

标准 NYUv2 labeled 测试集已经准备完成。需要重新导出时，单独执行：

```bash
conda run -n aiaa3201 python -m depth_project.data.nyuv2 \
  --mat Proj/data/nyuv2/nyu_depth_v2_labeled.mat \
  --split Proj/data/nyuv2/splits.mat \
  --output Proj/data/nyuv2/test
```

课程必需视频是指定的 davis_rollercoaster.mp4；自录视频属于可选研究。
以下抽帧命令仅供额外视频使用，不是课程指定视频的正式准备命令：

```bash
conda run -n aiaa3201 python -m depth_project.data.video \
  --video /absolute/path/static.mp4 \
  --output Proj/outputs/prepared_videos/static.npz \
  --fps 10 --max-resolution 640 --max-frames 300
```

NPZ 保存 RGB、实际采样 FPS 和原始帧索引。所有模型使用同一份 NPZ，不自行改变采样。

## 2. 立体匹配

```bash
conda run -n aiaa3201 python -m depth_project.stereo \
  --data Proj/data/middlebury --method sgbm --output Proj/outputs/stereo/sgbm
```

改成 `--method bm` 运行另一基线。结果包含原始视差、右视差、左右一致性掩码、米制深度、Bad-2、有效预测 EPE、coverage 和受控视差扰动结果。Middlebury 的 baseline 从毫米转米，深度使用 `f × baseline / (disparity + doffs)`；PFM 头的幅值不乘入像素视差。

可视化独立执行，不重算匹配：

```bash
conda run -n aiaa3201 python -m depth_project.visualization.stereo \
  --arrays Proj/outputs/stereo/sgbm/Adirondack/stereo.npz \
  --output Proj/outputs/stereo/sgbm/Adirondack/comparison.png
```

## 3. 图像与视频推理

```bash
conda run -n aiaa3201 python -m depth_project.inference \
  --model da2 --images Proj/data/nyuv2/test/rgb \
  --output Proj/outputs/da2_nyu --precision fp32

conda run -n aiaa3201 python -m depth_project.inference \
  --model vda --frames Proj/data/videos/official_demo_frames.npz \
  --output Proj/outputs/vda_demo --precision fp32
```

图像保存逐帧 NPY；视频保存 `depths.npz`。DA-V2 和相对 VDA 的输出为 relative inverse depth。默认使用 `checkpoints` 中现有 Small 权重；`--weights` 可指定其他文件。VDA metric 需要对应 metric 权重及 `--metric`，不能把相对权重当作米制模型。

`--backend upstream` 为默认路径，也可选择 `sdpa` 或 `flash`；FP32 默认，FP16 使用 `--precision fp16`。`profile.json` 是本次运行的数据记录，不是经过预热的论文效率复现。

## 4. 图像评价与光流诊断

```bash
conda run -n aiaa3201 python -m depth_project.evaluation.images \
  --predictions Proj/outputs/da2_nyu --ground-truth Proj/data/nyuv2/test/gt \
  --alignment inverse-ssi --output Proj/outputs/da2_nyu/metrics.json
```

DA-V2 可用 `--alignment inverse-ssi` 或 `inverse-median` 作为 oracle 对照。开发 SSI 使用非负斜率约束，不能冒充论文整段无约束 inverse-depth SSI。图像评价保存逐图精度和自定义边界指标。

```bash
conda run -n aiaa3201 python -m depth_project.evaluation.flow \
  --frames Proj/data/videos/official_demo_frames.npz \
  --predictions Proj/outputs/vda_demo/depths.npz \
  --output Proj/outputs/vda_demo/flow.json
```

光流诊断采用双向 Farneback 和前后向一致性掩码，相对预测使用整段固定 P1/P99 gauge。输出是光流对齐 MAE 和表观尺度变化代理，既不是三维 TAE，也不能剥离移动相机导致的真实 Z 变化。

## 5. 官方视频 benchmark

数据预处理、推理和评价分别执行。以已下载 Sintel 为例：

```bash
conda run -n aiaa3201 python -m depth_project.data.benchmark --dataset sintel

conda run -n aiaa3201 python -m depth_project.inference.benchmark \
  --manifest Proj/data/sintel/prepared/vda_protocol_v2/manifest.json \
  --model vda --setting long --output Proj/outputs/benchmarks/sintel/vda-long

conda run -n aiaa3201 python -m depth_project.evaluation.video \
  --manifest Proj/data/sintel/prepared/vda_protocol_v2/manifest.json \
  --predictions Proj/outputs/benchmarks/sintel/vda-long --setting long \
  --output Proj/outputs/benchmarks/sintel/vda-long/metrics.json
```

`--dataset bonn` 使用 TUM 时间戳一对一匹配，容差20ms；ScanNet/NYU 接收已对齐导出目录，传 `--source`；KITTI 另需 `--depth-root` 指向 annotated depth 的 val 目录。预处理结果只保存可复用数据和明确帧对应关系，不反复给每个输入文件计算哈希。

设置包括 `long`、`short`、`length-110/192/300/400/500`。长度实验在 Bonn/ScanNet 中固定同一批具备500帧的前10个序列，NYU 使用指定八场景。不同长度必须重新推理 VDA，不能简单裁剪长视频预测。未确认论文原始名单前，只称官方代码选择规则复现。

ScanNet 原始 TAE 单独使用 `depth_project.evaluation.tae`，参数为 `--manifest`、`--predictions`、`--output`，对应推理设置为 `--setting tae`。它保留官方重投影及输出 `×100` 的约定；与光流 MAE 分开保存。

## 6. 可视化与报告

```bash
conda run -n aiaa3201 python -m depth_project.visualization.video \
  --frames Proj/data/videos/official_demo_frames.npz \
  --predictions Proj/outputs/vda_demo/depths.npz --output Proj/outputs/vda_demo/demo.mp4

conda run -n aiaa3201 python -m depth_project.visualization.curves \
  --metrics Proj/outputs/vda_demo/flow.json --output Proj/outputs/vda_demo/curves.png
```

正式 Part 1 / Part 2 的结果说明、图表和成果包由 `depth_project.course report` 生成。
方向 A 已有实现和扩展评价；课程应用及最终 CVPR 报告单独完成。自录视频不属于必交要求。
