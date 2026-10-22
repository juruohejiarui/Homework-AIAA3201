# 论文 benchmark 复现记录

**全套 benchmark 尚未完成。** 只根据完整输出凭证报告实测；缺数据、缺权限或协议未确认的项目保留待完成。

## 存储与运行环境

数据完整性迁移已经完成，4,167,446,009 字节逐文件校验通过后删除旧目录。见 [迁移报告](storage/migration_report_zh.md)。共享数据按原始数据集名称位于 `/media/hjr/新加卷/Resouces/Datasets`；`~/Documents/ResourcesLow` 指向该目录，`Proj/data` 是兼容入口目录。项目视频缓存位于 `Proj/outputs/prepared_videos`。

保持 Python 3.13 / PyTorch 2.12；权重、代码和模型结果保留在项目中。VDA-S 与 DA-V2-S 使用同一组 RGB、帧索引和输入宽高比规则；FP32 默认，FP16 独立目录。新入口记录模型、精度和采样清单，源码排版变化不再使运行凭证失效。历史结果不覆盖。

VDA-S 参数量29,035,137，权重116,440,756字节；DA-V2-S 参数量24,785,089，权重99,218,434字节（由已加载模型及权重文件记录）。

## 数据访问状态

| 数据集 | 数据/管线状态 |
|---|---|
| KITTI | 待用户完成官方授权与下载；不代申请、不使用替代数据 |
| ScanNet | 待用户完成官方授权与下载；不代申请、不使用替代数据 |
| Bonn-RGBD | 官方公开压缩包下载/解压中；以下载日志和 verified/extraction receipt 为准 |
| nyu_depth_v2 | labeled 654 图像基准已保留；八场景 raw 视频另行下载/投影，不能用 labeled 代替 |
| MPI-Sintel | 已冻结 23 个序列；结果见下表 |

## 长视频：VDA-S 论文参考及实测

参考来自 [VDA 论文 Table 1](https://arxiv.org/html/2501.12375v1#S4.T1)。Sintel 约 50 帧，其他最多 500 帧。下列差值只供审阅；预处理或序列名单存在差异时，不宣称严格同协议复现。

| 数据集 | 论文 AbsRel / δ1 | 实测 AbsRel / δ1 | 差值 AbsRel / δ1 | 帧数 | 精度 |
|---|---|---|---|---:|---|
| kitti | 0.086 / 0.942 | 待完成 | — | — | FP32 |
| scannet | 0.110 / 0.876 | 待完成 | — | — | FP32 |
| bonn | 0.083 / 0.950 | 待完成 | — | — | FP32 |
| nyuv2 | 0.077 / 0.959 | 待完成 | — | — | FP32 |
| sintel | 0.339 / 0.584 | 0.352119 / 0.591667 | +0.013119 / +0.007667 | 1064 | FP32 |

## 全部已完成运行

| 数据集 | 设置 | 模型 | 精度 | 序列 / 帧 | 实测均值 |
|---|---|---|---|---:|---|
| sintel | long | da2 Small | fp32 | 23 / 1064 | {"abs_rel": 0.4276502138140012, "rmse": 7.177313643411607, "delta1": 0.5390019801167952} |
| sintel | long | vda Small | fp32 | 23 / 1064 | {"abs_rel": 0.3521193130954613, "rmse": 6.896128067559545, "delta1": 0.5916672391557712} |
| sintel | short | da2 Small | fp32 | 23 / 1064 | {"abs_rel": 0.4276502138140012, "rmse": 7.177313643411607, "delta1": 0.5390019801167952} |
| sintel | short | vda Small | fp32 | 23 / 1064 | {"abs_rel": 0.3521193130954613, "rmse": 6.896128067559545, "delta1": 0.5916672391557712} |

ScanNet 原始 TAE 的论文 VDA-S 参考值为 **0.703**；未获得 ScanNet 数据前，实测和差值留空。单位遵循官方 evaluator 的 `×100` 输出。DA-V2-S 是本项目同尺寸对照，不能拿论文中的 DA-V2-L 行作为其复现目标。

## 协议审计与已知差异

- 仓库固定到 `submodule-versions.txt`。必要补丁见 `patches/vda-benchmark-compat.patch` 与 `patches/vda-preprocessing-corrections.patch`，通过 `python -m depth_project.storage.patches` 重放。修复 NYU 函数/变量名、RGB/BGR 保存、TAE JSON 名称及 δ 指标张量设备；不改变评价公式。
- 几何评价直接调用官方 `eval_depthcrafter`：整段 inverse-depth 无约束 scale/shift，预测及对齐后 inverse-depth 下限 1e-3，GT 有效范围 >1e-3 且 <数据集上限，逐帧平均后逐序列等权平均。未使用旧开发流程的非负斜率约束。
- 评价入口拒绝缺失预测，不继承官方 evaluator 静默跳过缺帧的行为；不再逐层重复哈希。
- Sintel 官方 DPT 的单位为米；原公开提取代码的 uint16 截断与 factor=65535/650 不一致。本实现保留 float32 米制 GT（factor=1）。因此标为“官方评价器 + 纠正预处理”，不能将差值完全归因于模型。
- Bonn 使用官方 TUM 时间戳的一对一最近匹配（≤20ms），不再按两个目录的独立排序假定同步。序列按字典序冻结，保留匹配差值和源索引。这可能与原脚本按索引配对不同。
- NYU 必须先用标定和非线性深度转换投影到 RGB 平面；raw PGM 不是米制深度。`depth_project.data.nyuv2_raw` 按官方工具箱移植，输出 factor6000；本机尚未与 MATLAB 逐像素验证，正式同协议结论需保留此限制。
- TAE 直接调用官方双向三维重投影 evaluator，保留原最近像素、碰撞覆盖、hard-crop 与未偏移 K 的约定。裁剪后正确 K 应为 cx−11、cy−8；几何修正版必须另命名，不静默覆盖官方 TAE。
- 论文 TAE 样本数量与当前代码的 20-scene 默认值存在不确定性；本入口固定字典序前20、192帧中的[10:180]。未核实原始名单前，只称“官方代码选择规则复现”。
- 长度曲线支持 110、192、300、400、500：Bonn/ScanNet 固定具备至少500帧的前10序列（所有长度使用同一名单），NYU 固定八场景。短设置 Bonn[30:140]、KITTI前110、ScanNet前270每3帧、Sintel前100；NYU公开脚本只提供500设置，额外短设置不能冒充论文原始协议。
- Sintel 短/长设置使用相同源帧（每序列≤50）；可复用同一完整预测并注明等价性。其他设置重新运行 VDA，不能将长序列结果裁剪后当成短序列推理。
- 各序列 complete.json 保存推理耗时及峰值显存；这不是论文 A100/518×518 的正式延迟复现。正式效率测量另用 FP16，暂停数据预处理并说明 GPU、分辨率、预热与计时范围。
- 旧演示结果称为“光流对齐 MAE”，与三维 TAE 分开。旧非负 SSI、labeled 图像基准和首轮结果仍保留在 first_round 开发结果目录。

## 可恢复执行入口

在项目根目录执行独立的数据、推理和评价入口：

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

Bonn 已完成下载和解压；预处理使用 `depth_project.data.benchmark --dataset bonn`，随后分别运行 long、short 和五种长度。模型推理和评价分别执行；计算脚本不生成报告。已完成序列可恢复，不完整序列重新计算。历史源码快照仅作为已有结果的凭证保留，不是当前入口。

NYU 已改为由用户下载完整 Raw，旧的未完成选择性版本已经删除。完整38包清单及操作见 `download_data_zh.md`。下载完成后仅导出指定八场景供论文评价，不改变测试名单。

授权数据下载到共享盘后，ScanNet prepare 的 source 为官方导出的各 scene color/depth/pose/intrinsic 根目录；KITTI source 为 raw 根目录，另传 `--depth-root` 为带 proj_depth/groundtruth 的 val 根目录。未经授权不自动下载。

## 官方来源

- [VDA 官方代码](https://github.com/DepthAnything/Video-Depth-Anything) 与 [论文](https://arxiv.org/html/2501.12375v1)。
- [Bonn 官方下载](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html)：全集16.4GB。
- [Sintel RGB](https://sintel.is.tue.mpg.de/downloads) 与 [米制深度/相机](https://sintel.is.tue.mpg.de/depth)。
- [NYUv2 可访问的官方页面](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)：raw 时间戳同步及投影工具箱。
- [ScanNet 官方访问说明](https://github.com/ScanNet/ScanNet#scannet-data)。
