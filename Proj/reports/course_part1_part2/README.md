# Part 1 / Part 2 正式结果

本包包含全部正式预测、逐样本指标、可视化、效率记录与复现信息。

官方原链接重新下载后实际解码 70 帧；使用前 70 帧、24 fps，无跳帧、插帧或补帧。PDF写前100帧；本次按用户确认规则保留实际可用帧数。

## NYU：UniDepthV2 ViT-S/14

RGB-only FP32，全部654张官方测试图，GT读取filled depths；480×640，crop[45:471,41:601]，有限GT 0.001<z<10m，预测截断[0.001,10]m，无尺度对齐，逐图等权。原始预测未经截断保存。

| AbsRel | RMSE (m) | δ1 |
| --- | --- | --- |
| 0.12942544 | 0.42372887 | 0.87565331 |

## Middlebury：15个quarter-resolution训练场景

GT域为finite & disparity>0，预测finite & disparity>=0；MAE只在有效预测交集上计算，coverage以全部有效GT为分母。主汇总按像素加权，另存场景等权。左右一致性不缩小主评价域。

| 方法 | MAE (px) | 有效预测比例 |
| --- | --- | --- |
| BM | 1.811867 | 0.599567 |
| SGBM | 1.955948 | 0.759570 |

深度公式：f × baseline_mm / (1000 × (disparity_px + doffs))。OpenCV视差先除16。

## 指定视频

DA-V2-Small逐帧、VDA-Small官方离线推理，均为FP16 autocast；video/保存三份H.264原速演示及全部浮点预测。相同INFERNO色图，每方法整段固定P1/P99，亮色表示近处，relative depth颜色不代表共享米制尺度。实际上下限见video/normalization.json。

## 效率

计时包含预处理、推理、输出恢复与CPU传输，不含模型加载／磁盘写入；每模型独立进程、本机GPU、无缓存或DGX。均为3轮，UniDepth使用32张官方开发图而非测试图，单图模型预热5帧，VDA预热32帧窗口。实际网络输入形状及版本见efficiency/*.json。

| 模型 | ms/frame均值 | 标准差 | 峰值allocated MiB |
| --- | --- | --- | --- |
| unidepthv2 | 30.351 | 0.004 | 583.5 |
| da2 | 19.158 | 0.038 | 339.3 |
| vda | 26.178 | 0.105 | 2632.3 |

立体CPU计时见stereo/per_scene.csv，包含左／右两次匹配；不与GPU模型耗时混排。

## 文件与复现

- stereo/：两种方法各15场景、数组、无效区及对照图。
- nyu/：654张原始FP32米制预测、逐图表、总指标与代表图。
- video/：同帧RGB、两种预测、三份原速比较视频及连续帧图。
- failures/：三个课程素材案例；详见analysis.md。
- efficiency/：独立测量的每轮记录与汇总表。
- 每阶段identity.json：代码、输入、权重及设置身份。

仓库根目录执行：

```bash
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
python -m depth_project.course run
python -m depth_project.course verify
python -m depth_project.course report
```

使用现有aiaa3201环境。先准备官方Middlebury训练场景、NYU labeled MAT与splits，以及三个官方模型权重；命令不下载原始数据或权重。官方视频缺失时自动按PDF链接下载。身份改变时使用新的--output目录。

## 范围与来源

本包只交付Part 1／2，不包含Part 3、简单应用或最终CVPR论文，不作为完整课程提交包。无GT的视频只支持定性比较；不从色图声称米制精度。

公开代码：https://github.com/juruohejiarui/Homework-AIAA3201

模型与数据来源：Middlebury Stereo v3；NYU Depth V2；Depth Anything V2（NeurIPS 2024）；Video Depth Anything（CVPR 2025）；UniDepthV2（arXiv:2502.20110）。模型使用项目third_party中的官方代码和本地官方权重，版本及权重哈希见各identity.json。

## 完整结果与分析入口

[课程难例分析与局部图](analysis.md)。完整原速视频和全部浮点预测在本机 `Proj/outputs/course_part1_part2/`；完整成果包在本机 `Proj/outputs/part1_part2_results.zip`。大型输出不加入Git。

## 已公开的演示视频与完整结果

- [RGB / DA-V2 / VDA 三面板视频](video/rgb_da2_vda.mp4)
- [RGB / DA-V2 视频](video/rgb_da2.mp4)
- [RGB / VDA 视频](video/rgb_vda.mp4)
- [全部原始预测与完整成果包（Release）](https://github.com/juruohejiarui/Homework-AIAA3201/releases/tag/course-part1-part2-20261114)

逐场景立体结果见 [per_scene.csv](stereo/per_scene.csv)，NYU 全部654张逐图结果见 [per_image.csv](nyu/per_image.csv)。
