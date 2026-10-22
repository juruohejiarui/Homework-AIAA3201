# 首轮运行结果

这是一轮实际运行的 Part 1/2 基准与端到端调试，不是最终课程提交。环境与数据均已安装/下载，不再只是参考脚本。

## 环境

Python 3.13.16；PyTorch 2.12.0+cu130；torchvision 0.27.0；NumPy 2.5.4；Pandas 3.0.6；SciPy 1.18.1；xFormers 0.0.35；本地 FlashAttention 2.8.3+cu130torch2.12。三种注意力 CUDA 前向/反向验证通过，pip check 无冲突。精确版本见 requirements-lock.txt。

数据已迁往 ~/Documents/ResourcesLow 下的原始数据集名称目录；Proj/data 为各数据集软链接入口目录，HF_DATASETS_CACHE 指向共享 .cache/huggingface/datasets。三个官方 submodule 与版本记录已保存。Depth Pro 有 NumPy 声明兼容补丁，见 patches/。

## NYUv2：标准 654 张

参考协议：depths GT、480x640、0.001<GT<10m、crop [45:471,41:601]、预测截断、逐图计算后平均。课程链接缺少完整 NYU evaluator，尚不能称为严格官方分数。

| Method / alignment | AbsRel | SqRel | RMSE m | RMSE-log | δ1 | δ2 | δ3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA-V2 Small / inverse-SSI (oracle, a>=0) | 0.0617 | 0.0379 | 0.2974 | 0.0935 | 0.9554 | 0.9878 | 0.9958 |
| DA-V2 Small / inverse-median (oracle) | 0.3481 | 0.8852 | 1.7122 | 0.3852 | 0.4817 | 0.7525 | 0.8987 |
| Depth Pro / metric / no alignment | 0.1005 | 0.0640 | 0.3922 | 0.1353 | 0.9152 | 0.9799 | 0.9919 |

SSI 是非负斜率约束的 GT oracle 拟合；1 张图（0650）无约束斜率为负，采用约束边界常数解并保留评价，未丢弃失败样本。Median 与 SSI 显著不同，不能混为同一种精度。Depth Pro 没有测试尺度对齐。

边界为本项目自定义 log-depth Sobel P/R/F1（阈值0.05、容差1px），不是论文专用边界分数；详情见 JSON/CSV。

## Middlebury：15 场景

| Method | Bad-2 all GT (invalid计错) | 有效预测EPE px | LR coverage |
|---|---:|---:|---:|
| bm | 0.4681 | 1.9312 | 0.5070 |
| sgbm | 0.3357 | 2.1188 | 0.6217 |

以上是 Q 分辨率、各场景等权平均的参考评价，不是全分辨率排行榜分数。已保存原始左右视差、depth、LR mask 和每距离箱扰动实验。

## 官方演示视频

输入来自 VDA 自带 davis_rollercoaster.mp4，统一采样后共35帧、640x360、实际12 FPS；不是自录视频。两方法使用同一输入和光流有效像素，平均有效比例约80.43%。

- da2_demo：光流对齐 MAE=0.014932（各方法整段固定P1/P99 gauge单位）。
- vda_demo：光流对齐 MAE=0.013184（各方法整段固定P1/P99 gauge单位）。

此单段演示中 VDA 光流对齐 MAE 下降 11.70%；不能外推为普遍收益，也不能把移动相机的 median-depth 变化全部称为漂移。固定色标视频、逐帧曲线和原始浮点图均已保存。

## 性能记录

| Run | FPS | 秒/输出帧 | 峰值 allocated MiB |
|---|---:|---:|---:|
| da2_nyu | 60.84 | 0.0164 | 277.7 |
| depthpro_nyu | 5.39 | 0.1857 | 3790.0 |
| da2_demo | 65.23 | 0.0153 | 339.3 |
| vda_demo | 37.34 | 0.0268 | 2632.3 |

模型 GPU 推理串行执行。部分 CPU 评估/数据处理与推理并行，这一轮性能属于开发记录，正式效率表应在空闲机器独立重复计时。VDA 秒/帧是离线整段摊销值，不是在线单帧响应延迟。

## 输出与剩余工作

- outputs/first_round/summary.json、nyu_metrics.csv：主结果。
- outputs/first_round/nyu_comparison.png、temporal_curves.png：图像与时间曲线。
- outputs/first_round/compare_official_demo.mp4：RGB/DA-V2/VDA 对比。
- outputs/environment_check.json：开发环境 GPU 算子验证。
- part3_guidance_zh.md：Part 3 推荐方案、消融、评价和应用设计。

仍需按课程要求录制 static/moving/hard 片段、确认 NYU 官方协议、完成 Part 3 改进和应用评价、报告/arXiv/最终ZIP。这些未用官方演示视频冒充。
