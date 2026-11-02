# NYUv2 H5 与共享预测缓存

训练直接读取 `~/Documents/Resources/Datasets/nyu_depth_v2/h5`。
H5 含 RGB 和 FP32 米制深度；已有对齐 GT 不再执行 Raw 传感器投影。
训练帧对限于同一场景内相邻保留编号，不宣称原始连续帧或固定 0.1 秒。
官方 validation 没有可靠时序／场景映射，不构造视频帧对。

如需解压已有训练归档，可续跑以下命令；保留 tar，不覆盖冲突文件：

```bash
export PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"
python -m explore_a.data.nyu_h5 extract \
  --archives "$HOME/Documents/Resources/Datasets/nyu_depth_v2/archives/data" \
  --output "$HOME/Documents/Resources/Datasets/nyu_depth_v2/h5"
```

固定训练清单已经存在，不为配置清理重新运行 prepare 或重划 split。

共享缓存的 DA 预测位于 `glob/frontends/da2/frames`，原生输出位于该前端的
`native`。`glob/flow` 与 `glob/teacher` 保存共享光流和 MFTIQ，三个模型复用。
GT、尺度对齐参数和每帧特征归一化不存入预测缓存。
本机直接读取 H5；DGX 在线推理通过内存传 RGB／预测，不传 GT。
独立缓存生产器沿用临时无损 RGB 导出及确认后清理流程。

NPZ 为 FP32、DEFLATE 9，先在内存压缩，再为临时文件预分配准确压缩长度并原子提交。
不支持原生 fallocate 时跳过预分配，不补零模拟。消费者只读预测，不写回 online 结果。
超过 4,000 RGB 的视频只跳过缓存，仍保留训练样本。

完整生产、状态、停止、恢复与 DGX 命令统一见 [DA 操作指南](DA_TRAINING.md)。
