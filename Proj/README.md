# Depth Project

项目代码位于 `src/depth_project`，采用标准 Python 包布局。没有统一大脚本，也不保留旧入口。

## 使用现有环境

从仓库根目录执行一次本地注册，不重新安装依赖：

```bash
conda run -n aiaa3201 python -m pip install --no-deps --no-build-isolation -e Proj
```

本机 `aiaa3201` 的解释器为 `/home/hjr/miniconda3/envs/aiaa3201/bin/python`，已安装 PyTorch 2.12.0+cu130。后续命令使用 `conda run -n aiaa3201`，避免落入 base 或编辑器的其他解释器。此包不依赖 uv，也不创建 `.venv`。

## 模块职责

| 模块 | 用途 |
|---|---|
| `data` | labeled/raw NYUv2、视频帧、各 benchmark 的数据读取和同步 |
| `stereo` | BM/SGBM、标定几何、左右一致性、视差敏感性 |
| `models` | DA-V2、Depth Pro、VDA 各自的适配器 |
| `inference` | 独立推理、预测保存、GPU 串行调度 |
| `evaluation` | 图像精度、官方视频指标、三维 TAE、光流诊断 |
| `visualization` | 独立生成图片、曲线和视频 |
| `storage` | 下载、解压、一次性复制迁移、submodule 补丁 |

计算命令保存 NPY/NPZ、JSON 和必要日志；不会创建或覆盖 Markdown 报告。报告直接编辑。

## 文档与结果

- [运行指南](GUIDE_zh.md)：各独立入口及输入输出。
- [下载交接](download_data_zh.md)：完整 NYUv2 Raw、KITTI、ScanNet 及已下载数据。
- [论文复现记录](paper_reproduction_zh.md)：实测值、论文参考值与协议差异。
- [Part 3 指南](part3_guidance_zh.md)：后续改进、消融和应用评价。
- [首轮历史记录](FIRST_RUN_zh.md)：重写前的完整实验结果，仍保留在 `outputs/first_round`。

共享数据位于 `~/Documents/ResourcesLow/Datasets`，实际目录为 `/media/hjr/新加卷/Resouces/Datasets`；按原始数据集命名。权重位于 `checkpoints`，本项目结果位于 `outputs`。已有数据、模型和结果不因代码重写而删除。

## 官方依赖

三个 submodule 的固定提交见 `submodule-versions.txt`。必要补丁位于 `patches`，单独执行：

```bash
conda run -n aiaa3201 python -m depth_project.storage.patches
```

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [Depth Pro](https://github.com/apple-aiml-research/ml-depth-pro)
