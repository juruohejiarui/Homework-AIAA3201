# 数据下载交接

2026-10-23 更新。数据使用共享路径 `~/Documents/ResourcesLow/Datasets`，实际挂载位置由链接解析，不再写死历史挂载点。

## 1. 当前安排

| 数据 | 当前状态 | 后续操作 |
|---|---|---|
| NYUv2 完整 Raw | 完整清单38个 ZIP，共421.54GB；此次未启动下载 | 使用下方新入口，由你运行 |
| NYUv2 labeled | 完整1449对 MAT 已有，含795训练/654测试划分 | 直接复用，无需重下 |
| NYUv2 预处理训练镜像 | 已冻结公开 HF 的14个 TAR，35.15GB | 需要时用 prepared 模式，与完整 Raw 分开保存 |
| KITTI | 已完成论文相关验证实验，已有 Raw 和深度包 | 不再按历史缺失清单重复下载；训练标注从已有包展开 |
| ScanNet v2 | 仍需官方授权 | 取得授权后下载冻结名单的 .sens |
| Bonn / Sintel / Middlebury / 权重 | 已有 | 无需重新下载 |

## 2. NYUv2：新的下载入口

完整 Raw + labeled，默认复用已有文件，未完成包续传：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_nyuv2.sh
```

HF 预处理训练子集 + labeled（这是35.15GB的抽样训练数据，不是完整 Raw）：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_nyuv2.sh --mode prepared
```

只查看默认完整清单，不下载：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_nyuv2.sh --plan
```

新下载入口使用现有 conda aiaa3201 与系统 aria2，Raw 下载完成时核对官方 MD5，HF 分片核对冻结的 SHA256。完整 Raw 尚未发现可确认等价的 HF/ModelScope 镜像，预处理训练数据优先走 HF。

解压使用系统 unzip / tar。目录布局、各模式、镜像核查、解压指令及训练划分见 [NYUv2 完整指南](nyuv2_download_zh.md)。**不要继续使用旧 nyuv2_raw_aria2.txt**：它包含历史固定挂载路径。

现有 `depth_project.data.nyuv2_raw` 只准备 VDA 的八个评价场景，不是全部 Raw 训练预处理器；完整下载后仍需建立训练场景同步、投影和 split。

## 3. KITTI 与其他训练数据

现有 `KITTI/archives/data_depth_annotated.zip` 已包含训练标注，无需重新下载。使用系统 unzip 展开：

```bash
dataset="$HOME/Documents/ResourcesLow/Datasets/KITTI"
unzip -q -n "$dataset/archives/data_depth_annotated.zip" 'train/*' -d "$dataset/depth"
```

训练 RGB 使用现有 Raw 对应 drive/camera/frame。当前论文 val drive 继续保留评价，不混进训练。旧 downloader/repair 清单是历史下载记录，不表示现在仍然缺少这些包。

原有补包解压工具仍保留为需要恢复时的入口：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/extract_kitti_recovery.sh
```

多数据集训练不限于 NYUv2。已有 KITTI、NYUv2，加上 Virtual KITTI 2、DIODE 及带位姿 RGB-D 数据的用途、体积、划分与资源压力见 [多数据集说明](multi_dataset_training_zh.md)。这里不自动启动额外下载。

## 4. ScanNet v2

1. 按[官方访问说明](https://github.com/ScanNet/ScanNet#scannet-data)签署协议并取得官方下载工具。
2. 保存数据到 `~/Documents/ResourcesLow/Datasets/ScanNet/`。本项目需要 `.sens`，不需要语义网格和2D语义标签。
3. 先冻结扫描名单，再使用授权脚本逐项下载。原论文的确切名单尚未得到确认，不能把任意第三方片段集合称为论文原始 split。

取得 `download-scannet.py` 后，单个扫描的官方下载形式为：

```bash
python download-scannet.py -o ~/Documents/ResourcesLow/Datasets/ScanNet/raw \
  --id scene0000_00 --type .sens
```

此扫描 ID 仅展示官方命令用法，不代表最终评价名单。授权脚本如要求专用 Python 版本，可单独运行，不为它降级现有模型环境。

使用官方 SensReader 导出后，整理为以下布局：

```text
ScanNet/exported/<scene>/
  color/<frame>.jpg
  depth/<frame>.png
  pose/<frame>.txt
  intrinsic/intrinsic_depth.txt
```

RGB/深度按同一帧 ID 对应，位姿为 camera-to-world。保留深度内参，不使用裁剪后图像配上未说明的其他内参。准备命令：

```bash
conda run -n aiaa3201 python -m depth_project.data.benchmark \
  --dataset scannet --source ~/Documents/ResourcesLow/Datasets/ScanNet/exported
```

## 5. 已完成的数据与镜像

Bonn 原包为 `Bonn-RGBD/archives/rgbd_bonn_dataset.zip`，解压内容位于 `Bonn-RGBD/raw/bonn`。其 SHA256 为 `d2afdc286ecdb02a3e57cb98e3908185b1133a3f79f181970064eb177221a84b`，与 [geyongtao/bonn](https://huggingface.co/datasets/geyongtao/bonn) 中同名原包一致。新的 Bonn 下载入口默认使用固定提交的 HF 镜像，并复用已有完整包。

Sintel 已包含训练 RGB 和2015-03-05版本的米制深度，位于 `MPI-Sintel/archives` 与 `MPI-Sintel/raw`。不要另下光流测试包代替深度数据。
