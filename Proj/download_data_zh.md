# 数据下载交接

## 1. 需要你下载的内容

| 数据 | 当前状态 | 需要的内容 |
|---|---|---|
| NYUv2 Raw | 旧的选择性下载已删除；完整下载未启动 | 38个官方 ZIP，421,539,441,278 字节，约421.54GB；均附官方 MD5 |
| KITTI | ModelScope 深度包已整理，验证集已展开；Raw 下载正在进行 | 13个验证序列的 synced+rectified Raw RGB；缺失清单见 `storage/kitti_val_raw_urls.txt` |
| ScanNet v2 | 待授权与下载 | 冻结扫描名单对应的 `.sens`，包含颜色、深度、内参和位姿 |
| Bonn | 已完整下载并解压 | 无需重下；16.4GB 原包与 HF 镜像 SHA256 完全一致 |
| Sintel | RGB、深度均已下载 | 无需重下 |
| NYUv2 labeled、Middlebury、模型权重 | 已保留 | 无需重下 |

当前 `~/Documents/ResourcesLow` 指向 `/media/hjr/新加卷/Resouces`，共享数据根目录为 `~/Documents/ResourcesLow/Datasets`。保留 `Resouces` 拼写。项目入口与默认路径已按此布局调整；下载前确认新加卷已挂载即可。

## 2. 完整 NYUv2 Raw

已从[官方页面](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)列出的分包及官方服务器的 classrooms 包生成完整清单，并取得38份 MD5。官方页面的约428GB与当前实际分包字节数有差别，以冻结清单为准。尚未找到可验证的完整 Raw HF/ModelScope 镜像；常见 NYUv2 镜像是 labeled 或抽样图像集，不能替代连续视频。

安装 aria2（如果尚未安装），从仓库根目录下载：

```bash
sudo apt install aria2
cd /home/hjr/Documents/Projects/Homework-AIAA3201

aria2c -c -j 3 -x 4 -s 4 \
  --file-allocation=none \
  --auto-file-renaming=false \
  -i Proj/storage/nyuv2_raw_aria2.txt
```

目标目录已写入清单，不需要另传 `-d`：

```text
/media/hjr/新加卷/Resouces/Datasets/nyu_depth_v2/archives/raw/
```

清单每项都包含官方 MD5，由 aria2 在下载完成时核对。中断后重复同一命令续传，保留 `.aria2` 文件；不要重新运行清单探测，也不要删除已完成 ZIP。

若不安装 aria2，可用现有 curl 顺序续传，并在每个包完成后核对一次 MD5：

```bash
conda run --no-capture-output -n aiaa3201 python -m depth_project.storage.download nyuv2-raw
```

完成下载后再解压，ZIP 和原始场景分别保留：

```bash
conda run --no-capture-output -n aiaa3201 python -m depth_project.storage.extract \
  --archives /media/hjr/新加卷/Resouces/Datasets/nyu_depth_v2/archives/raw/*.zip \
  --output /media/hjr/新加卷/Resouces/Datasets/nyu_depth_v2/raw/scenes
```

解压过程中 ZIP 库核对 CRC；不用在解压前后额外重复全量扫描。预计需同时容纳约422GB压缩包及展开文件；本机先前确认约2.7TB可用。

论文评价仍只导出指定八场景，不改变测试名单：

```bash
conda run -n aiaa3201 python -m depth_project.data.nyuv2_raw \
  --source /media/hjr/新加卷/Resouces/Datasets/nyu_depth_v2/raw/scenes \
  --output /media/hjr/新加卷/Resouces/Datasets/nyu_depth_v2/exported_toolbox_python_v1
```

这一步包含字节转换、时间戳同步和标定投影。Python 投影移植尚未完成与 MATLAB 的逐像素对照，报告会保留该限制。Raw PGM 不可以直接按毫米或米使用。

## 3. KITTI

1. 已下载 [ModelScope 镜像](https://modelscope.cn/datasets/OmniData/KITTI_depth_completion)，完整包位于 `~/Documents/ResourcesLow/Datasets/KITTI/depth_completion/raw/KITTI_depth_completion.tar.gz`，无需重新下载深度标注。
2. 已提取 `KITTI/archives/data_depth_annotated.zip`，并展开完整 `val/` 到 `KITTI/depth/val/`：13个序列，左相机3426张 GT。训练集保留在 ZIP 中，需要训练时再展开。用户已确认外层包 SHA256 正确，不重复校验。`depth_completion/sample/image` 中的抽样图片不能替代连续 Raw RGB。
3. 在[Raw 页面](https://www.cvlibs.net/datasets/kitti/raw_data.php)下载这些 val drive 对应的 **synced+rectified** 包和标定。解压后应为 `KITTI/raw/<date>/<drive>/image_02/data/*.png`，帧名与 GT 一致。无需下载无关 Raw drives。

当前已有 Raw `2011_09_26_drive_0001_sync`，但它不属于上述验证集；`drive_0002` 的 ZIP 正在由用户启动的全量 `downloader.sh` 下载。此次未中断该下载器，也未移动它正在写入的文件。

核对时，验证集3426张左相机 GT 尚无匹配的已展开 RGB。缺失清单包含13个 Raw drive ZIP，以及另外4个日期的标定包；`2011_09_26` 标定已存在。详细帧数见 `outputs/kitti_data_inventory.json`。

若只复现当前验证集，可在原下载终端按 Ctrl+C 停止全量下载器，再从仓库根目录改用以下清单。不要与原下载器同时写同一个文件。目标仍为现有 `KITTI/raw`，以便续传当前 `drive_0002` ZIP；保留下载包。

```bash
cd /home/hjr/Documents/Projects/Homework-AIAA3201
aria2c -c -j 2 -x 4 -s 4 \
  --file-allocation=none \
  --auto-file-renaming=false \
  -d ~/Documents/ResourcesLow/Datasets/KITTI/raw \
  -i Proj/storage/kitti_val_raw_urls.txt
```

下载全部完成后解压，不删除 ZIP：

```bash
for archive in ~/Documents/ResourcesLow/Datasets/KITTI/raw/*.zip; do
  unzip -q -n "$archive" -d ~/Documents/ResourcesLow/Datasets/KITTI/raw
done
```

验证集的完整 drive 名称也可直接列出：

```bash
find ~/Documents/ResourcesLow/Datasets/KITTI/depth/val -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
```

全部准备好后：

```bash
conda run -n aiaa3201 python -m depth_project.data.benchmark \
  --dataset kitti --source ~/Documents/ResourcesLow/Datasets/KITTI/raw \
  --depth-root ~/Documents/ResourcesLow/Datasets/KITTI/depth/val
```

不把第三方不同 split 的 KITTI 包混入该目录。网站登录、条款确认由你完成，凭据不写入仓库。

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
