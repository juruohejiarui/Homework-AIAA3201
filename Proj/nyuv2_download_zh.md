# NYUv2 完整下载与训练数据说明

**2026-10-24 新入口：** 推荐使用TUM风格的 `bash Proj/download_nyuv2.sh plan|download|verify|extract`，会逐包读取官网MD5并校验已有ZIP；旧接口仍保留。详见 [完整Raw下载说明](src/explore_a/NYUV2_DOWNLOAD.md)。默认方向A训练已改为NYUv2 Raw＋TUM＋KITTI，具体划分与真实数据状态见 [DATASETS.md](src/explore_a/DATASETS.md)。下文为旧接口说明。

2026-10-23 更新。脚本由你执行；本次没有启动大文件下载。

## 1. 先区分三种数据

| 数据 | 内容与用途 | 下载规模 | 当前来源 |
|---|---|---:|---|
| 完整 Raw | 464场景的原始 RGB、传感器深度和加速度数据，可自行同步、投影和选择训练序列 | 冻结清单38个ZIP，共421,539,441,278字节，约421.54GB | 本次未找到确认完整的公开 HF/ModelScope 镜像，使用官方分包 |
| 官方 labeled MAT | 全部1449对，含米制 depths、rawDepths、语义/实例信息与场景名 | 2,972,037,809字节，约2.97GB | 本机已存在完整文件，直接复用；缺失时走官方地址 |
| 预处理训练镜像 | FastDepth风格 H5，数据卡标称47,584训练图、654评估图 | 14个TAR，约35.15GB | 优先 HF sayakpaul/nyu_depth_v2；同内容 heuliuh/nyu_depth_v2 备用 |

完整 Raw 与预处理训练镜像是两种不同资产，不能把47,584张子集称作全部原始视频。Raw 的“unlabeled”主要指没有完整人工语义标注，仍有 Kinect 深度测量；训练前需要做同步、深度转换、相机投影与有效区域处理。

本机 labeled MAT 已包含全部1449张，不只是现有 test/ 中导出的654张。795训练图由 splits.mat 的 trainNdxs 指定，654测试图由 testNdxs 指定；仅需在后续训练准备时导出训练索引对应内容，无需重新下载 MAT。

## 2. 完整 Raw + labeled：你当前要的完整下载

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_nyuv2.sh
```

默认 full 模式包含38个 Raw ZIP、原始 labeled MAT、固定版本的 splits.mat 和官方工具箱 ZIP。已存在且长度符合清单的文件会复用，不重复哈希。尚未完成的文件会续传；新下载的 Raw 使用官方 MD5，MAT/split 使用项目已知 SHA256，由 aria2 完成一次校验，不额外再扫一遍。

中断后重复同一命令，保留 .aria2 文件。不要同时启动两个相同模式的下载进程。脚本只下载、不自动解压、不启动训练、不改变已有 test/ 数据。

只查看下载计划、不下载：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_nyuv2.sh --plan
```

实际路径始终根据当前 ResourcesLow 链接确定，不再使用旧清单里写死的挂载点。需要 aria2c；本机已经安装，不安装或改动 Python 环境。

## 3. 需要现成的大规模图像训练集时

以下模式下载 HF 预处理训练镜像，同时保留/补齐 labeled、split 和工具箱，不下载421GB Raw：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_nyuv2.sh --mode prepared
```

它可与 full 模式先后运行；两者各自保存，互不替代。HF 主仓库固定提交50579a0b591445f91bf7269b28611ad1cc6a05d2，备用仓库固定提交2417527acfbc40025d8e041ebbd06f6982bb94e4；14个分片的长度和 LFS SHA256 均已记录，两个仓库的对应分片 SHA256 相同。

数据直接以 TAR/H5 下载，不调用仓库的旧式 dataset Python loader，不把浮点深度转成8位灰度图。已读取到首个路径 train/basement_0001a/00001.h5；H5 中使用 rgb 和 depth 字段。正式训练前仍需按所采用的 FastDepth 预处理定义确认尺寸、深度单位和无效值；不能仅从 image 类型或灰度外观推断单位。

镜像的 val 在语义上是官方654图评估集，应保持为测试数据。内部调参验证集从训练场景中另划，不能反复用这654张选择超参数。

仅补齐或复用 labeled 资产：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_nyuv2.sh --mode labeled
```

## 4. 保存布局和系统解压命令

```text
~/Documents/ResourcesLow/Datasets/nyu_depth_v2/
├── nyu_depth_v2_labeled.mat
├── splits.mat
├── archives/raw/*.zip
├── raw/scenes/
├── toolbox/toolbox_nyu_depth_v2.zip
├── prepared_fastdepth/archives/*.tar
├── prepared_fastdepth/train/
├── prepared_fastdepth/val/
└── test/                         # 已有654张课程测试数据
```

下载全部完成后，Raw 使用系统 unzip 解压，保留 ZIP：

```bash
dataset="$HOME/Documents/ResourcesLow/Datasets/nyu_depth_v2"
mkdir -p "$dataset/raw/scenes"
for archive in "$dataset/archives/raw/"*.zip; do
  unzip -q -n "$archive" -d "$dataset/raw/scenes"
done
unzip -q -n "$dataset/toolbox/toolbox_nyu_depth_v2.zip" -d "$dataset/toolbox"
```

训练镜像使用系统 tar 解压，保留 TAR：

```bash
dataset="$HOME/Documents/ResourcesLow/Datasets/nyu_depth_v2"
for archive in "$dataset/prepared_fastdepth/archives/"*.tar; do
  tar --extract --file="$archive" --directory="$dataset/prepared_fastdepth"
done
```

分片原包保留；若解压中断，可重新解压未完成的分片，用包内原文件覆盖该分片的不完整输出。不要对仍带 .aria2 的包执行解压，也不要把人工修改的训练文件放到原始展开目录。

完整 Raw 的 ZIP、展开数据、同步投影结果和训练缓存会同时占空间，建议按超过1TB的工作区规划；精确需求取决于是否保存未压缩帧、填充深度和重复缓存。下载/解压主要占网络、磁盘和 CPU，不占模型训练显存。

## 5. 镜像核查结论

| 候选 | 核查结果 | 本脚本处理 |
|---|---|---|
| sayakpaul/nyu_depth_v2 | 公开，可读14个 TAR/H5 分片，预处理训练子集 | prepared 模式优先使用 |
| heuliuh/nyu_depth_v2 | 对应14个分片 SHA256 相同 | prepared 模式备用 URI |
| SRPLB1/benchmarks-nyuv2 | 有同名2.97GB MAT，但仓库需要人工授权，未授权请求401；内容哈希未确认 | 不作为默认可用源，缺 MAT 时用官方 |
| WyettZ/nyuv2-raw-depth | 448×448 的 labeled 变体，797/652划分 | 不替代官方1449划分，也不替代完整 Raw |
| 0jl/NYUv2 | 主要是从官方地址下载的 loader，未托管完整 Raw | 不当作下载加速镜像 |
| ModelScope 搜索结果 | 未找到本次可确认完整的 Raw 或可直接使用的等价 MAT 资产 | 不因名字包含NYU就替换数据 |

来源与校验清单在 storage/nyuv2_download_sources.json、storage/nyuv2_full_download_manifest.json。旧的 nyuv2_raw_aria2.txt 含历史固定挂载路径，本次新脚本不使用它，请使用新入口。

## 6. 训练与评价边界

- labeled 训练使用795张，最终评价保留654张；不把全部1449张混在一起训练后继续报告原测试集分数。
- Raw 训练按场景分离，保留测试场景；不能把测试图附近的同场景视频帧混进训练。BTS/FastDepth等还各有训练取帧规则，不应只凭同为NYUv2就混用划分。
- 现有 nyuv2_raw 导出器针对论文八个视频场景，不是全部训练场景预处理器。完整 Raw 下载完成不代表训练集同步、投影与 split 已经处理完成。
- 训练时的监督、尺度标定和测试集 oracle 对齐要分开记录。
- KITTI 和其他训练数据可以并用，但要保持米制单位、不同相机内参、有效深度上限与场景划分，见 [多数据集训练说明](multi_dataset_training_zh.md)。

资料：[NYUv2 官方页面](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[HF 主训练镜像](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2)、[备用镜像](https://huggingface.co/datasets/heuliuh/nyu_depth_v2)、[BTS](https://github.com/cleinc/bts)、[FastDepth](https://github.com/dwofk/fast-depth)。
