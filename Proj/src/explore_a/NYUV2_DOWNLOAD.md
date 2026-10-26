# NYUv2 完整 Raw：官方分包、MD5 与手动下载

本入口与TUM一样分为 `plan`、`download`、`extract`；另提供独立 `verify`。完整Raw下载后用于相邻帧训练，不使用47,584图的预处理镜像代替全量Raw，也不限于labeled训练图。

## 下载和解压

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiaa3201
cd /home/hjr/Documents/Projects/Homework-AIAA3201

# 读取官网链接、逐包官方MD5和大小，不下载ZIP正文。
bash Proj/download_nyuv2.sh plan

# 下载完整Raw；aria2按官方MD5自动校验，支持续传。
bash Proj/download_nyuv2.sh download --jobs 3

# 单独解压，原包保留；不启动训练。
bash Proj/download_nyuv2.sh extract
```

Raw包沿用已有位置 `~/Documents/ResourcesLow/Datasets/nyu_depth_v2/archives/raw`，不会复制一套重复原包。解压到 `raw/scenes`，训练适配器会递归读取所有场景。

计划保存在 `Proj/storage/nyuv2/plan.json`；每个官方 `.md5` 原文及合并清单保存在相邻 `plan.metadata` 目录。已有计划无需重新plan，直接继续download。首次plan可加 `--output /绝对路径/nyu_depth_v2`；后续命令从计划读路径。

当前官网表格缺少此前清单中的 `classrooms.zip`。脚本会保留已知官方分包，重新获取其官方MD5和响应头验证，并在清单的 `discovery` 中标注“旧官方清单补充、当前表格未列出”。不会将网站表格缺项当作应删除的数据，也不引入第三方镜像。

2026-10-24已实际重新核对：表格37包＋官方直链的classrooms包，共38包，**421,539,441,278字节（421.54GB）**，全部38份官方MD5与原冻结清单一致。本次只读取网页、MD5文本和响应头，没有下载ZIP正文。核对结果保存在 `Proj/storage/nyuv2/official_sources.json`。

## MD5 行为

- 新下载包由aria2的 `checksum=md5=...` 和 `--check-integrity=true` 校验。
- 已存在的完整ZIP若没有有效校验记录，先读取并核对官方MD5，不能仅凭大小跳过。
- 校验成功后记录包路径、大小、mtime和官方MD5；未变化的包可复用记录，解压前不再重复全包读取。
- 校验失败返回错误并标明文件，保留原包供处理，不将其视为训练数据。

需要日后独立重新核对全部包时，可以执行下面的命令；它会实际重新读取并计算官方MD5。正常下载已校验，无需立即再重复执行。

```bash
bash Proj/download_nyuv2.sh verify
```

脚本同时保留旧 `--mode full|prepared|labeled` 接口；本次完整Raw推荐使用上面的新子命令。新入口只下载Raw，不重复下载本机已有的MAT和划分文件。如果改用新数据根目录，需将现有 `nyu_depth_v2_labeled.mat` 和 `splits.mat` 放入该根目录，用于场景级测试排除。

## 使用三来源训练

默认配置现在是 `Proj/configs/explore_a/mixed.yaml`，包含 **KITTI＋TUM＋NYUv2 Raw**。全部数据准备好后由用户执行：

```bash
bash Proj/explore_a.sh status
bash Proj/explore_a.sh prepare --extract-kitti
bash Proj/explore_a.sh all
```

`prepare`会生成固定清单；随后`all`复用它，继续即时计算训练与评价。若已有清单，不重复执行prepare；直接all。Raw里的官方测试房间、既有评价房间以及新划出的开发房间全部排除于训练。

所有可用训练Raw相邻帧对进入候选池，不设场景数或帧数上限。按用户选择保留30,000步和来源等概率、有放回采样，因此不保证每一帧都被抽到一次。labeled是Raw子集，不另行重复计入；MAT用于识别官方测试场景，训练深度监督来自Raw传感器投影后的有效测量。

数据来源与实际状态见 [DATASETS.md](DATASETS.md)，完整启动参数和held-out规则见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。官网：[NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)。
