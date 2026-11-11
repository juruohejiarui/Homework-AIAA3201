# ETH3D：论文深度评价子集与下载

核对日期：2026-11-04。仅准备命令和核实元数据，未启动数据包下载。

## 下载范围

目标是官方 high-res multi-view **training** 部分：13 个场景、454 张 DSLR
图像以及对应的公开渲染深度。这里 training 是 ETH3D 原始立体重建基准的划分名称；
这些图像可以作为单目深度模型的测试集。它不是连续视频集合。

- Depth Pro 表 18 明确为 454 张，4032×6048。
- UniDepth 官方 ETH3D 类的 `test_split` 是 `train.txt`，并非官方隐藏 GT test。
- Depth Anything V2 报告标准 ETH3D 相对深度评价；未在论文中附独立逐图清单。
- Video Depth Anything 将 ETH3D 放在单图评价中，并说明沿用 DA-V2 的图像评价。
  不应将 ETH3D 加入其五个视频 benchmark。

固定下载清单为 `storage/eth3d/high_res_454_plan.json`，仅选取每场景的
`*_dslr_jpg.7z`（原始 JPEG 及标定）和 `*_dslr_depth.7z`（对应公开深度），
共 26 包。无需额外下载 NEF raw、undistorted RGB、扫描点云或 occlusion 包。
官网顶端的“all ground truth”是扫描评价资料，不能当成逐图深度包的体积。

场景：courtyard、delivery_area、electro、facade、kicker、meadow、office、pipes、
playground、relief、relief_2、terrace、terrains。

## 实际体积

本次逐包 HEAD 复核全部 26 个包的 Content-Length，均与固定清单一致。
解压体积来自 `storage/eth3d/archive_header_inventory.json` 的 7z 成员头部清单；
该清单的压缩字节数也与全部下载条目一致。

| 内容 | 压缩包字节数 | 下载 GB | 解压字节数 | 解压 GB |
|---|---:|---:|---:|---:|
| RGB 与标定 | 5,055,081,894 | 5.055082 | 5,145,111,098 | 5.145111 |
| 对应深度 GT | 5,930,741,673 | 5.930742 | 44,284,133,376 | 44.284133 |
| 合计 | 10,985,823,567 | 10.985824 | 49,429,244,474 | 49.429244 |

保留压缩包加解压文件合计 60,415,068,041 字节，60.415068 GB，约 56.265917 GiB。
GB=10^9 字节，GiB=2^30 字节。官方未提供此清单对应的文件摘要；下载检查大小，
7z 解压检查归档 CRC，解压后 verify 检查图像、深度文件名及数量和标定文件。

## 执行命令

从仓库根目录执行，激活 aiaa3201：

```bash
cd /home/hjr/Documents/Projects/Homework-AIAA3201
conda activate aiaa3201
plan="$PWD/Proj/storage/eth3d/high_res_454_plan.json"
bash Proj/download_eth3d.sh status --manifest "$plan"
bash Proj/download_eth3d.sh download --manifest "$plan" --jobs 2
bash Proj/download_eth3d.sh extract --manifest "$plan"
bash Proj/download_eth3d.sh verify --manifest "$plan"
```

下载中断后重跑 download，解压中断后重跑 extract。既有完成包按冻结清单复用。
解压需要 `7z` 或 `7zz`；本机已经有 `/usr/bin/7z`。

固定计划解析后的目标是：

```
/media/hjr/Data/Resources/Datasets/ETH3D/high_res_multiview_454/
  archives/   # 压缩包
  data/       # RGB、标定、深度
  extracted/  # 解压完成记录
```

此路径对应 `~/Documents/Resources/Datasets/ETH3D/high_res_multiview_454`；
`~/Documents/Resources` 当前解析到 `/media/hjr/Data/Resources`。
下载和解压均在 Resources，不使用 ResourcesLow。

## 隐藏 GT 的官方 test 与提交

本方案已有公开 GT，可直接在本地计算单目深度指标，不需要提交。
官方隐藏 GT test 是另外的立体／多视图重建评测，不能用它获得上述论文的单目
AbsRel／δ1 成绩。若另做该任务，入口是 https://www.eth3d.net/login ，
格式规则是 https://www.eth3d.net/documentation#result-submission ：

- 注册、登录后通过 Submit 上传单个 zip/7z。
- high_res_multi_view 或 low_res_many_view：每场景一个 binary PLY 点云和同名 txt。
- low_res_two_view：每样本一个 float32 PFM 视差和同名 txt。
- txt 记录 `runtime <秒数>`；压缩包根目录使用对应任务名称。
- 单目相对逆深度 NPY/PNG 不能直接作为这些任务的提交文件。

## 一手来源

- ETH3D 数据和分包：https://www.eth3d.net/datasets
- 深度与提交格式：https://www.eth3d.net/documentation
- Depth Pro 表 18：https://arxiv.org/html/2410.02073v2#A3
- UniDepth 数据类：https://github.com/lpiccinelli-eth/UniDepth/blob/main/unidepth/datasets/eth3d.py
- DA-V2：https://arxiv.org/html/2406.09414v2
- VDA 图像／视频评价：https://arxiv.org/html/2501.12375v3
