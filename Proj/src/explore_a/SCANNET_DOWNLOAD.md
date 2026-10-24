# ScanNet 训练子集：下载与校验

**当前需求已更正为完整 v1 数据集，请执行 [v1 全集说明](SCANNET_V1_DOWNLOAD.md) 中的命令。** 本文保留子集功能与共用校验机制说明。

2026-10-24。用户自行执行；本次编写脚本时未启动数据集下载。

## 下载内容

- 从官方 `scannetv2_train.txt` 选择场景，排除验证划分的物理空间。
- 默认随机种子2026，选择20个不同物理空间，各取一个扫描；三种融合器等实验共享该固定清单。
- `.sens` 使用官方复用的 `v1/scans/`，`.txt` 使用 `v2/scans/`。这是按 v2 划分管理的原始 RGB-D，不是将 v1/v2 当两套独立序列。
- 不下载测试集、网格、语义附件，不自动导出图像或启动训练。
- 文件保存在 `~/Documents/ResourcesLow/Datasets/ScanNet/scans/<scene>/`；本机现有 ResourcesLow 链接决定实际磁盘位置。
- 需要 Python 3 标准库和 aria2c，本机已找到 `/usr/bin/aria2c`；无需更改训练环境。

## 执行步骤

### 1. 生成20场景计划，只读取小型清单和响应头

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_scannet.sh plan --max-scenes 20
```

显示精确的计划字节数、GB/GiB、输出目录及官方摘要覆盖数量；此命令不下载 `.sens`/`.txt` 数据正文。

默认计划为 `Proj/storage/scannet/plan.json`；配套 `.metadata/` 目录保存所读取的官方 train/val 清单、选定场景，若提供官方摘要 URL 也保存其清单。计划文件已存在则拒绝覆盖，防止续传时重新抽样。

### 2. 下载并完成大小检查、可用摘要比对和本地基准记录

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_scannet.sh download --jobs 2
```

默认2个并发文件，每文件最多2连接。使用证书验证开启的 HTTPS、重试和 aria2 断点续传。完成后再将 `.part` 改名为正式文件，记录 `plan.verified.json`。

中断后重跑相同 `download` 命令；不要删除 `.part` 和 `.part.aria2`。已完成文件依据已有摘要复核后跳过。若已有正式文件尺寸或摘要不匹配，脚本会报错而非覆盖它；完成文件损坏时先由用户决定移走哪个文件再重下。

### 3. 日后独立复核（不联网、不下载）

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_scannet.sh verify
```

比较计划大小、可用官方摘要和先前记录的本地 SHA-256。没有任何摘要基准的文件不会被标为摘要校验成功。不要在首次基准记录前手动修改下载文件，也不要将收据重建当成已有文件真实性证明。

## 官方校验码的实际状态

目前**未取得官方 MD5/SHA 校验清单**。搜索 ScanNet 官方资料没有找到公开清单；官方根目录列举返回403，以下有限候选返回404：

`SHA256SUMS`、`MD5SUMS`、`checksums.txt`、`v1/scans.md5`、示例 `.sens.md5` 与 `.txt.sha256`。

前一轮所读取的数据响应头也没有 `Content-MD5`/`Digest`。这些结果不证明服务器上任何位置都不存在校验码，只表示本次没有找到可用官方来源。Apache ETag 是服务器标识，本脚本不会将其当作 MD5/SHA。

| 检查 | 能说明什么 |
|---|---|
| 官方 Content-Length 与下载字节数一致 | 检查大小、常见截断；不能排除同大小内容错误 |
| 官方发布摘要与本地摘要相同 | 若取得原始摘要，可核对下载内容与该参考一致 |
| 首次完成后保存本地 SHA-256 | 建立日后损坏/变化检测基准；**不能独立证明首次内容等于官方原件** |

默认在未取得官方摘要时执行大小检查并保存本地基准，收据中 `official_checksum_verified=false`，不会显示官方摘要通过。

若之后得到官方 GNU md5sum/sha256sum 格式清单 URL，可在**生成新计划**时使用：

```text
plan --official-checksums-url <官方HTTPS清单地址> --require-official-checksums
```

脚本会保存清单并匹配远端路径、本地相对路径或唯一文件名，缺失任一文件摘要时拒绝严格模式。只接受 ScanNet 两个官方服务器域名的 HTTPS 清单地址；不把用户自行计算或无出处的第三方哈希标成官方值。

也可对已有计划执行 `download --require-official-checksums` 或 `verify --require-official-checksums`；若当前计划无官方摘要，会在下载/复核前报错。计划生成后不能通过下载命令临时更换摘要来源。

## 调整数据规模

另建一个不同名称的计划即可；不要覆盖原有清单。

例如，先规划100个不同空间：

```bash
bash /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/download_scannet.sh plan \
  --max-scenes 100 \
  --manifest /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/storage/scannet/plan-100.json
```

下载和复核均加相同 `--manifest`。`--all-train` 则选择官方训练划分中所有符合隔离条件的扫描，不保证每空间只有一次扫描。`--scene-list /绝对路径/scenes.txt` 可以指定每行一个训练场景ID。

输出位置通过 `plan --output /绝对路径/ScanNet` 指定，然后保存在计划中。`download` 和 `verify` 只读取既有计划，不重新抽样，也不接受更换输出位置。

官方 v2 的1513原始扫描清单同时包含训练与验证；本脚本按正式训练清单选择，因此不会直接把1513个全部用于训练。

## 检查范围

本地使用临时小文件和模拟传输检查了4组逻辑：摘要清单解析、官方/本地摘要不匹配、成功下载后记录并复核、失败下载保留续传文件。Shell入口通过语法检查。没有运行实际大文件传输，实际网络续传仍取决于服务器和 aria2 的运行结果。

资料：[ScanNet 官方](https://github.com/ScanNet/ScanNet)、[官方划分](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark)、[本项目先前的官方端点核对](SCANNET_DOWNLOAD_REVIEW.md)。
