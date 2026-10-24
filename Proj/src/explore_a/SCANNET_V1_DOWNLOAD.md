# ScanNet v1 完整数据集下载

这是用户当前确认的范围：**完整 v1 发布数据，不抽样、不限训练划分、不混入 v2 文件**。此前20场景方案仅保留为可选历史方案。

## 完整范围

- 官方 `v1/scans.txt` 的全部1513个扫描。
- 每个扫描全部13种文件：RGB-D `.sens`、v1 `.txt`、网格、分割/实例/语义标注及2D标注压缩包。
- v1 标签映射，以及官方独立 `--task_data` 分支的分类/体素任务数据和预训练模型，共5个任务文件。
- 共19674个数据文件，全部使用 `https://kaldir.vc.in.tum.de/scannet/v1/` 地址。
- 输出默认 `~/Documents/ResourcesLow/Datasets/ScanNet/v1/`，与此前 v2 元数据隔离。

2026-10-24 已联网确认 v1 场景清单有1513项，并对一个扫描的13类文件及5个任务文件逐一检查 HEAD，均返回200及大小。没有在本次检查中启动全集下载；全部文件总大小由下面的 plan 命令实际汇总，不沿用旧脚本的1.2TB粗略标注。

2026-10-24 全量计划已生成。根据 `Proj/storage/scannet/v1-full.json` 的官方响应头大小汇总，19674个不同URL合计 **1,432,450,292,390字节，即1.43245TB / 1.30281TiB**。其中 `.sens` RGB-D 数据1029.506GB，网格249.949GB，2D标注压缩包113.871GB，元数据及分割JSON33.606GB，附加任务5.518GB（此处GB均为十进制）。这是下载文件本身的总大小，不包含解压或导出帧新增空间。

此前将完整 ScanNet 估为约400GB不正确，应以本次完整v1清单为准。排除 `.sens` 后的其余文件约402.944GB，但没有证据表明这就是此前估计的来源。此前另一次约1.649TB统计对应v2默认全集，范围不同。

## 用户执行

```bash
cd /home/hjr/Documents/Projects/Homework-AIAA3201

# 1. 汇总全集文件和准确大小；只读取清单及响应头，不下载数据正文
bash Proj/download_scannet_v1.sh plan

# 2. 下载全部 v1 文件，支持续传，并执行下载后校验
bash Proj/download_scannet_v1.sh download --jobs 2

# 3. 日后独立复核，不联网、不下载
bash Proj/download_scannet_v1.sh verify
```

plan 需要查询约2万个文件的大小。默认元数据并发32，每完成50项显示进度、处理速度和预计剩余时间；可加 `--metadata-workers 64` 提高并发。实际加速取决于服务器响应和是否限流，不保证线性提升。此阶段只读响应头，带宽和磁盘利用率低是正常现象。

已完成的大小查询逐条保存到 `Proj/storage/scannet/v1-full.heads.jsonl`，中断后重跑 plan 会复用缓存。旧版正在运行的进程不会自动应用更新，需要先 Ctrl+C 停止再重新运行；旧版未保存的进度无法恢复。不要同时运行两个相同计划的 plan 进程。缓存复用的是先前查询到的大小；若明确需要重新查询服务器，可先将该缓存文件改名，再运行 plan。

计划固定保存在 `Proj/storage/scannet/v1-full.json`。如果此前生成了20场景的 `plan.json`，无需删除或覆盖；v1入口使用独立计划。

下载中断后重跑同一 download 命令，保留 `.part` 和 `.part.aria2`。计划已存在时不用重新 plan。若要改输出磁盘，在第一次 plan 时加 `--output /绝对路径/ScanNet-v1`；后续 download/verify 从计划读取路径。

## 校验含义

目前未找到可用的官方 MD5/SHA 校验清单，因此默认执行：官方大小检查、保存本地 SHA-256 基准、日后与本地基准比较。**不会把本地哈希记录标记为官方摘要验证通过**。如果获得官方清单URL，可在首次 plan 时加 `--official-checksums-url <官方HTTPS地址> --require-official-checksums`，同时下载清单并严格校验。

所保存的官方 ETag 不是 MD5/SHA 校验码。没有官方摘要时不能仅凭大小一致宣称内容逐字节等于原件。详情见 [校验说明](SCANNET_DOWNLOAD.md)。

完整下载不代表将1513个扫描全部用于训练；后续训练/验证仍按物理场景隔离。
