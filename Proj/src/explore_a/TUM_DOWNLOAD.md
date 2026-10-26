# TUM RGB-D：官网全普通序列下载

2026-10-24 已从[官网](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)抓取89个不同 `.tgz`：47个普通RGB-D序列、33个 `_validation` 序列、9个标定包。原始链接快照保存在 `Proj/storage/tum/official_links.json`。没有下载任何数据包。

默认下载全部47个普通序列，保留原包，解压单独执行。`_validation` 缺少公开轨迹真值，不等于其PNG必然没有深度；不用于本方案首轮训练。无需下载重复的ROS bag。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiaa3201
cd /home/hjr/Documents/Projects/Homework-AIAA3201

# 重新解析官网，汇总47包实际大小，只下载网页和响应头。
bash Proj/download_tum.sh plan

# 由你启动下载；默认3个文件并发，支持重跑续传。
bash Proj/download_tum.sh download --jobs 3

# 下载结束后，单独解压。
bash Proj/download_tum.sh extract
```

默认根目录 `~/Documents/ResourcesLow/Datasets/TUM-RGBD`，原包在 `archives`，解压结果在 `sequences`。计划固定为 `Proj/storage/tum/plan.json`，下载/解压按已保存的计划读取目录，不随当前默认路径变化。

如果需要全部89包，使用独立计划；不要覆盖已有普通序列计划：

```bash
bash Proj/download_tum.sh plan --all --manifest Proj/storage/tum/plan-all.json
bash Proj/download_tum.sh download --manifest Proj/storage/tum/plan-all.json
bash Proj/download_tum.sh extract --manifest Proj/storage/tum/plan-all.json
```

首次plan可用 `--output /绝对路径/TUM-RGBD` 改下载磁盘。官网无本脚本可用的摘要清单，下载后检查计划大小，解压读取gzip/tar结构；不会宣称已经完成官方MD5/SHA认证。已有同名文件大小不符且不是有效续传时会报错，保留文件供用户处理。

训练时再做RGB/深度时间戳关联、5000比例换算及场景划分。下载完整数据不意味着允许将全部序列用于训练，详见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。
