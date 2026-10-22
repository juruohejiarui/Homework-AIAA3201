# 存储配置与下载清单

此目录只保存人工审核的配置、冻结下载清单和最终迁移说明，不放执行日志或 Python 入口。

- `fstab.proposed`、`fstab.diff`：供人工审核，不由程序覆盖系统配置。
- `fstab.original`、`fstab.original.sha256`：生成审核稿时的基线。
- `nyuv2_full_download_manifest.json`：38个完整 Raw 分包的字节数、来源和官方 MD5。
- `nyuv2_raw_aria2.txt`：可直接交给 aria2 的带校验和清单。
- `nyuv2_raw_urls.txt`：纯 URL 清单。
- `migration_report_zh.md`：已经完成的复制、独立校验和入口切换结果。

实际下载和解压命令见 [下载交接](../download_data_zh.md)。运行工具位于标准包 `depth_project.storage`，不依赖旧迁移状态文件。

原逐文件 SHA256 迁移清单和评价器验证数据保存在 `outputs/storage_history`，保留结果但不参与日常运行。旧日志、临时锁和进程 PID 文件已删除。

## 自动挂载审核

新加卷 UUID 为 `9E04A88104A85DD1`，挂载点为 `/media/hjr/新加卷`。本任务不执行 sudo、不修改 `/etc/fstab`。

如果尚未应用自动挂载配置，先比较当前 `/etc/fstab` 与 `fstab.original`；有其他变化时应重新合并，不能直接覆盖。人工应用顺序为备份 `/etc/fstab`、创建挂载点、覆盖审核稿、`systemctl daemon-reload`、挂载该分区。已有手动挂载无需为运行项目重复应用。
