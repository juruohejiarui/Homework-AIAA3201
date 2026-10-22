# 数据迁移报告

状态：**complete**。目标 UUID：`9E04A88104A85DD1`，实际挂载点 `/media/hjr/新加卷`，读写模式 `rw`。

源目录：`/media/hjr/Data/Resources/Datasets/AIAA3201_Project3`，已在复制及独立进程校验全部通过后删除。

总计：**4,167,446,009 字节**。

| 映射 | 普通文件 | 清单条目（含空目录/链接） | 字节 |
|---|---:|---:|---:|
| middlebury | 122 | 155 | 94598443 |
| nyuv2 | 1311 | 1318 | 4062077275 |
| videos | 1 | 1 | 10770291 |

完整逐文件 SHA256、大小、目录集合、软链接目标和阶段时间保存在 `../outputs/storage_history/migration_state.json`。该文件为本机运行凭证，不提交 Git，不参与日常运行。

- 原下载的两个 ZIP、labeled MAT 和 splits MAT 与已有下载哈希一致。
- `rsync -rl --checksum --partial` 仅复制，不跟随内部链接，不自动删除源。
- 独立 Python 进程重新读取源/目标，比较完整集合及每个文件 SHA256，并确认源未变化。
- 654 张 RGB 和 654 份 GT、MAT、ZIP CRC、视频缓存可读性通过。
- 3 个 smoke RGB 绝对软链接在源审计前改为等价相对链接，记录见 `../outputs/storage_history/source_link_repairs.json`；复制前后链接文本一致。
- 全部校验通过后删除的范围仅为旧 AIAA3201_Project3 目录。共享 Datasets 和原 Resources 链接保留。
- 新入口已建立，通过入口读取检查完成。HF_DATASETS_CACHE 已切换至 ResourcesLow/.cache/huggingface/datasets。
- 12 项临时目录迁移测试通过，覆盖中断恢复、错误 UUID、父挂载、只读、哈希冲突、已有目录、源变化及重复运行。之后按用户要求采用完整实际流程验证。

本报告不意味着 benchmark 复现完成；相关结果另见 `../paper_reproduction_zh.md`。
