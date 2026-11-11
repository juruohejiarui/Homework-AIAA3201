# Ensemble 评价

ensemble 已接入当前 evaluate_v2：仍使用 `evaluate_trained_models.py` 入口，
协议为 `shared-case-v2-ac-sintel`，默认系统输出为 `shared_models_evaluation_v2`。
Sintel 沿用 final 全部 1064 帧、逐图 AC 评分与独立的视频指标；不回退到旧版。

在 `maintenance/evaluate_models.json` 中注册普通成员，再以名称组成 ensemble。
`maintenance/evaluation_models.json` 是同一文件的符号链接。
`unet_guided_mean` 保留原有 lr1e-4 Farnebäck 三种子组合：

```json
{
  "name": "unet_guided_mean",
  "kind": "ensemble",
  "members": [
    "farneback_unet_guided_lr1e-4",
    "farneback_unet_guided_seed_1",
    "farneback_unet_guided_seed_2"
  ]
}
```

2026-11-11 新增 lr3e-4 批量评价配置。Farnebäck、RAFT 各有 seed 2026、42、114
三个已完成 30,000 步训练的模型，均使用各自的 `best.pt`，不使用 `last.pt`。

| 新增 ensemble 名称 | 成员 | 目的 |
| --- | --- | --- |
| `farneback_lr3e-4_mean3` | Farnebäck lr3e-4 三种子 | 检验同一光流下种子平均的收益 |
| `raft_lr3e-4_mean3` | RAFT lr3e-4 三种子 | 与 Farnebäck 三种子组合对照 |
| `farneback_raft_lr3e-4_mean6` | 上述全部六个模型，逐模型等权 | 检验两种光流的互补性 |

三个组合在评估前固定，不依据 test 或 benchmark 分数挑选成员。
所有注册模型、冻结参考及组合一起评价，保持共享前端和每种光流的计算：

```bash
OMP_NUM_THREADS=1 /home/hjr/miniconda3/envs/aiaa3201/bin/python \
  /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/maintenance/evaluate_trained_models.py run \
  --registry /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/maintenance/evaluate_models.json \
  --scopes benchmark test \
  --output /home/hjr/Documents/Projects/Homework-AIAA3201/Proj/outputs/explore_a/shared_models_evaluation_v2
```

TUM validation 输入准备好后可在 `--scopes` 中追加 `tum-validation`。
现有输出目录中身份仍匹配的已完成评分会复用；新增组合需要其成员预测，
因此处理这些组合时仍可能重新执行已评估成员的前向。完整结果写入同目录的 `comparison.md/.csv/.json`。

成员数量没有固定上限，至少一个，不允许重复名称。每个直接成员等权：
`prediction(x, y) = sum(member_prediction(x, y)) / len(members)`。
对当前融合头，平均的是原始 DA 逆深度输出，之后才执行原有评价对齐与评分。
不先按 GT 对齐各成员，不平均指标或模型参数，不训练 ensemble。

## 添加不同 seed 或学习率成员

例如在注册表中增加以下普通成员，再把名称加入 `members`：

```json
{
  "name": "farneback_unet_guided_seed42",
  "config": "configs/explore_a/mixed_h5_da2_diffseed_1.yaml",
  "checkpoint": "outputs/explore_a/mixed_h5_balanced/runs_da2_guided_30k_seed_1/farneback-rule-unet_guided/best.pt"
}
```

其他 seed、学习率和结构按实际 checkpoint 增加条目。使用已完成的 `best.pt`。
成员可为不同光流、不同融合结构，或具有相同输出表示的冻结 image/sequence 模型。
米制深度和相对逆深度不能放进同一个平均。输入、画布、前端和几何设置沿用公共兼容检查。
ensemble 也能作为另一个 ensemble 的直接成员；此时仍对直接成员等权，不展开成叶成员等权。

## 执行

仓库根目录、`aiaa3201` 环境中，只评价 ensemble：

```bash
OMP_NUM_THREADS=1 python Proj/maintenance/evaluate_trained_models.py run \
  --models unet_guided_mean \
  --scopes dev \
  --output Proj/outputs/explore_a/shared_models_ensemble_v2
```

完整评价范围按需使用 `--scopes test benchmark tum-validation`。benchmark 默认 long，
仍遵循现有完整 cohort 和评分协议。省略 `--models` 则评价注册表中的所有模型与 ensemble。
也可以同时传入普通模型和 ensemble 的名称。

```bash
python Proj/maintenance/evaluate_trained_models.py report \
  --output Proj/outputs/explore_a/shared_models_ensemble_v2
```

新实现会改变代码身份，请使用新的评价输出目录；旧结果保留。相同命令可恢复已完成 case。
若同名 ensemble 的成员或依赖身份改变，会拒绝混用旧结果；换新 ensemble 名称或输出目录。

## 执行与结果

- 仅选择 ensemble 时，自动加载其依赖；不额外输出未选择成员的独立评分。
- 同时选择成员和 ensemble 时，同一帧的成员只前向一次。
- 同一光流的几何准备、冻结 DA、光流和缓存读取由现有调度器共享。
- DGX 仍只执行所需普通成员，传回预测；主机按同一逻辑平均和评分，不传 GT。
- 普通帧模型逐帧平均，无整段预测缓存。若包含 sequence 模型，使用自动清理的临时文件。
- 输出的 `comparison.json/.csv/.md` 包含独立 ensemble 行。`session.json` 记录成员、
  checkpoint 快照和实现身份；各 case 的 ensemble 收据也记录其成员定义。
- ensemble 不加入训练模型选项，不创建优化器或 ensemble checkpoint。
