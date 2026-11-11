# 实验结果分析

- [2026-11-11：全部评估结果与 Sintel 诊断](SINTEL_RESULTS_ANALYSIS_20261111_zh.md)
- [本次诊断的图片、动图、结果快照与脚本](sintel_20261111/)

这里只保留经过整理的报告和支撑证据。完整训练记录、检查点、缓存与批量评估输出
仍保存在被忽略的 `outputs/`、`checkpoints/` 等本地目录，不加入 Git。

`sintel_20261111/diagnose.py` 可重跑五段视频抽样诊断；需要本地数据、检查点及
`outputs/explore_a/shared_models_evaluation_v2` 中固定的模型快照。
`report.py` 从当前评估结果和诊断 JSON 重新生成报告及图表；运行它会更新报告快照。
从仓库根目录使用 `PYTHONPATH=Proj/src` 和项目现有 Python 环境执行。
