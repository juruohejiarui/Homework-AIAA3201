# 实际验证记录（2026-10-21）

## 重写后的实际验收

以下结果来自新的 `depth_project` 包，输出位于 `outputs/rewrite_acceptance/`。后面的首轮记录是历史实验，旧入口已删除，不再作为运行方式。

- 显式使用 conda 环境解释器与 `conda run -n aiaa3201` 均可导入 PyTorch 2.12.0+cu130；`conda list` 显示 torch 为该环境中的 PyPI 包。此前使用 uv pip 指定该 conda 解释器安装，没有创建独立 uv 环境。本次只通过 pip 注册本地包，未重装环境依赖。
- DA-V2 Small 和 VDA Small 各完成官方示例的35帧 FP32 推理；Depth Pro 完成现有 NYUv2 的3张 smoke 图像推理与深度评价。
- SGBM 完成 Adirondack 一个带 GT 场景；独立光流评价、对比视频、指标曲线和立体匹配图片均已生成。
- 新官方评价入口读取已有 Sintel 长视频预测，完成23序列、1064帧评价：AbsRel=0.3521193130954613、RMSE=6.896128067559545、δ1=0.5916672391557712，与原记录一致。此次未重复运行完整 benchmark 推理。
- Python 源码编译与 Git 补丁空白检查通过。没有扩展测试矩阵或重新下载数据。
- `storage/` 的旧日志、锁、PID 与临时审计文件已删除；详细迁移凭证保存在 `outputs/storage_history/`。下载清单、挂载审核文件及最终迁移说明保留。

完整 NYUv2 Raw 下载尚未启动，操作见 `download_data_zh.md`。KITTI、ScanNet 仍待用户授权与下载，五套 benchmark 尚未全部完成。

## 已完成

1. 保留 aiaa3201 的 Python 3.13.16，安装 PyTorch 2.12.0+cu130 / torchvision 0.27.0。
2. 升级 NumPy/Pandas/SciPy/h5py/OpenCV/timm 等科学计算与模型库，安装 xFormers 0.0.35 与本地 FlashAttention 2.8.3 wheel。
3. pip check：No broken requirements found。
4. SDPA / FlashAttention / xFormers 的 CUDA FP16 前向与反向通过；对 FP32 math reference 的最大前向绝对误差约 2.79e-4。
5. xFormers info 显示 cutlass、FA2、Triton 等可用。wheel 构建信息是 torch2.10/cu130，但新版 ABI 在本机 torch2.12 实际算子测试通过；没有仅凭版本字符串认定可用。
6. 三个官方 submodule 已初始化并记录提交。Depth Pro NumPy 声明按 Python 版本分流，补丁保存于 patches/，幂等应用验证通过。
7. 数据与权重完整下载，weights_only=True 可读取所有 checkpoint；NYU HDF5 形状正确。SHA256 见 downloads_manifest.json。
8. Datasets 已迁至新加卷，Proj/data 为包含按数据集命名软链接的入口目录；HF_DATASETS_CACHE 已切换至 ResourcesLow/.cache/huggingface/datasets。
9. Middlebury：15 场景 BM 和 15 场景 SGBM；深度、左右一致性、coverage 与扰动敏感性均输出。
10. NYU：654 标准测试帧 DA-V2 与 654 帧 Depth Pro 推理，三个评价设置（SSI、median、metric-none）全部完成。
11. 官方视频：35 帧同输入 DA-V2 / VDA，光流评价、固定整段色标视频与预览可解码。
12. 完整 VDA FlashAttention 路径与 xFormers 对照：本输入 FP16 输出 max_abs=0、mean_abs=0。SDPA 通过算子级验证；额外完整 VDA SDPA 对照未执行（该可选命令未获批准），不将其写成通过。
13. 六项回归测试通过：米制 doffs、已知深度误差、PFM 字节序与方向、backward warp 方向、负斜率 SSI 保留、Middlebury 非单位 PFM 头部。
14. 环境脚本 bash 语法、submodule 补丁幂等性均通过。

## 调试中修复的问题

- 旧 PFM 读取器把头部 -0.003922 的绝对值乘进 Middlebury 原始像素视差，导致 GT 缩小约255倍。已移除该乘法，增加真实头部回归测试，并重跑全部 BM/SGBM；旧错误分数已被正确结果替换。
- NYU 0650 的 inverse-depth SSI 无约束斜率为负。采用显式 slope>=0 约束的边界最优常数解，保留样本、记录计数；没有跳过失败例，也没有将其称为官方协议。
- Depth Pro 已知焦距分支需要 Tensor，wrapper 已做转换。
- Python 3.13 无 NumPy1.x wheel；用 NumPy2.x 并修订 Depth Pro 过时依赖声明，没有降级 Python 或 torch。

## 实际范围与局限

- NYU crop/GT/depth bounds/alignment 使用 GUIDE_zh.md 明示的参考协议，尚需助教确认完整官方 evaluator。
- 视频使用作者仓库官方示例；尚无用户自录 static/moving/hard clips，不能据此宣称完成所有课程视频要求。
- 时间误差是固定 clip gauge 下的光流一致性；moving-camera median-depth 变化不等于真实尺度漂移。
- 某些 CPU 评估与模型推理同时进行，profile 属于首轮开发记录；正式效率表应在空闲机器单独重复。
- Part 3 已审阅并交付方案，尚未实现创新方法、消融、真实应用评价或最终报告。



本次共享迁移与论文 benchmark 的实际验收见 storage/migration_report_zh.md、outputs/storage_history/evaluator_validation.json 和 paper_reproduction_zh.md。原始指标与兼容评价器在 Sintel alley_1 全50帧的最大差异为1.53e-8；三维 TAE 的恒等位姿和已知 Z 平移检查均为0，后者不是 ScanNet 实测值。
