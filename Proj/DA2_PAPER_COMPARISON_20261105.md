# DA-V2 原论文与当前 Sintel / KITTI 结果对照

2026-11-05。已核对原论文、官方讨论和公开后续评估源码；完成 Sintel final 全量 1,064 帧 FP32 推理（两种输出插值）及 KITTI 161/652 帧冻结 DA-V2-S 推理。未修改训练、生产评估程序或原有结果。


## 当前采用的参考：DepthAnything-AC

用户于 2026-11-05 接受暂时使用这篇后续论文作为参考。论文为南开大学团队 Boyuan Sun、Modi Jin、Bowen Yin、Qibin Hou 的 **Depth Anything at Any Condition**，2025-07-02 发布，arXiv:2507.01634。当前官方页面与仓库以 arXiv 预印本引用，未据此宣称已在正式会议/期刊发表，也不将其当作 DA-V2 原作者的评估代码。

采用依据是这套评估的实际可复现性：公开逐帧名单、GT 读取代码、评分代码，以及我们对同一官方 DA-V2-S 权重的独立全量复现。

| 数据集 / 参考 | 论文中的 DA-V2-S AbsRel / δ1 | 本地复现 AbsRel / δ1 | 状态 |
|---|---|---|---|
| AC 表 5，Sintel final，1,064 帧 | 0.240 / 0.712 | **0.239519220 / 0.712226400** | 直接执行未修改的公开评分函数，按论文精度完全一致 |
| AC 表 5，KITTI 离散 652 帧 | 0.083 / 0.934 | 0.083170305 / 0.933985807 | 全部正 GT 拟合、逐图深度评分复算，按论文精度一致；KITTI 推理沿用官方 infer_image 的 align_corners=True，未声称所有推理细节逐项相同 |
| DA-V2 原论文表 2，Sintel | 0.500 / 0.718 | 尚未匹配 | 不再用该行作为当前 Sintel 直接数值对照 |

**采用的 Sintel 协议：** training/final 的 23 段全部 1,064 帧，GT 先截断到 72 m，保留所有正 GT 像素；逐图在逆深度上拟合 scale/shift，预测逆深度下限 1/80，再转回深度并截断至 [1e-5,80] m；在深度空间算 AbsRel/δ1，最后 1,064 帧等权平均。恢复预测分辨率时 align_corners=False。这里不能换成“排除 GT>72 m”，也不能换成“各视频先平均再等权”。

这套协议用于当前 Sintel 空间精度的论文对照；现有连续视频测试仍保留原始 cohort、视频统一对齐和对应结果。AC 的 KITTI 652 张不能直接替换融合头的 held-out 视频测试，因为其中 396/95 帧属于当前训练/dev 清单。

机器可读协议为 `outputs/audits/sintel_kitti_20261105/adopted_protocol.json`；全量逐帧结果和源码哈希为 `ac_source_reproduction.json`。生产评估已接入：共享评估、独立 benchmark 与参考模型入口统一使用本节 Sintel 数据和评分标准。默认输出目录已切为 shared_models_evaluation_v2，旧结果不迁移。已新增 da2_fp32 注册项，固定 FP32、短边518、align_corners=False，用于与本次复现基线对照；训练前端 raw_native/raw_canvas 仍保持训练时配置。生产评分函数通过全量1064帧与未修改AC源码的数值一致性检查，实际参考注册项另通过一帧推理逐位核验。

## 追查原论文得到的关键更正

[DA-V1 作者在官方 issue #174 的回复](https://github.com/LiheYoung/Depth-Anything/issues/174)明确说明：GT 转为 disparity，原始预测经 scale/shift 拟合后直接在 disparity 空间评测，不转回深度，也不先做 0–1 归一化。这是同系列作者的直接说明；DA-V2 的公开相对评估请求 #91/#98 仍没有完整实现，不能把这条回复当作已得到 DA-V2 全部评估细节。

因此此前默认“拟合逆深度后转回深度就等同原论文”的判断需纠正。若对齐后的正逆深度为 q、GT 深度为 z，disparity AbsRel 为 |q−1/z|/(1/z)，depth AbsRel 为 |1/q−z|/z；二者数值不相同。δ1 在全部预测为正且没有截断时对倒数变换保持不变。

已依此对全部 clean/final 预测重新计算。GT 截断 72 m、未经额外下限截断的逆深度评分：clean 0.464593 / 0.709698，final 0.523329 / 0.712288；final 的 align_corners=False 为 0.523513 / 0.712301。尚未匹配原表的 0.500 / 0.718。负逆深度如何处理、GT 的实际截断/掩码、原表精确 cohort 等细节仍未恢复；不将最接近的诊断设置标成原论文复现成功。完整固定诊断保存在 `disparity_reproduction.json`。

## KITTI 名单是否连续

对 DA-V2 公开的 652 张文件名单逐项统计：28 个 drive，段内排序后的 624 个相邻关系中，**613 处有跳帧，11 处连续**；最长连续小段为 12 帧。这是稀疏抽样名单，并未覆盖各 drive 的每一帧。当前项目 13 段、2,806 帧的段内 2,793 个相邻关系均相差 1。

DA-V2 原论文相对深度表 2 没有公开足以锁定其确切名单的完整材料；其仓库 `metric_depth` 名单不能单独证明表 2 的 cohort。能够确认的是：公开 652 张名单属于离散图像评估，VDA 官方视频脚本则按连续序列提取；当前 benchmark 与前者的取样、与逐图评分的对齐单位都不同。完整统计见 `kitti_cohort_continuity.json`。

## 早期结果：原论文与当前 benchmark 的直接数值对照

表内为 AbsRel / δ1。δ1 在本表以比例记录。

| 数据集 | DA-V2 原论文表 2，ViT-S | 本地 native FP16 逐图 | 本次独立 FP32 逐图复算 | 本次独立 FP32 整视频复算 |
|---|---|---|---|---|
| Sintel | 0.500 / 0.718 | 0.250964817 / 0.711981879 | 0.251133158 / 0.711898469 | 0.427650208 / 0.539001892 |
| KITTI | 0.078 / 0.936 | 0.085067258 / 0.931136168 | 0.085109923 / 0.931169271 | 0.128544232 / 0.835726464 |

原论文 PDF 第 8 页表 2 已通过页面渲染核对。当前 native 预测不经过融合头使用的 canvas 往返，适合用于检查 DA-V2 基线；仍不能在缺少原论文完整协议的条件下称为严格复现。

当前 native FP16 的整视频指标分别为 Sintel 0.427425040 / 0.539039634、KITTI 0.128525334 / 0.835735857。

## 单图、离散取样和视频必须分开判断

- KITTI Raw 本身是 10 Hz 同步录制的视频序列。官方 Depth Prediction 任务按单张图像的深度输出评分，未定义跨时间一致性指标。不能因此把 KITTI 原始数据称为没有连续图像的单图数据集。
- DA-V2 原论文表 2 属于单图相对深度评估。VDA 论文表 2 将相同的 DA-V2-L 原论文数值列在 single-image depth estimation 表中；VDA 另以表 1 报告视频结果。
- “逐张输入”不能推出“离散抽帧”。DA-V2 没有公开足以核对其原论文 Sintel 逐帧名单的评估代码和取样列表，正文与附录也未交代连续片段覆盖。此次官方 GitHub issue 搜索没有找到 Sintel 评估说明；相对深度评估请求 #91 和 #98 未给出作者发布的完整评估实现。
- 因此，能确认 DA-V2 原论文结果是 image-benchmark 成绩；不能进一步把其 Sintel cohort 判为离散抽样，也不能判为 23 个完整连续视频。缺失点是评估名单和实现，不是模型是否接受视频张量。
- 可以确认完整视频协议的是 VDA：逐场景读取帧，DA-V2 每帧独立推理，整段视频拟合同一对 scale/shift，再对帧指标汇总。按用户关注“是否实际覆盖连续片段”的定义，这属于视频评估。

## 实际 cohort 和数据验证

| 数据集 | 输入 | 相邻编号检查 | 有效评分 |
|---|---|---|---|
| Sintel | training/clean，23 段、1,064 帧 | 1,041 个段内相邻关系全部相差 1 | 1,045 帧在 0.001–70 m 内有有效 GT；mountain_1 的其余 19 帧没有此范围内有效像素，按既有规则不计分 |
| KITTI | depth/val 对应 image_02，13 段、2,806 帧，每段最多 500 帧 | 2,793 个段内相邻关系全部相差 1 | 2,806 帧 |

Sintel 所有 1,064 帧准备后的 float32 GT 均逐像素等于对应官方 DPT 解码后的米制深度，所有现有 FP32 预测均通过原 complete.json 的 SHA-256。没有发现本地 Sintel 深度转换或预测文件损坏。

KITTI 较早的预测收据没有保存哈希；本次核对 source_indices 并为全部现有预测记录当前 SHA-256，不把这些新哈希写成历史完整性证明。

## 评分复算与敏感性检查

1. 使用既有 FP32 原始预测，在相同 GT 掩码和深度范围下独立重新计算逐图与整视频对齐；相同权重 SHA-256 与当前评估记录一致。
2. 与较早 VDA 官方评估入口保存的结果比对：Sintel 原结果 0.427650214 / 0.539001980；KITTI 原结果 0.128544232 / 0.835726612。本次独立复算与其几乎相同，差异在浮点精度量级。当前共享评估的 FP16 结果也接近。这支持当前重构没有把这两组视频几何指标算错。
3. 对 Sintel 同一批 FP32 预测做下列固定诊断；这些是敏感性检查，不能标为已查明的原论文协议，也不能挑选最接近论文的一项宣称复现。

| GT 有效范围 / 预测深度截断 | 全部有效帧平均 AbsRel / δ1 | 先段内平均再序列等权 AbsRel / δ1 |
|---|---|---|
| GT <70 m，预测截断 70 m | 0.242908718 / 0.707896232 | 0.251133158 / 0.711898469 |
| GT <80 m，预测截断 80 m | 0.253632065 / 0.705118352 | 0.263603854 / 0.707749419 |
| GT <1000 m，预测截断 1000 m | 0.896705716 / 0.652108882 | 1.035100864 / 0.651280915 |
| GT <70 m，预测截断 1000 m | 1.038030832 / 0.706511862 | 1.216162944 / 0.710233271 |
| 所有正有限 GT，预测截断 70 m | 0.290972141 / 0.626704475 | 0.295771160 / 0.624635246 |

上述检查没有复现原论文 0.500 / 0.718。FP16/FP32、70/80 m 以及帧/序列平均均不足以解释该差异。原因尚未查明；不能把“不匹配”直接归为数据损坏，也不能把 δ1 接近当作复现成功。

## 对当前实验结论的影响

- 当前 KITTI 视频结果有意义：实际使用连续片段，整视频统一对齐对尺度随时间漂移施加了约束；逐图对齐没有同样约束。
- 整视频 SSI 的 AbsRel/δ1 仍属于几何精度指标，不能独自代替运动补偿后的时间一致性指标。
- 前次查出的 KITTI video benchmark 与 held-out 重合，影响“两份独立测试证据”的说法；它不抹去这两种取样/处理下结果的用途。
- 同样，换成视频指标不能消除真正的训练泄漏。当前 KITTI 视频 cohort 与融合头 train/dev 的物理 drive 隔离，所以本轮确认其用途成立。652 帧 Eigen 列表有 396/95 帧落入融合头 train/dev 的边界问题仍保留，不应把它与当前视频测试混称。
- 当前数据来源/取样和评分规则可作为 VDA 风格视频基准使用；其论文 DA-V2 视频对照使用 Large，而本地是 Small，也缺少已核实的完整论文视频名单。因此本轮是同一 Small 模型的代码与数据复核，不是 VDA 论文 Large 行的数值复现。

## 复核文件与命令

均位于 `outputs/audits/sintel_kitti_20261105/`：

- `papers/depth_anything_v2.pdf`、`papers/depth_anything_v2.txt`：下载的原论文及文本。
- `sintel_rescore.json`：全量 GT/预测校验和逐图评分敏感性检查。
- `sintel_video_rescore.json`：Sintel 独立视频复算。
- `kitti_rescore.json`：KITTI 独立逐图/视频复算及预测哈希。

从仓库根目录运行：

```bash
/home/hjr/miniconda3/envs/aiaa3201/bin/python Proj/outputs/audits/sintel_kitti_20261105/rescore_sintel.py
/home/hjr/miniconda3/envs/aiaa3201/bin/python Proj/outputs/audits/sintel_kitti_20261105/rescore_kitti.py --dataset sintel
/home/hjr/miniconda3/envs/aiaa3201/bin/python Proj/outputs/audits/sintel_kitti_20261105/rescore_kitti.py --dataset kitti
```

## 一手来源

- [DA-V2 原论文表 2 与附录](https://arxiv.org/html/2406.09414v2)
- [VDA 单图/视频两类评估与补充协议](https://arxiv.org/html/2501.12375v2)
- [VDA 官方补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Chen_Video_Depth_Anything_CVPR_2025_supplemental.pdf)
- [KITTI Raw 官方数据说明](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- [KITTI Depth Prediction 官方任务与指标](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [DA-V2 官方仓库相对深度评估请求 #91](https://github.com/DepthAnything/Depth-Anything-V2/issues/91)
- [DA-V2 官方仓库相对深度转换/评估实现请求 #98](https://github.com/DepthAnything/Depth-Anything-V2/issues/98)

## 新增复跑入口

```bash
/home/hjr/miniconda3/envs/aiaa3201/bin/python Proj/outputs/audits/sintel_kitti_20261105/verify_ac_source.py
```

该命令使用本次保存的全部 Sintel final / align_corners=False 预测，核对原始预测收据哈希，再直接运行保存的公开评分函数；不会重新推理或更改历史结果。其必要源码已保存于审计目录 `papers/ac-code/`，不依赖临时克隆目录。

- [AC 论文表 5 与附录 A.4](https://arxiv.org/html/2507.01634v1)
- [AC 官方仓库](https://github.com/HVision-NKU/DepthAnythingAC)
- AC 源码版本：`f1209ed2e103344b346aef69a7e30c5e602e1ccd`。
