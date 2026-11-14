# Project 3 课程交付审计与完成指南

审计日期：2026-11-12（Asia/Shanghai）。审计基线：本地提交 `c64874c`；审计前工作区干净。

唯一课程要求来源：`/home/hjr/Downloads/Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency (1).pdf`，5 页。
本次重新读取全文，并检查当前 `Proj/` 源码、被 Git 忽略的输出、训练记录、数据链接及公开 GitHub 元数据。
以下“未找到”指本次检查的仓库及本地 `Proj/outputs`，不代表用户其他位置一定没有副本。
本次只做审计及编写本文，没有启动推理、训练、下载、提交或发布。

## 结论

方向 A 已有实际实现、已完成训练、跨种子实验、公开代码和失败分析。
课程交付的缺口集中在：立体结果恢复、指定视频、应用演示、正式效率记录、CVPR 报告和结果打包。
NYU 已有一份经本次核实的历史 Depth Pro 无对齐结果，可以作为合规米制基线的现存证据；当前 UniDepthV2 视频表不能替代它。
没有依据要求为达标继续训练更多模型、增加种子、扩大数据集或完成全部额外 benchmark。

## 可直接复制的缺口表

优先级：P0 为核心实验或必交文件；P1 为支撑材料、可复现性和展示规范。二者均需处理，但 P0 应优先安排。

| 优先级 | 要求及 PDF 位置 | 当前核实状态 | 仍缺内容／完成标准 |
| --- | --- | --- | --- |
| P0 | NYU 654 张无对齐米制评价，§5.2、§6.1–6.2 | 历史结果可用：Depth Pro 原生 AbsRel 0.1004695084、RMSE 0.3922163964 m、δ1 0.9151581454；654 个 ID 与官方 split 完全一致，逐图均值复算一致。当前 UniDepthV2 已用于视频评价，但未找到它的 labeled 654 张正式结果 | 正式选择一个米制基线。复用 Depth Pro 时整理历史记录、推理设置及可复现入口；选择当前 UniDepthV2 时补跑一次完整 labeled 评价。不能把 H5 视频测试或 SSI 数字当成这项结果 |
| P0 | Middlebury 15 个 trainQ 立体对，§5.1、§6.2 | 15 个带 GT 的场景及代码齐全；实际目录名为 `trainingQ`；当前 `outputs/stereo` 不存在 | 恢复或重新生成一种方法的 15 场景结果、视差与无效区图、米制转换示例、MAE 和有效比例。评分 GT 掩码由 `>=0` 改为严格 `>0`，说明预测无效规则和汇总方式；BM、SGBM 二选一即可 |
| P0 | 指定片段前 100 帧、原帧率，§2、§6.1 | 仓库视频和本地上游副本均实测只有 70 帧、24 fps；现有 NPZ 是 35 帧、12 fps，索引 0、2、…、68 | 先解决 70／100 帧的素材冲突。获取课程确认的正确素材或对 70 帧的许可；保留原帧率与逐帧索引。当前不能将 35 帧或复制补帧版本标为达标 |
| P0 | DA-V2-S 与 VDA-S 视频对比，§5.1–5.2 | Small 权重及离线推理入口存在；当前未找到指定片段的正式预测输出或比较 MP4；当前 99 行汇总不含 VDA | 在确认后的同一帧序列上运行两种 Small 模型，使用默认 FP16；保存浮点预测、RGB／DA-V2／VDA 并排视频和设置记录，标明 relative depth |
| P0 | 方向 A 与未改进方法比较，§2、§5.3 | warp、双向一致性拒绝、置信度融合均有实现；已有 30k 训练记录和 99 行跨数据评价 | 固定最终 checkpoint，补指定片段上的原模型／方向 A 对照，说明闪烁、边界、运动伪影与局限。最终模型按 dev 决定，不能按展示片段或 test 表挑最好者；无需把静态 NYU 伪造成视频 |
| P1 | 公平显示与同设置对照，§2、§5.2、§6.2 Note | 通用视频 renderer 对不同预测分别计算整段 P1/P99；没有让改进版继承原版范围的接口 | 为 baseline 保存整段范围并让其改进版复用；相同色图、帧号、输出尺寸，固定方法与其改进的推理分辨率。DA 与 VDA 可各自采用固定范围，但不能宣称共享米制色标 |
| P1 | 消融实验，§2 Experiments | 学习率、种子、光流、ensemble 比较已有结果；报告解释了多种混杂因素 | 将已有单因素比较整理成正式消融表，写明哪些变量固定。若声称某个门控／遮挡机制有效，再补针对该机制的对照；不必把现有全部模型都重新训练。当前 `--ablations` 主要服务 `final_flow`，不能直接当成 Guided 的消融入口 |
| P0 | 推理时间及峰值显存，§6.2 | 有共享流水线 wall time 和执行路由统计；通用推理具有计时功能，但本地未找到正式 `profile.json` | 对报告使用的 UniDepthV2／历史米制基线、DA、VDA、最终方向 A 记录硬件、精度、输入尺寸、ms/frame 和峰值显存。方向 A 包含 DA、光流、warp 和融合；缓存／DGX 混合吞吐单列，不能充当单机模型延迟 |
| P0 | 一项简单应用输出，§1–2、§6.2 | 未找到课程应用实现或输出；现有 RGB GIF 与深度诊断图不构成应用 | 用提供图像／片段做深度感知背景模糊、点云等一种应用；保存 RGB、depth、应用结果及参数。选择方向 A 后仍需要应用输出；额外应用界面不必做 |
| P1 | 三个实际难例，§6.2 | 已有 Sintel／Bonn／KITTI 五段机制诊断及失败分析 | 从课程提供的 Middlebury、NYU 或指定视频中补选三个实际难例，注明来源、帧号／图像 ID，展示局部对照并解释失败。额外视频数据的分析可以保留，不能独占“提供素材中的三个难例” |
| P0 | CVPR 6–8 页报告，参考文献除外，§2 | 现有中文方法文档和 Sintel 报告可作素材；未找到 Project 3 的 `.tex`、`.bib` 或正式报告 PDF | 写成完整 CVPR 报告：摘要／引言、相关工作、方法流程图、定量表、定性图、消融、失败及结论；摘要末放公开仓库链接；覆盖推荐阅读 §4 的十篇论文，并引用实际使用的数据／代码／借鉴方法 |
| P1 | 公开仓库与 README，§2 | GitHub API 已确认仓库公开，远端已有 `Proj/README.md` 与报告索引；本地 checkpoint 位于 Git 忽略目录 | 在项目入口加入课程三部分和最终结果导航、正式命令、最终权重获取方式／版本及代表性视觉结果；补充干净克隆后的数据与权重路径说明。不要求把大权重直接塞入 Git |
| P1 | 文档与当前要求一致，§2、§6 | `GUIDE_zh.md` 仍把自录视频列为课程要求，尾部仍称 Part 3 和失败分析待做，路径与历史结果描述也已过时 | 清理这些旧说法，明确必需课程实验与额外研究的范围；确保 README 的演示、NYU 和 stereo 命令指向真实存在的输入及输出 |
| P0 | `depth_results.zip` 与最终提交，§2 | 本地未找到该 ZIP；Canvas 提交状态不在本次检查范围内 | 汇总视频、应用结果、代表图、浮点预测及说明，生成并解压检查 ZIP；另外准备正式报告 PDF，人工确认 Canvas 实际提交。arXiv 是可选项 |

## 现存证据

1. 米制历史结果：`outputs/explore_a/nyu_leakage_audit_20261031/best_nyu/metrics.json`。
   其中 `depthpro_native` 是无对齐米制基线；`best` 是当时的旧融合模型，不是当前 DA 方向 A。
   不能将两者混作当前模型的 NYU 提升。现存记录包含 654 张逐图指标；本次核对 ID 和均值，未重跑原预测。
2. 当前研究结果：`outputs/explore_a/shared_models_evaluation_v2/comparison.json`，共 99 行，
   覆盖 benchmark/Bonn、KITTI、Sintel 和 held-out test/KITTI、NYUv2、TUM。
   这 99 行包括单模型、ensemble 与 raw baseline，不是 99 个独立训练模型。
   表中主要准确率列采用 GT 对齐；部分 UniDepthV2 case 另存原生米制指标，但数据仍是视频划分，不是 labeled 654 张协议。
3. 已完成训练示例：`outputs/explore_a/mixed_h5_balanced/runs_da2_guided_30k_lr3e4/farneback-rule-unet_guided/complete.json`
   记录 30,000 步及 best dev SSI AbsRel；旁边保留 `best.pt`、`validation.jsonl`、训练日志与身份记录。
4. 已整理机制分析：`reports/SINTEL_RESULTS_ANALYSIS_20261111_zh.md`、`reports/sintel_20261111/`。
   `diagnose.py` 中的 GIF 写入对象是 RGB 帧；它们不是 RGB／depth 对照视频。
   其静态深度图采用逐图 GT gauge，适合作为有标注诊断，不可直接用于要求固定整段归一化的正式无 GT 视频演示。
5. 方向 A 实现：`src/explore_a/geometry/warp.py`、`src/explore_a/models/guided.py`、`src/explore_a/models/inputs.py`。
6. 课程适配缺口：`src/depth_project/stereo/matching.py` 的 GT 掩码；`src/depth_project/visualization/video.py` 的每文件独立范围。
7. 已核查公开仓库：<https://github.com/juruohejiarui/Homework-AIAA3201>，API 返回 `visibility=public`；
   远端 `Proj/README.md` 和 `Proj/reports/README.md` 的内容已读取。

## 完成指南

### 1. 先固定最后的实验口径

- 课程结构：Part 1 立体与逐帧相对深度，Part 2 一个单图米制基线及 VDA，Part 3 现有 DA 方向 A。
- 米制基线可继续使用历史 Depth Pro 结果，或补做当前 UniDepthV2 labeled 评价，二选一即可。
  从当前维护状态看 UniDepthV2 的代码和权重已齐全；复用 Depth Pro 则要恢复／提供其可复现推理说明。
- 最终方向 A 使用哪个 checkpoint、光流和 seed／ensemble，应从既有 dev 记录决定并记录理由。
- 先解决指定视频 70／100 帧冲突；在获得确认之前，可准备真实 70 帧流程，但不能把该项标为完成。
- 独立 NYU 图像无历史时，当前 DA 方向 A 按协议回退到当前预测。
  它是相对深度方法，不与米制基线进行“无对齐米制提升”排名。
  视频方向 A 的课程评价采用 §5.3 允许的受控定性比较和运行时间；已有额外有 GT 视频结果作为补充证据。

### 2. 补齐立体与米制基线

实际完整立体数据在 `Proj/data/middlebury/MiddEval3/trainingQ`，15 场景均具备 RGB、标定和 GT。
先修正 `matching.py` 的评分 GT 定义为 `isfinite & (disparity > 0)`，预测无效规则明确为 matcher 的无效编码，
再恢复／生成全部结果。有效预测上的 MAE 和全部有效 GT 上的预测覆盖率必须共同报告，避免仅通过丢弃难像素改善 MAE。
逐场景列两项指标；若给总数，说明采用场景等权还是像素加权。

以下命令为后续执行指南，本次未执行。均从仓库根目录运行：

```bash
P="$PWD/Proj"
PY=/home/hjr/miniconda3/envs/aiaa3201/bin/python
export PYTHONPATH="$P/src${PYTHONPATH:+:$PYTHONPATH}"

# 修正评分掩码之后运行。一种立体方法就满足要求。
"$PY" -m depth_project.stereo \
  --data "$P/data/middlebury/MiddEval3/trainingQ" \
  --method sgbm --output "$P/outputs/course/stereo/sgbm"

"$PY" -m depth_project.visualization.stereo \
  --arrays "$P/outputs/course/stereo/sgbm/Adirondack/stereo.npz" \
  --output "$P/outputs/course/stereo/sgbm/Adirondack/comparison.png"
```

若选择 UniDepthV2，当前 `comparison_nyu.py` 的独立分支可以执行无对齐评价；必须明确指定 `--model unidepthv2`，
不要用 `--model all` 把 DA 的 GT-assisted SSI 混入课程主表。
当前该分支保存逐图指标，但不保存预测数组，需要另外补保存浮点预测以便可视化和复核。

```bash
"$PY" -m explore_a.evaluation.comparison \
  --config "$P/configs/explore_a/mixed_h5_da2_lr3e4.yaml" \
  --checkpoint "$P/outputs/explore_a/mixed_h5_balanced/runs_da2_guided_30k_lr3e4/farneback-rule-unet_guided/best.pt" \
  --output "$P/outputs/course/nyu_unidepthv2" \
  --scope nyu --model unidepthv2 --nyu-root "$P/data/nyuv2"
```

上述命令中的融合 checkpoint 是当前 CLI 的快照参数；UniDepthV2 单图预测本身不读取融合输出。
最终验收：654 个官方 ID，原始 `depths` 米制 GT，480×640，规定裁剪，GT 0.001–10 m，预测截断，无尺度对齐，逐图等权。
现有 score 使用严格开区间 `(0.001,10)`，报告注明这一实现细节。

### 3. 一次完成指定视频、方向 A、可视化和效率材料

1. 对确认的源视频保留原帧率、不跳帧，保存源文件、帧索引及原 FPS。当前素材为 24 fps；
   `depth_project.data.video` 默认 10 fps，必须显式改成源 FPS，`--max-frames 100`。
   现有 70 帧视频即使设 100，也只会输出 70 帧。
2. 在同一序列运行 DA-V2-Small、VDA-Small 离线 FP16 和选定方向 A；保存全部浮点输出。
3. 方向 A 的对照必须使用它实际消费的原始 DA 分支及相同画布／分辨率。
   现有 `da2_fp32` 与融合前端同时存在精度、尺寸规则、插值差异，不能单独用它量化融合模块收益。
4. 改进视频复用其 baseline 整段 P1/P99，保存实际数值；VDA 可以另用自己的整段范围，所有图保持同一色图。
   导出 `RGB | DA raw | VDA | DA + A` 或两组清楚的并排视频，并保留源帧号。
5. 对每个报告方法单独计时。建议加载后预热、同步 GPU、重复测量，记录均值和波动；
   这些是测量建议，PDF 本身未规定预热次数。注明是否包含预处理、CPU 传输和光流。
   VDA 报整片耗时／帧数；方向 A 报全部必要组件的总耗时，并说明首帧成本和历史复用策略。
   记录峰值 allocated 显存与硬件／软件版本；混合缓存吞吐另列。
6. 现有 `inference.engine.predict` 已同步 GPU 并记录峰值显存，但没有预热和硬件字段；
   可以扩展其记录，不能直接把多模型共享 case 的 `wall_s` 当成每个方法的独立耗时。

### 4. 整理消融、难例和应用

- 现成可用的消融：学习率、FB／RAFT、单 seed／ensemble；一次只讨论相应变化，并报告额外成本。
  若要论证置信度设计的独立作用，可增加固定 checkpoint 的固定权重融合对照；若要论证遮挡拒绝，
  可增加保持边界检查但去掉 FB 拒绝的对照。实现时需改变真实推理路径，不能只改图标签。
  这两项是机制分析建议，不是课程规定的固定消融数量。
- 从 NYU／Middlebury／指定视频挑三个实际难例，保留 RGB、预测及局部放大；有 GT 的场景可加误差图。
  对指定视频只能讨论可观察的闪烁／拖影／边界，不据此声称米制更准确。
- 应用建议选深度感知背景模糊：在一张提供图像上固定焦点和模糊强度，输出 RGB、depth 与效果图。
  若扩展到原方法与改进方法应用对照，应用参数必须固定。方向 A 的一项简单应用交付本身不强制再发明第二个改进。

### 5. 写作、公开复现和提交验收

报告建议安排约 1 页引言／相关工作、1–1.5 页方法、3–4 页实验与分析、约半页结论，
根据实际图表调整，正文总计 6–8 页，参考文献另计。
核心表应包括立体结果、NYU 米制结果、受控消融和效率；视频放连续帧及局部图。
推荐阅读 §4 对应 [1]–[10] 十篇；文末 [11]–[12] 等实际使用的数据来源也要引用。
方法图解释两帧 RGB、单图预测、光流、warp、拒绝和融合，以及 GT 只用于训练／评价。

公开 README 加入“课程结果”入口、数据准备、依赖、最终权重下载／身份、训练与评价命令、代表图、
正式数值及已知限制；修正旧的自录视频要求。最终头的 `best.pt` 目前仅在被忽略的本地目录，
应提供可访问的 release／下载链接或清晰的重建方式。发布动作需用户另行授权。

建议 ZIP 结构（这是交付组织建议，不是课程指定目录名）：

```text
depth_results/
  README.md
  video/          # 正式并排视频、帧索引、归一化与设置
  application/    # RGB、depth、应用输出、参数
  stereo/         # 代表图、15 场景指标
  nyu/            # 正式米制表、代表图与协议
  ablations/      # 受控比较表
  failures/       # 三个课程素材难例
  efficiency/     # 硬件、尺寸、精度、耗时、显存
  predictions/    # 选定交付范围的浮点预测
```

PDF 要求保存浮点预测，但没有规定必须把所有训练缓存和所有额外 benchmark 预测塞进 ZIP。
保留正式实验的可追溯预测，ZIP 中放与展示对应的预测和说明即可明确交付范围。
打包后在新目录解压，检查媒体可播放、JSON／NPY 可读、相对路径可用，报告中的链接和指标一致。
最终提交报告 PDF 和 `depth_results.zip`；代码通过公开 GitHub 提供，不向 Canvas 散交原始代码文件。

## 本次未列为必补的额外实验

arXiv、更多训练 seed、完整三模型矩阵、TUM validation 全量评价、更多 KITTI／Sintel／Bonn 实验、
光流数值 benchmark、深度边缘 P/R、相机位姿与复杂应用界面，均不是本次课程达标所必需。
现有扩展成果可用来增强分析，但不能替代上表核心课程交付。
