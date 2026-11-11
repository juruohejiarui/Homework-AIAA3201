from pathlib import Path
import json, shutil, hashlib
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PROJ=next(p for p in ROOT.parents if (p/'src/explore_a').is_dir())
EVAL=PROJ/'outputs/explore_a/shared_models_evaluation_v2'
for name in ['comparison.json','comparison.md','session.json']:
 shutil.copy2(EVAL/name,ROOT/('evaluated_'+name))
shutil.copy2(PROJ/'maintenance/evaluate_models.json',ROOT/'evaluated_registry.json')
rows=json.loads((EVAL/'comparison.json').read_text())
groups=defaultdict(dict)
for r in rows: groups[r['scope'],r['dataset']][r['model']]=r
records=[json.loads(p.read_text()) for p in EVAL.glob('*/*/*/*/metrics.json')]
cases=defaultdict(dict)
for r in records: cases[r['scope'],r['dataset'],r['sequence'],r['camera'],r['case']][r['model']]=r
fb='farneback_lr3e-4_mean3';ra='raft_lr3e-4_mean3';mix='farneback_raft_lr3e-4_mean6'
def gain(a,b):return 100*(a-b)/a
def table(headers,data):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in data])
stats=[]
for (scope,ds),d in groups.items():
 selected=[v for k,v in cases.items() if k[:2]==(scope,ds)]
 pairs=[(x['raw_canvas'],x[fb]) for x in selected if fb in x]
 # Resample whole case units. Sintel image scores preserve frame weighting.
 b=np.array([x['metrics']['absrel_img'] for x,y in pairs]);f=np.array([y['metrics']['absrel_img'] for x,y in pairs])
 w=np.array([x['metric_counts']['absrel_img'] if ds=='sintel' else 1 for x,y in pairs])
 rng=np.random.default_rng(20261111);ix=rng.integers(0,len(b),(10000,len(b)))
 gains=100*np.sum(w[ix]*(b[ix]-f[ix]),1)/np.sum(w[ix]*b[ix],1)
 item={'scope':scope,'dataset':ds,'case_units':len(b),'improved':int((f<b).sum()),'worse':int((f>b).sum()),'bootstrap_95':np.percentile(gains,[2.5,97.5]).tolist()}
 stats.append(item)
(ROOT/'case_statistics.json').write_text(json.dumps(stats,indent=2))
sintel=[]
for k,d in cases.items():
 if k[:2]!=('benchmark','sintel'):continue
 b=d['raw_canvas'];f=d[fb]
 sintel.append({'sequence':k[2],'frames':b['frames'],'raw':b['metrics']['absrel_img'],'fb':f['metrics']['absrel_img'],
                'gain':gain(b['metrics']['absrel_img'],f['metrics']['absrel_img']),
                'weighted_delta':(f['metrics']['absrel_img']-b['metrics']['absrel_img'])*b['metric_counts']['absrel_img']/1064})
sintel.sort(key=lambda x:x['gain'])
(ROOT/'sintel_sequences.json').write_text(json.dumps(sintel,indent=2))
fig,ax=plt.subplots(figsize=(10,7))
ax.barh([x['sequence'] for x in sintel],[x['gain'] for x in sintel],color=['#278a66' if x['gain']>0 else '#cc624c' for x in sintel])
ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Image AbsRel reduction vs raw_canvas (%)');ax.set_title('Sintel: Farneback lr3e-4 mean3, all 23 sequences')
fig.tight_layout();fig.savefig(ROOT/'sintel_sequence_gains.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,metric,title in zip(axs,['absrel_img','absrel_video'],['Per-image alignment','Whole-sequence alignment']):
 x=np.arange(len(groups)); labels=[]
 for j,model in enumerate([fb,ra,mix]):
  vals=[]
  for (scope,ds),d in groups.items():vals.append(gain(d['raw_canvas'][metric],d[model][metric]))
  ax.bar(x+(j-1)*.25,vals,.25,label=['FB mean3','RAFT mean3','Mixed mean6'][j])
 labels=[f'{s}\n{d}' for s,d in groups];ax.set_xticks(x,labels,fontsize=8);ax.axhline(0,color='black',lw=.7);ax.set_title(title);ax.set_ylabel('AbsRel reduction (%)')
axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'all_dataset_gains.png',dpi=160);plt.close(fig)

summary=json.loads((ROOT/'sample_summary.json').read_text())
doc='''# 当前训练结果分析与 Sintel 诊断

日期：2026-11-11。对象：当前 `shared_models_evaluation_v2` 中全部 99 行汇总结果，包含 11 个注册单模型/冻结参考、4 个 ensemble 和自动 raw baseline；其中 benchmark 17 行/数据集，test 16 行/数据集。

## 结论

1. lr3e-4 在真实数据集上的小幅改善具有跨种子、跨片段的一致性；Sintel 的收益受场景影响明显，有益片段与退化片段相互抵消。不能把整体接近零理解为所有片段都学不到。
2. Sintel 的主要限制是：融合头只在两帧 DA2 预测之间选择，难以修复两帧共有的结构性深度误差；困难运动片段还存在历史可用区域减少和较高融合风险。真实数据训练与 Sintel 的域差异、训练目标与正式评分的差异共同影响收益。
3. 抽样门控没有接近零：可用区域上的历史权重约 0.39–0.55，排除了“完全没用历史”这个简单解释。部分片段仍有明显 oracle 余量，不能认定当前模型已达到可实现上限，更不能直接据此证明欠拟合。
4. RAFT 未形成稳定优势；增加 seed 的等权 ensemble 收益很小。当前结果更支持 Farnebäck lr3e-4 作为成本较低的候选。最终选择应在 dev 上完成，test/benchmark 用于报告。

## 1. 数据、口径与覆盖

- 本报告读取正式结果，不覆盖它们。比较表、session 和 registry 的快照均保存在本报告旁的审计目录中。
- 六组范围：benchmark/Bonn、KITTI、Sintel；held-out test/KITTI、NYUv2、TUM。没有本次 TUM validation 的结果，也没有 lr5e-4、旧 RAFT lr1e-4、final_flow 的本次结果；不能据此评价这些实验。
- Sintel：training/final，23 段、1064 帧。逐图按 AC：GT 截断 72 m，inverse-depth 无约束 LS，预测深度截断 80 m，depth AbsRel/δ1，按帧平均。视频指标保留 GT<70 m、整段统一拟合、序列宏平均。两列的 mask、范围、对齐与聚合不同，不能把其差直接解释为闪烁强弱。
- 其他数据集按原有协议。raw_canvas 是与融合头相同输入预测、经过融合画布的基线；raw_native 是相同前端的原图输出。DA2 FP32 同时改变精度、输入尺寸规则和插值设置，不能把其差距全归于 FP16。
- 本次新增模型、ensemble 与 raw_canvas 的 case 和评分帧数一致。Bonn 26 段共 12541 输入帧，其中 12521 帧有有效逐图评分；Sintel 1064 帧全部参与 AC。
- KITTI benchmark 是 held-out test 的重叠子集；二者不是独立的两次泛化验证。训练只包含 KITTI、TUM、NYUv2，Sintel 与 Bonn 不参与融合头训练。RAFT 冻结权重本身包含 Sintel 等光流训练数据，不能把整条 RAFT 流程称为完全未接触 Sintel。
- 全部主表指标经过 GT 对齐，是相对几何准确性。UniDepthV2 的这些优势不等同于未对齐的米制误差优势。

## 2. 所有实验的总体趋势

下表均相对 raw_canvas，正数为 AbsRel 下降。每格为“逐图 / 整段”的相对改善百分比。

'''
data=[]
for key,d in groups.items():
 data.append([f'{key[0]}/{key[1]}']+[f'{gain(d["raw_canvas"]["absrel_img"],d[m]["absrel_img"]):+.3f}% / {gain(d["raw_canvas"]["absrel_video"],d[m]["absrel_video"]):+.3f}%' for m in ['unet_guided_mean',fb,ra,mix]])
doc+=table(['范围','FB lr1e-4 ×3','FB lr3e-4 ×3','RAFT lr3e-4 ×3','混合 lr3e-4 ×6'],data)
doc+='''

![全部范围的增益](outputs/audits/sintel_interpretation_20261111/all_dataset_gains.png)

### 学习率与随机种子

Farnebäck lr3e-4 的三种子组合相比 lr1e-4 三种子组合，在五个真实数据范围的逐图 AbsRel 额外改善约 0.15%–0.47%。同一数据集上，三个 lr3e-4 种子通常非常接近；KITTI test 三个单模型为 0.087513、0.087504、0.087518，均优于三个 lr1e-4 单模型。这是优化改善的证据，不能单独区分欠拟合、训练长度不足或损失/架构瓶颈。Sintel 上 lr3e-4 并未延续该趋势。

### 光流与 ensemble

RAFT lr3e-4 三种子组合在六组范围的逐图 AbsRel 均弱于 Farnebäck 三种子组合。其历史可用区域有时更大，但“更多像素通过几何检查”不等于历史深度更准确。六模型组合在 KITTI 的整段 AbsRel 上更好，却牺牲了相对 FB×3 的逐图误差；Bonn、NYUv2 的部分指标仅有极小收益，TUM 没有优势。种子平均主要降低训练随机性，不会创造新的深度信息。

### 冻结参考

UniDepthV2 在五组真实数据的四项主指标上均明显领先当前融合头。Sintel 上则存在指标权衡：其整段 AbsRel 最低，但 δ1 低于 DA2，不能说全部指标最好。DA2 FP32 在 Sintel 的逐图 AbsRel 为 0.239514，优于 raw_canvas 的 0.241631，也优于所有融合结果；这说明前端配置造成的差异已经大于当前融合收益。KITTI test 上 DA2 FP32 的逐图误差和 δ1 也好于融合头，而融合头的整段 AbsRel 较低。NYUv2、TUM 和 Bonn 上当前 FB 融合可小幅优于 DA2 FP32，但仍明显落后于 UniDepthV2。

### 片段一致性

以下以 FB lr3e-4×3 相对 raw_canvas 的逐图 AbsRel 为准。bootstrap 以完整 case 为抽样单元；Sintel 保留帧数权重。它仅描述本批片段重采样的波动，不代表新的数据集或训练随机性；KITTI test 的相机分支也不是完全独立样本。

'''
doc+=table(['范围','改善 / case 数','退化','增益的 case bootstrap 95% 区间'],[[f'{x["scope"]}/{x["dataset"]}',f'{x["improved"]}/{x["case_units"]}',x['worse'],f'{x["bootstrap_95"][0]:+.2f}% 至 {x["bootstrap_95"][1]:+.2f}%'] for x in stats])
doc+='''

## 3. Sintel 的“接近零提升”由什么构成

FB lr3e-4×3 在 23 段中有 14 段改善、9 段退化。最大改善 market_6 为 +3.08%，最大退化 ambush_2 为 −2.54%；总逐图 AbsRel 却从 0.241631 变为 0.241693（−0.026%）。Sintel 的高基础误差也说明它不是已经“太容易、没有提升空间”。

![逐段增益](outputs/audits/sintel_interpretation_20261111/sintel_sequence_gains.png)

'''
doc+=table(['片段','帧数','raw AbsRel Img','FB×3 AbsRel Img','相对改善','对总 AbsRel 的变化贡献'],[[x['sequence'],x['frames'],f'{x["raw"]:.6f}',f'{x["fb"]:.6f}',f'{x["gain"]:+.3f}%',f'{x["weighted_delta"]:+.7f}'] for x in sintel])
doc+='''

### 对齐方式会改变结论，不能据此删掉不利指标

正式整段评分为每个预测分支独立拟合 inverse-depth LS，其优化目标不是 depth AbsRel。重新拟合可能降低 inverse 的平方误差，却增大 depth AbsRel；因此固定 baseline 对齐后的正收益，与重新拟合后的退化可以同时出现。

'''
d=groups['benchmark','sintel']
doc+=table(['Sintel 模型','逐图 AbsRel（AC）','整段 AbsRel（独立拟合）','整段 AbsRel（baseline 固定拟合）'],[[m,f'{d[m]["absrel_img"]:.6f}',f'{d[m]["absrel_video"]:.6f}',f'{d[m]["absrel_baseline_fixed_video"]:.6f}'] for m in ['raw_canvas',fb,ra,mix]])
doc+='''

FB×3 的 baseline-fixed video AbsRel 降幅约 0.66%，但正式 video AbsRel 退化约 0.17%。这表明融合确实改变了预测，不能据此宣称没有学习；正式 AC 逐图和独立拟合视频仍无提升，也不能把问题全部解释为指标“掩盖”了成功。ordinal_accuracy 从 0.916227 到 0.915858（FB×3），RAFT×3 为 0.917320：排序也没有呈现全面一致的收益。这里的 ordinal 是传感器/渲染 GT 派生的随机深度对，不是 DA-2K 官方标注。

## 4. 视频抽样、可视化与 oracle

本次新增 100 帧诊断：三个 Sintel 片段（最差 ambush_2、最好 market_6、基础误差高但历史可用性高的 alley_1），以及一个 Bonn、一个 KITTI 片段。选择旨在解释机制，刻意包含不同表现，不把这五段当成无偏总体抽样。每段取中间连续 20 帧，ambush_2 因长度仅 21 帧而取索引 1–20。前一帧保持真实相邻帧，不人为重置历史。

仅对两个 lr3e-4 seed2026 单模型做详细前向，不能将其诊断数值直接当作六模型均值的行为。本机 online 推理、禁用 DGX 与缓存，仅改变传输/执行路径，不修改模型、输入和推理数值规则。诊断保存于 `reports/sintel_20261111/`，每帧结果见 `sample_diagnostics.json`，汇总见 `sample_summary.json`，脚本为 `diagnose.py`。

下表使用同一帧 raw_canvas 对 GT 的固定正 scale/shift，将 raw、fused 和 oracle 一起转换到深度后评分。Sintel GT 按 AC 截断 72 m，但该诊断不是正式 AC：不重新拟合每个分支，使用 FP64 拟合；其他数据遵循原有效 GT mask。因此该表只能作机制比较，正式排名仍见主表。

oracle 定义：在几何允许使用历史的位置，GT 告知当前值与对齐后历史值之间的最佳连续混合比例；不可用位置保留当前值。GT 不进入正常推理。它是该固定 gauge、该两帧凸混合族的乐观上界，并不是正式 AC 分数的上界。历史有效性由还原到原图后的 availability>0.999 定义，避免边界插值污染。

'''
data=[]
for x in summary:
 for model,z in x['models'].items():
  data.append([x['sequence'],f'{x["start"]}–{x["end_inclusive"]}', 'FB' if model.startswith('farneback') else 'RAFT',
   f'{z["availability"]*100:.1f}%',f'{z["weight_available"]:.3f}',f'{z["current_history_inverse_error_corr"]:.3f}',
   f'{x["raw_shared_gauge_absrel"]:.5f}',f'{z["shared_gauge_absrel"]:.5f}',f'{z["oracle_shared_gauge_absrel"]:.5f}',
   f'{gain(x["raw_shared_gauge_absrel"],z["oracle_shared_gauge_absrel"]):.1f}%'])
doc+=table(['片段','零基帧索引','模型','历史可用率','可用区历史权重','两帧逆深度误差相关','Raw','Fused','Oracle','Oracle 余量'],data)
doc+='''

可用区权重不是全图权重，也不是历史信息正确率。误差相关只在可用且有 GT 的像素上计算，不能替代光流 EPE；KITTI 同样高度相关，因此“相关高”本身不能独立解释 Sintel。

### ambush_2：困难运动、模糊、遮挡与远背景

连续图像里有近距离人物/武器运动、明显运动模糊与大面积浅色远背景。FB、RAFT 可用率都约 56%–57%，远低于抽样 KITTI；其中约 38.4% 的正 GT 像素 >=72 m，会进入 AC 的远深度截断区。人物与背景边界的误差改变呈正负混杂，融合不能简单统一纠正。FB 的同 gauge 误差从 0.6360 降为 0.6207，oracle 为 0.5204：存在可利用信息，但学到的门控只利用了一部分。完整片段、三种子 ensemble 的正式 AC 反而退化，进一步说明单段样本、单模型、对齐协议之间需要分别报告。

![ambush_2 预测对照](outputs/audits/sintel_interpretation_20261111/ambush_2.png)

[ambush_2 连续 RGB 动图](outputs/audits/sintel_interpretation_20261111/ambush_2.gif)

### market_6：有互补信息时，Sintel 也能改善

场景包含车轮、木梁、人物和细结构遮挡，连续画面视角有明显变化。可用率 FB 72.6%、RAFT 80.8%；同 gauge 下 FB 从 0.3482 降为 0.3344，优于 RAFT 的 0.3452。完整片段 FB×3 的正式 AC 改善 3.08%，说明合成域并非一律无法使用历史。RAFT 更高可用率却没有更好误差，排除了“只要换更强光流便能解决”的简单解释。

![market_6 预测对照](outputs/audits/sintel_interpretation_20261111/market_6.png)

[market_6 连续 RGB 动图](outputs/audits/sintel_interpretation_20261111/market_6.gif)

### alley_1：历史可用，但两帧共享深度结构错误

可用率约 95%–96%，历史权重约 0.46–0.55；两帧误差相关约 0.975–0.978，FB 实际修改幅度仅为当前预测 P90−P10 尺度的约 0.62%。图中人物和背景表面的大尺度深度关系在 raw、FB、RAFT 间几乎相同，历史平均不能产生缺失的正确表面结构。FB 的 raw 0.6742、fused 0.6690、oracle 0.6332：即使知道逐像素最佳混合，仍残留较大误差；其 oracle 相对余量仅约 6.1%。这段说明光流可用性不足不是唯一原因。

![alley_1 预测对照](outputs/audits/sintel_interpretation_20261111/alley_1.png)

[alley_1 连续 RGB 动图](outputs/audits/sintel_interpretation_20261111/alley_1.gif)

### 与真实视频对照

Bonn static 的白墙、门与地面几何较简单，KITTI 片段主要为庭院道路和建筑。Bonn 的 FB 可用率 86.8%，RAFT 近 100%；KITTI 为 93.9% / 99.7%。但 Bonn 这 20 帧中 FB 的同 gauge 误差微升，RAFT 改善；完整 500 帧的正式 FB×3 仍改善 1.77%。片段内门控收益并不均匀，不能用单个可视化样本代替完整视频的结论。KITTI 抽样 FB 同 gauge 从 0.0920 降至 0.0875，oracle 0.0779，显示模型利用了部分历史信息。

![Bonn 预测对照](outputs/audits/sintel_interpretation_20261111/rgbd_bonn_static.png)

[Bonn 连续 RGB 动图](outputs/audits/sintel_interpretation_20261111/rgbd_bonn_static.gif)

![KITTI 预测对照](outputs/audits/sintel_interpretation_20261111/2011_09_26_drive_0079_sync.png)

[KITTI 连续 RGB 动图](outputs/audits/sintel_interpretation_20261111/2011_09_26_drive_0079_sync.gif)

图例：每张图抽三个时刻，列依次为 RGB、GT、raw、FB、RAFT、FB−raw 的逐像素 AbsRel 差、FB 历史权重。深度图在同一行共享 GT P5–P95 色阶，深紫近、红远，黑色为无有效 GT；误差差图 blue=改善、green≈0、red=退化，范围 ±0.02；权重 blue=0、red=1。KITTI 图只显示有激光 GT 的稀疏区域，不能把黑色视为模型没输出。

## 5. 机制判断与证据强弱

### 已由代码或结果确认

- **模型能力受凸混合限制。** 当前 `GuidedUNet` 仅输出 blend/reset 两个门控；最终值为当前预测与光流 warp、仿射对齐历史预测的凸组合，没有额外深度修正项，没有独立时序深度 backbone。两者同向犯错时，即使门控完美，也不能超出该区间恢复正确结构。
- **两帧而非长时序。** 历史输入是前一帧原始 DA2 深度，不递归使用融合结果；每帧只有前一帧信息。换 seed、增加 ensemble 不会增加时间跨度。
- **门控没有关闭。** 五段可用区平均权重均远大于零。难点是挑选历史是否正确，以及历史本身是否包含新信息。
- **训练分布和评分分布不同。** 训练数据为真实 KITTI/TUM/NYUv2；Bonn 是较接近真实 RGB-D 的迁移，Sintel 是电影式渲染与复杂非刚性动态。不能把 DA2 自身说成没见过合成数据：DA2 教师使用合成标注，学生经大规模真实伪标签学习，域偏差主要针对当前融合头与具体渲染/运动模式。
- **训练损失并不等于 AC AbsRel。** 相对深度主损失在 baseline 固定 GT gauge 下归一化 inverse-depth，外加 blend/reset BCE；正式 Sintel 评分重新拟合后在 depth 空间测量。这种差异与固定 gauge 改善、正式指标不改善的结果一致，但尚未通过损失消融证明因果。
- **当前 GuidedUNet 没有额外 temporal loss。** `two_frame.supervised()` 对非 FinalFlow 在计算普通 fusion_loss 后直接返回。即使配置有 temporal_weight，不能据此说当前这些 guided 模型优化了专门的双向时序损失；训练日志的 temporal_loss=0 与此一致。这也使“video 指标稍好”不能替代直接闪烁评估。
- **画布降低 Sintel 细节分辨率。** 原图 436×1024 进入 384×512 融合画布时有效图像为 218×512，上下 padding 共 166 行，占画布高度约 43.2%。细结构与运动边界容易损失；但 raw_canvas 与 raw_native 的逐图差仅约 0.000341，且 KITTI 也经缩放，所以画布不是目前全部问题的充分解释。

### 当前最合理的综合解释

Sintel 的误差包含两类：两帧均犯的语义/结构错误，以及运动、遮挡、模糊导致的历史传递错误。当前架构只能降低两帧间可互补的那部分误差；在简单真实几何上收益较一致，在 Sintel 上好坏像素和好坏片段更容易抵消。训练域与损失差异进一步让门控在困难片段选择不足。抽样 oracle 证明“仍有门控学习空间”和“部分结构误差超出凸混合能力”同时存在，单纯延长训练或换更大光流未必解决两者。

Sintel 官方明确包含大运动、镜面反射、运动/散焦模糊和大气效果；final pass 包含完整渲染效果，与本次 ambush_2 的视觉现象一致。[MPI Sintel 官方介绍](https://sintel.is.tue.mpg.de/)，[原始论文](https://files.is.tue.mpg.de/black/papers/ButlerECCV2012-corrected.pdf)。DA2 训练数据与教师/学生机制参见[原论文](https://arxiv.org/abs/2406.09414)。AC 口径依据[Depth Anything at Any Condition](https://arxiv.org/abs/2507.01634)及[官方代码](https://github.com/HVision-NKU/DepthAnythingAC)。本报告不把这些外部资料当作当前模型具体退化的因果实验证据。

## 6. 对后续工作的建议

1. 保留当前正式口径；同时在报告中展示 baseline-fixed 诊断，避免在重新拟合之后看不到预测实际变化，也避免只选有利指标。
2. 当前低成本候选优先考察 FB lr3e-4 单模型/三种子平均；六模型 ensemble 没有足够一致的收益支撑其推理开销。用 dev 确定模型，不按本次 test 反复挑 seed。
3. 若下一步要专门解释“优化不足还是输入不足”，对完整 Sintel 23 段做相同 mask、同 gauge 的两帧 oracle，按动态/遮挡/边界区域分解。抽样已证明存在余量，但不能把抽样 oracle 当作完整测试集上限。
4. 若目标是提升 Sintel 泛化，优先在独立训练数据中加入相应动态与渲染条件，或研究训练损失与目标评价的一致性；不要将当前 Sintel benchmark 直接加入训练后继续宣称独立泛化。
5. 若 oracle 本身改善有限，应考虑超出凸混合的修正头或更丰富时序信息；这是架构/协议选择，需要与当前两帧约束一起明确。不能仅以 RAFT 更强为由继续增加光流计算。
6. 若目标是视频稳定性，需要增加遮挡/运动补偿的时序误差或一致性可视化；目前整段 SSI 指标没有直接测量闪烁。本报告没有训练新模型、修改正式评分或启动完整评估。

## 附录：当前全部比较结果

以下为审计时正式 `comparison.md` 的完整快照。保留所有行和原始精度，便于与上述分析核对。

'''
doc+=(EVAL/'comparison.md').read_text().replace('# Shared-input model comparison','### 完整主表',1)
path=PROJ/'reports/SINTEL_RESULTS_ANALYSIS_20261111_zh.md'
doc=doc.replace('](outputs/audits/sintel_interpretation_20261111/','](sintel_20261111/')
path.write_text(doc)
sources=['src/explore_a/models/guided.py','src/explore_a/models/inputs.py',
         'src/explore_a/training/relative.py','src/explore_a/training/two_frame.py',
         'src/explore_a/evaluation/session.py','src/explore_a/evaluation/sintel_ac.py',
         'src/explore_a/frontend.py']
provenance={
 'source_sha256':{p:hashlib.sha256((PROJ/p).read_bytes()).hexdigest() for p in sources},
 'result_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                  ['evaluated_comparison.json','evaluated_session.json','sample_diagnostics.json']},
 'diagnostic_models':['farneback_unet_guided_lr3e-4','raft_unet_guided_lr3e-4'],
 'diagnostic_execution':'local online, CUDA, no DGX or cache, same frozen/head precision settings',
 'clip_selection':'three Sintel contrasting cases plus Bonn static and KITTI drive0079; center 20 consecutive frames; not a random sample'}
(ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2))
print(path)
print(json.dumps(stats,ensure_ascii=False,indent=2))
