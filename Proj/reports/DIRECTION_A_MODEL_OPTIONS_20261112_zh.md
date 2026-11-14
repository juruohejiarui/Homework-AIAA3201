# 方向 A 新模型：严格两帧条件下的论文与方案讨论

日期：2026-11-12。用户已确认继续严格两帧。本记录只做研究与设计，不授权下载、
修改当前实验协议或启动训练。现有权重、配置、划分、缓存及结果均保留。

## 判断

建议把新训练的研究问题设为：**预训练双视图特征交互，能否比两张单图预测的输出融合
更有效地利用另一帧，改善几何准确性和视频稳定性？**

如果接受较大重构，优先研究 DA3-Small 的双视图轻量微调，加入方向 A 的显式光流、
FB 拒绝和置信度融合。工程成本更低的备选是 DA2-Small 预训练编码器/解码器内部的
双帧特征适配。后者借鉴 Mono-ViFI 的特征对齐，不复现其三帧自监督系统。

现有 guided 正式结果中，真实数据逐图改善约 1%–2.4%，Sintel 整体收益接近零；
这些是既有测试结果的机制诊断，不作为新方案调参集。
现有 flow_refine/final_flow 已包含特征对齐、残差和部分上下文机制，不能把
“增加 GRU/残差/FiLM”单独视为新贡献；重要区别是引入预训练的深层几何特征和解码能力。
guided 的两帧输出凸融合确有范围约束，但这不能概括为所有现有 refine 模型都受相同约束。

## 论文与开源情况

| 论文/模型 | 可借鉴内容 | 严格两帧下的判断 |
| --- | --- | --- |
| Depth Anything 3，2025 技术报告 | 任意视图深度与相机几何联合预测；有 DA3-Small 预训练权重 | 支持 N=2 的自然输入；大重构首选候选。Small 约 80M 参数，比 DA2-S/VDA-S 更重 |
| Mono-ViFI，ECCV 2024 | MAFA 运动感知特征对齐、OAFF 遮挡感知特征融合；训练代码公开 | 原版三帧；借其结构，改为当前/前一帧及现有流与 FB mask，不照搬三帧训练和 VFI 系统 |
| NVDS / NVDS+，ICCV 2023 / TPAMI 2024 | 对现有单图深度做学习式视频稳定；公开 stabilizer 权重 | 适合研究预训练稳定器，但已有多帧设置、DPT/MiDaS 输入约定需要适配；不是直接可用的两帧替代 |
| Video Depth Anything / oVDA，CVPR 2025 / 3DV 2026 | 冻结单图编码器、预训练时空解码器；clip SSI、TGM、SaSCon | 值得借监督方式；原版 VDA 窗口和 oVDA 8/16 帧缓存不符合严格两帧，不能直接截断并声称完整复现 |
| DyFN，CVPR 2026 | 冻结预训练编码器/解码器，仅训练特征统计调制模块 | 原版 ConvGRU 汇总历史，公开主模型约 320M；仅借“特征统计影响尺度”的机制，双帧改造需要单独验证 |
| FlashDepth，ICCV 2025 | DPT 内时序模块、Small 预训练权重与完整训练入口 | 原版有流式状态，不是独立帧对系统；严格两帧下不是主选，Mamba 环境也增加集成工作 |

直接来源：

- DA3：[论文](https://arxiv.org/abs/2511.10647)、[官方代码及模型表](https://github.com/ByteDance-Seed/Depth-Anything-3)。
- Mono-ViFI：[论文](https://arxiv.org/abs/2407.14126)、[官方代码](https://github.com/LiuJF1226/Mono-ViFI)。
- NVDS：[官方代码和权重](https://github.com/RaymondWang987/NVDS)。
- VDA：[论文](https://arxiv.org/abs/2501.12375)、[官方代码](https://github.com/DepthAnything/Video-Depth-Anything)。
- oVDA：[论文](https://arxiv.org/abs/2510.09182)、[官方代码](https://github.com/FriedFeid/OnlineVideoDepthAnything)。
- DyFN：[论文](https://arxiv.org/abs/2605.25308)、[官方代码](https://github.com/shawLyu/Streaming_DyFN)。
- FlashDepth：[论文](https://arxiv.org/abs/2504.07093)、[官方训练代码](https://github.com/Eyeline-Labs/FlashDepth)。

另外，CODD 的官方实现以双目视差、运动和融合模块组成系统。其权重不能直接作为
DA 相对逆深度融合头初始化；可借 reset/可靠性机制，不优先移植整套系统。
[官方来源](https://github.com/facebookresearch/CODD)。

上述公开代码/权重状态来自网页检查，没有在本机安装或运行这些新模型。
论文各自的 benchmark 数值不直接与本项目 evaluate_v2 主表比较。

## 主候选：DA3-Small 双视图轻量适配

### 输入和结构

1. 每个样本只有 `(I_prev, I_current)`，不跨帧对保留 KV、ConvGRU 或预测状态。
2. 以预训练 DA3-Small 联合处理两张图；两视图在深度输出前交互，而不是独立预测后仅做选择。
3. 冻结大部分原模型；在跨视图 attention 中加入 LoRA，或在 DualDPT 特征处加入小 adapter。
   先选一类主要适配位置，避免同时加入多个难归因的新模块。
4. 保留冻结光流、FB 几何掩码、历史深度 warp；将历史候选及对齐特征送入置信度融合模块。
   不把预训练 depth_conf 自动视作校准过的融合概率，可靠性仍需要训练/验证。
5. 若希望进一步消除最终凸融合范围约束，可在预训练解码器内进行置信度加权的特征融合，
   或由适配后的解码器产生纠偏；避免仅在两个固定深度值之间选权重。

DA3 本身的联合预测已经提供不同于 DA2 单帧输出的模型族；这部分变化必须有单独基线。
不能把“换了更强预训练模型”的收益全部归因于光流融合。

### 重要实现约束

- DA3-Small 的深度输出不能按 DA2 原始相对逆深度直接解释，也不能默认是米制。
  需要单独声明表示；如内部用逆深度，则做明确的正深度取倒数转换。
- 两帧 joint prediction 不代表整段视频天然无尺度漂移，各帧对仍可能选择不同的尺度。
- 2D 光流只给像素对应，不能自动修正同一点在相机运动下的 Z 变化。
  如采用位姿几何重投影，需要处理动态物体和位姿误差；否则监督时匹配 GT 的深度变化，
  不能盲目要求 warped previous depth 与 current depth 相等。
- 官方高层 API 的 `forward` 带 `torch.inference_mode()`，内部还使用 `no_grad()`。
  训练不能直接调用这个封装；需要对底层网络建立保留梯度的训练入口。
  [API 源码](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py)
- 公开权重和推理代码不等于可直接复用本项目的训练脚本。需要明确训练参数组、
  LoRA/adapter 保存格式、双帧数据接口和 evaluate_v2 的新模型适配器。
- 训练参数少不等于只承担小头的显存成本；训练 LoRA 或 decoder adapter 时仍要保留相应激活。
  AMP、分辨率、microbatch 和梯度检查点应由受影响的短程显存/吞吐测量决定。
- 当前 hybrid RPC 返回冻结推理结果，不传回可反向传播的计算图。新模型可在本机或 DGX
  启动完整的训练前向，但不能让 trainable LoRA 经现有无梯度 RPC 执行后再在主机训练。
  仍然冻结的光流等组件可继续复用相应调度；新特征和旧深度 NPZ 的身份不能混用。

## 低工程成本备选：DA2-Small 内部双帧特征融合

```text
I_prev ── shared pretrained encoder ── F_prev ── feature warp ─┐
                                                            ├─ masked confidence adapter
I_current ─ shared pretrained encoder ── F_current ────────────┘
             optical flow + FB mask ─────────────────────────┘
                       ↓
             pretrained DPT decoder → current prediction
```

在多尺度特征处使用形如
`F_current' = F_current + M * gate * adapter(F_current, warped_F_prev, difference, flow)`
的残差适配。adapter 末层零初始化，初始行为尽量保持原预训练模型。
共享两次冻结编码器前向，光流同样冻结；不新增从随机浅层 RGB CNN 开始的完整深度分支。
可先冻结原 DPT，仅训练 adapter，后续是否解冻最后的解码层由 dev 结果决定。

这使网络能从语义和结构特征重新解码深度，不再把输出硬限制在两个原始预测之间。
但它仍可能只学到利用当前特征的空间纠偏；必须用同容量单帧对照和历史扰动验证时序贡献。
Mono-ViFI 原始三帧权重/训练损失不直接等于这一两帧适配模型。

特征 flow 坐标必须匹配编码器 resize/padding 与特征 stride；现有 canvas 流不能不经坐标映射
就直接 resize 到另一个预处理空间。现有 NPZ 最终预测缓存也不包含预训练中间特征。
可以保留原缓存供旧模型使用，但新模型不能把它当完整特征输入缓存。

## 训练目标：不能只换网络

以下是项目适配建议，不是某一篇论文的原样配置。

- **逐帧几何准确性**：在有效训练 GT 上监督逆深度/深度，并保留空间梯度约束。
- **两帧共同标尺**：两张输出共享一组拟合参数建立 pair-SSI 或首帧锚定目标；
  不为每张输出分别拟合后再声称已约束视频尺度漂移。
- **运动感知时间项**：在 FB、图像有效区和双端 GT 有效区内，匹配预测变化与 GT 变化。
  相机移动或物体运动时，真实深度可以改变；单纯最小化 warp 后的深度差可能过度平滑。
- **保留预训练能力**：小学习率/参数高效适配；必要时在无 GT 或高置信度区域加入
  冻结模型的保持项，但不能用保持项压过有效 GT 的纠错监督。

仍沿用当前 KITTI/NYU H5/TUM 固定划分。NYU H5 的保留帧相邻不等于原始帧间隔；
不按 val/654 编号拼时序，不因新模型自动下载 Raw 或引入 benchmark 训练数据。
模型选择和超参数只看 train/dev，不根据现有 Sintel/test 结果挑配置。

## 能解释结果的对照

至少区分以下五项，训练预算按见过的 RGB/帧对数和算力成本报告：

1. 原始冻结 DA2-Small / 当前 guided：保留既有参照。
2. 新预训练底座，不训练、不加自定义融合：识别换 backbone 的收益。
3. 同底座、同容量的单帧适配：识别数据微调和空间纠偏的收益。
4. 真正两帧 + 光流 + FB + 置信度的完整模型：识别时序增量。
5. 同一完整 checkpoint 的历史/flow 干预：self-reference、打乱历史或移除显式 warp，
   根据控制目的分别设计；保留相同输入数量与预算，并报告干预的分布变化。

选择 DA3 时，另区分其冻结单视图与冻结双视图输出，不能仅以 DA2 为唯一基线。
保留 evaluate_v2 的精度与视频协议，另外报告无逐帧 GT 对齐的漂移诊断、
静态区域 warp 稳定性与边界表现。误差改善和时间一致性改善分开陈述。
最终仍需课程要求的相同 100 帧演示、固定视频归一化、runtime 和峰值显存。

## 建议顺序

接受大重构时，先围绕 DA3-Small 的双视图适配做设计；其多视图能力可以直接在严格两帧
条件下发挥，不必硬截断长视频模型。希望控制集成工作时，采用 DA2 内部特征融合备选。
不建议继续优先扩展 guided 的 seed/ensemble，也不建议把长视频模型改成两帧后直接宣称
继承论文的时序效果。正式实现与训练预算待用户选定路线后另行确定。
