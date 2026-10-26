# 模型与源码：由用户下载

2026-10-24。下面四个权重地址已通过官方端点 HEAD 确认返回 200；本次没有下载权重正文。已有 Depth Pro 直接复用。

| 模型 | 官方下载地址 | 保存到 Proj 下 | 本次响应大小 |
|---|---|---|---:|
| RAFT Large `C_T_SKHT_V2` | [raft_large_C_T_SKHT_V2-ff5fadd5.pth](https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth) | `checkpoints/explore_a/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth` | 21,106,607 B |
| MFTIQ UOM | [UOM_bs4_200k.pth](https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/UOM_bs4_200k.pth) | `checkpoints/explore_a/mftiq/UOM_bs4_200k.pth` | 108,034,251 B |
| ResNet-50 V2 | [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) | `checkpoints/explore_a/backbones/resnet50-11ad3fa6.pth` | 102,540,417 B |
| DINOv2 ViT-S/14 register | [dinov2_vits14_reg4_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth) | `checkpoints/explore_a/backbones/dinov2_vits14_reg4_pretrain.pth` | 88,291,785 B |

四个新增权重约 320MB。下面命令只作为你手动执行的下载方式：

```bash
cd /home/hjr/Documents/Projects/Homework-AIAA3201
mkdir -p Proj/checkpoints/explore_a/{raft,mftiq,backbones}

curl -fL --retry 5 -C - \
  https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth \
  -o Proj/checkpoints/explore_a/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth
curl -fL --retry 5 -C - \
  https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/UOM_bs4_200k.pth \
  -o Proj/checkpoints/explore_a/mftiq/UOM_bs4_200k.pth
curl -fL --retry 5 -C - \
  https://download.pytorch.org/models/resnet50-11ad3fa6.pth \
  -o Proj/checkpoints/explore_a/backbones/resnet50-11ad3fa6.pth
curl -fL --retry 5 -C - \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth \
  -o Proj/checkpoints/explore_a/backbones/dinov2_vits14_reg4_pretrain.pth
```

另需 DINOv2 的模型定义源码；在训练前手动获取一次，运行时从本地读取：

```bash
git clone --depth 1 https://github.com/facebookresearch/dinov2.git Proj/third_party/dinov2
```

如果目录已经存在，直接复用，不重复 clone。骨干构造明确使用 `pretrained=False` 再加载本地权重，运行时不会通过 Hub 下载。为隔离可选 xFormers 算子的环境差异，可在启动前设 `export XFORMERS_DISABLED=1`，使用 DINOv2 的 PyTorch 路径。

MFTIQ 所需的 UOM 推理源码已随本项目提供，位于 `vendor/mftiq`，来源和局部适配记录见其中 `UPSTREAM.md`。保留作者 CC BY-NC-SA 4.0 许可和署名；无需安装作者整套旧版环境，也无需下载其 RoMa、FlowFormer 或其他光流权重。

correlation 优先使用已安装的 `spatial_correlation_sampler`，否则使用本项目的等价 PyTorch 实现：单像素 kernel、7×7 位移窗、通道求和。后备实现保留 checkpoint 结构，但可能慢于已编译扩展。

下载结束后执行：

```bash
conda activate aiaa3201
bash Proj/explore_a.sh check-weights
```

该命令检查文件存在、已知文件大小及 DINOv2 源码入口；不下载任何模型，不执行推理，不做全量哈希。所有外部 checkpoint 均用 `weights_only=True` 加载并严格匹配网络结构。
