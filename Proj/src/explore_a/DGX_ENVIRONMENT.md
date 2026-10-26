# DGX环境实测记录

实测日期：2026-10-25。独立Conda环境`aiaa3201`；未修改fyp-train/vllm。

| 项目 | 本机x86_64 | DGX aarch64 |
|---|---|---|
| Python | 3.13.16 | 3.13.16 |
| torch | 2.12.0+cu130 | 2.12.0+cu130 |
| torchvision | 0.27.0+cu130 | 0.27.0+cu130 |
| numpy | 2.5.4 | 2.5.4 |
| scipy | 1.18.1 | 1.18.1 |
| opencv-python | 5.0.0.93 | 5.0.0.93 |
| pillow | 13.0.0 | 13.0.0 |
| timm | 1.0.30 | 1.0.30 |
| einops | 0.8.2 | 0.8.2 |
| h5py | 3.16.0 | 3.16.0 |
| matplotlib | 3.11.2 | 3.11.2 |
| tensorboard | 2.21.0 | 2.21.0 |
| flash-attn | 2.8.3+cu130torch2.12 | 2.8.3+cu130torch2.12 |

DGX GPU：NVIDIA GB10，CUDA可用。外部FlashAttention2和原生SDPA Flash均完成一次
FP16小张量前向并返回有限值。这只是算子兼容结果；模型实际调用次数与回退原因需查看
正式生成后每个任务的`local.complete.json`，本次未运行正式缓存。

直接依赖均匹配本机版本，ARM GPU轮子单独安装；传递依赖允许由ARM解析器选择兼容版本。
完整记录：[ARM freeze锁文件](../../requirements-arm64-cache.lock.txt)、
[结构化环境报告](../../storage/dgx-environment.json)。锁文件中的editable源码路径为DGX固定项目路径。

环境导入检查不等于模型精度或端到端性能测试。缓存记录保持FP32，MFTIQ/DINOv2保持FP32；
不承诺ARM与x86逐位一致。
