> 当前正式视频集合固定为 KITTI、Sintel、Bonn RGB-D、TUM 官方 validation；
> 最新入口与全量范围见 [DA 指南](src/explore_a/DA_TRAINING.md)。
> 下文是此前候选数据集调研，其他集合不自动进入当前评价。

# 视频 benchmark 范围与 TUM 官方 validation 下载

核对日期：2026-11-02。只整理视频候选和下载工具；不改变训练、划分、缓存或评价结果。
课程单图 NYUv2 评价继续单列，不将 labeled 编号拼成视频。

## TUM 官方 validation 是什么

官方页面：
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

官方轨迹提交入口：
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/online_evaluation

当前网页抓取89个TGZ：47 ordinary、33 validation、9 calibration。
validation有明确的 `_validation` 文件名，不是把普通序列随机拆成90%/10%。
隐藏的是 motion-capture 相机轨迹；RGB和传感器深度仍公开。
提交页面接受 estimated trajectory，计算ATE/RPE，不接受单目深度图作为深度预测成绩。
我们的深度可以与RGB-D传感器深度在本机评分；没有公开轨迹时不伪造GT位姿用于3D TAE。
网页AVI是预览；下载工具选择原始RGB-D TGZ，保留16位深度与时间戳。

Sturm等2012 workshop论文第三节明确区分全公开training和隐藏轨迹的testing部分；
官网目前把后者叫validation。该早期论文的数量不是当前网页数量。
https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros_ws.pdf

UniDepthV2论文实验节称使用TUM validation split，但没有逐项列出本文可直接复核的序列/帧ID；
不能仅凭validation一词就断言其评价等于全部33个官方validation序列。
https://arxiv.org/html/2502.20110v2#S4.SS1

## 当前训练与官方validation的关系

逐项去掉 `_validation` 后，普通对应序列分别有7个在当前train、7个在dev、19个在test。
这是名称/录制场景关系，不是像素重复证明；不同录制不等于不同物理房间。
官方fr1/desk_validation描述也说明它拍摄与desk/desk2相同的四张桌子。
所以官方validation可做另一轨迹的同域测试，但不能自动称为未见房间或三源零样本测试。
本次没有重划现有manifest；挑选最终评价子集时要另行固定物理场景和帧清单。

## 下载与续传

已只查询页面及HEAD元数据，生成 `storage/tum/validation_plan.json`：
33个TGZ共27,254,725,677字节（27.255GB），尚未下载任何TGZ。
默认保存到 `~/Documents/ResourcesLow/Datasets/TUM-RGBD/official_validation/`。
计划中保存该目录当时解析后的绝对路径；download/extract复用该计划，不接入训练清单。

项目根目录执行：

```bash
conda activate aiaa3201
cd /home/hjr/Documents/Projects/Homework-AIAA3201
bash "$PWD/Proj/download_tum_validation.sh" status
bash "$PWD/Proj/download_tum_validation.sh" download --jobs 2
bash "$PWD/Proj/download_tum_validation.sh" extract
```

下载中断后重复download命令，aria2续传；下载完成按冻结大小与aria2状态核对。
官网没有在该清单提供官方摘要，不声称做过官方哈希验证。
原包在 `archives/`，解压在 `sequences/`，不下载ROS bag/AVI，不上传任何预测。
解压复用现有路径/成员类型检查与完成标记。

只选部分序列时使用另一计划，示例不代表已选定评价协议：

```bash
plan="$PWD/Proj/storage/tum/validation_small_plan.json"
bash "$PWD/Proj/download_tum_validation.sh" plan --manifest "$plan" \
  --sequences freiburg3_walking_xyz_validation freiburg3_sitting_xyz_validation
bash "$PWD/Proj/download_tum_validation.sh" download --manifest "$plan" --jobs 2
bash "$PWD/Proj/download_tum_validation.sh" extract --manifest "$plan"
```

已有计划不会被plan覆盖。原来的普通TUM下载入口行为保持不变。

## ScanNet实际评价范围

DepthCrafter公开脚本明确以 `scans_test` 为输入，scene_number=100；
官方v2 test名单正好100个扫描：scene0707_00至scene0806_00。
不是全库，也不是312个v2 val扫描。
https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_test.txt
https://raw.githubusercontent.com/Tencent/DepthCrafter/main/benchmark/dataset_extract/dataset_extract_scannet.py

VDA公开提取代码是输入目录排序后前100个扫描、每扫描先保留前510个图像，再生成：

| 协议 | 每扫描帧选择 | 上限 |
|---|---|---:|
| short | 前270个已导出帧，每隔3帧 | 90帧 |
| long | 前500个已导出帧，stride=1 | 500帧 |
| 代码TAE | 前192个已导出帧，stride=1 | 192帧 |

输入目录必须已是目标test集合；全库前100不能代替官方test。
这里的stride是已导出图像顺序；原始帧号/时间戳应再核对，不能无条件宣称某个fps。
VDA CVPR正文TAE表头写170帧，而代码提取192帧，二者不一致，应单独标记。
论文不同视频长度曲线另只用ScanNet10段、Bonn10段、NYUv2 8段，每段500帧，
不是把全部ScanNet拿来做长度消融；确切10段仍需固定名单。
https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/main/benchmark/dataset_extract/dataset_extract_scannet.py
https://arxiv.org/html/2501.12375v3#S4.SS2

下载评价素材主要需要目标扫描的.sens，可导出RGB、depth、pose、intrinsic；无需下载全部语义网格。
仍须按ScanNet官方流程取得授权。本机旧v1下载只剩.part，不能算test素材已就绪。

## Spring实际评价范围

官方47段：37 train（5000组立体帧，有公开GT）、10 test（1000组，仅公开图像）。
官方服务器收的是stereo/optical-flow/scene-flow预测，不是本项目统一的单目视频深度排行榜。
https://openaccess.thecvf.com/content/CVPR2023/papers/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.pdf
https://spring-benchmark.org/faq

Depth Pro把Spring用于单帧深度边界F1；论文没有给出足以确认“全部37段或某个子集”的
完整序列/逐帧清单。不能把37+10全库当作已确认论文protocol。
https://arxiv.org/html/2410.02073v2#S4

本项目若新增视频深度评价，可使用公开GT的train中的固定完整序列，称项目评价子集；
train是原数据集名字，不意味着用它训练融合头。需左目RGB、左目视差、相机标定，
按官方单位转换深度；只选左右立体图不是时序输入。未自动挑选、下载或评价Spring。

## 仅视频的候选表

| 候选 | 当前/可采用范围 | 就绪状态 |
|---|---|---|
| KITTI | 当前13段short，共1269帧 | 已有 |
| Sintel | 当前23段，共1064帧 | 已有 |
| Bonn | 当前26段short，共2860帧 | 已有 |
| TUM RGB-D | 官方33 validation，或固定其子集 | 清单已备，待下载；注意场景关系 |
| NYUv2 Raw | VDA指定8段，最多500帧/段 | Raw未准备；不用labeled/H5训练帧替代 |
| ScanNet v2 | test100段的short90/long500；长度消融10段另列 | 待授权及下载.sens |
| Spring | 37个公开GT train序列中的固定视频子集 | 待选择，不冒充原论文精确划分 |
| FLSea | 水下视频，需固定图像/深度对应与序列 | 可扩展，待准备视频protocol |
| DDAD、nuScenes | 驾驶视频来源，需要LiDAR投影和时序清单 | 可扩展，原单图评价不能直接当视频协议 |

静态ETH3D/DIODE/iBims/SUN/Middlebury/Booster/DA-2K等不列入这次视频候选。
指定DAVIS rollercoaster作为课程视频定性展示单列，没有GT深度，不合入定量深度均值。

## 验证

运行新下载选择/续传调用测试及受影响的数据回归：7项通过。
实际33个官方HEAD成功；薄入口help/status与shell语法通过，新增/修改Python行宽≤100。
未运行正式训练、GPU推理、全量解码、数据集载荷下载或预测提交。
