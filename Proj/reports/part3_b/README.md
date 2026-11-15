# Part 3 B · NYUv2 Metric Scale Calibration

We fitted one global multiplier on the 795 official NYUv2 development images
and evaluated it unchanged on the 654 official test images. The multiplier was
**0.9340532609**. On the test split, AbsRel fell from **0.129425 to 0.103003**,
RMSE fell from **0.423729 m to 0.388024 m**, and δ1 rose from **87.5653% to
90.0662%**. This is a separate Part 3 experiment.

- [Report section (download and open the standalone HTML)](report-preview.html)
- [RGB, ground truth, uncalibrated and calibrated comparison](nyuv2_metric_scale_comparison.png)
- [Frozen scale](scale.json), [aggregate and per-image test metrics](comparison.json),
  and [model, environment and input hashes](run.json)
- [Reproduction commands](../../GUIDE_zh.md#方向-bnyuv2-全局米制尺度标定)

The [prediction archives](https://github.com/juruohejiarui/Homework-AIAA3201/releases/tag/part3-b-nyuv2-v1)
contain the original FP32 depth arrays in meters. Each archive contains a single
directory named `development_predictions` or `test_predictions`. Extract both
under `Proj/outputs/part3_b/` and follow the evaluation command in the guide to
recompute the scores without running model inference. The evaluator applies
the frozen multiplier before clipping for scoring. Obtain ground truth and
the split file from the [official NYUv2 source](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).

- Development (795 maps) SHA-256: `7cdbd9cad73e0cb3b44bfdd80169b7fc7106838f7af77da936499e701119bfea`
- Test (654 maps) SHA-256: `2dd841c5080990ad228a1b01b489129d0700fb20903d887b7ee30a5c2836faf5`
