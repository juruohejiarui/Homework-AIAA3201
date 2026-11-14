"""Render real outputs, recompute their scores, and validate the portable bundle."""
from pathlib import Path
import json
import subprocess
import zipfile

import cv2
import numpy as np
from scipy.io import loadmat

from depth_project.course.common import (color, csv_file, nyu_score, panel, read,
                                        stereo_score)
from depth_project.io import file_hash, read_rgb, write_json
from depth_project.paths import PROJECT


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def movie(path, images, fps):
    first = next(images)
    h, w = first.shape[:2]
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        process.stdin.write(first.tobytes())
        for image in images:
            process.stdin.write(image.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f'ffmpeg failed: {path}')


def render(root):
    data = PROJECT/'data/middlebury/MiddEval3/trainingQ'
    for method in ['bm', 'sgbm']:
        for scene in sorted(data.iterdir()):
            folder = root/'stereo'/method/scene.name
            if not folder.exists():
                continue
            with np.load(folder/'arrays.npz') as d:
                disparity, gt, depth = d['disparity'], d['gt_disparity'], d['depth_m']
                upper = float(np.percentile(gt[np.isfinite(gt) & (gt > 0)], 99))
                valid = np.isfinite(disparity) & (disparity >= 0)
                prediction = np.where(valid, disparity, np.nan)
                invalid = np.zeros((*gt.shape, 3), np.uint8)
                invalid[~valid] = (0, 0, 255)
                image = panel([cv2.imread(str(scene/'im0.png')), color(gt, 0, upper),
                               color(prediction, 0, upper), invalid,
                               color(depth, .001, 10, inverse=True)],
                              ['RGB', f'GT disparity [0,{upper:.1f}] px',
                               method.upper()+' disparity; same scale', 'Invalid prediction = red',
                               'Metric depth [.001,10] m; bright=near'], width=400, height=300)
                save(folder/'comparison.png', image)
                save(folder/'invalid.png', invalid)
    results = read(root/'nyu/metrics.json')
    rows = results['per_image']
    ordered = sorted(rows, key=lambda r: r['abs_rel'])
    selected = {r['image_id'] for r in ordered[:2]+ordered[-3:]}
    selected.update(rows[i]['image_id'] for i in np.linspace(0, 653, 8).astype(int))
    nyu_data = PROJECT/'data/nyuv2/test'
    for image_id in sorted(selected):
        name = f'{image_id:04d}'
        pred = np.load(root/'nyu/predictions'/f'{name}.npy')
        gt = np.load(nyu_data/'gt'/f'{name}.npy')
        rgb = cv2.imread(str(nyu_data/'rgb'/f'{name}.png'))
        mask = np.zeros_like(gt, bool)
        mask[45:471, 41:601] = True
        mask &= np.isfinite(gt) & (gt > .001) & (gt < 10)
        error = np.where(mask, np.abs(np.clip(pred, .001, 10)-gt)/np.maximum(gt, .001), np.nan)
        image = panel([rgb, color(gt, .001, 10, True), color(pred, .001, 10, True),
                       color(error, 0, 1)],
                      [f'NYU RGB id={name}', 'GT depth [.001,10] m; bright=near',
                       'UniDepthV2 depth [.001,10] m', 'AbsRel pixel error [0,1]; crop only'],
                      width=480, height=360)
        save(root/'nyu/figures'/f'{name}.png', image)
    write_json(root/'nyu/figures/selection.json', {
        'ids': sorted(selected), 'selection': '8 evenly spaced official test positions + 2 best and 3 worst AbsRel; descriptive only',
        'metric_color_range_m': [.001, 10], 'error_color_range': [0, 1]})
    frames = np.load(root/'video/frames.npz')['frames']
    source = read(root/'video/source.json')
    depths = {m: np.load(root/'video'/m/'predictions.npy', mmap_mode='r') for m in ['da2', 'vda']}
    ranges = {m: read(root/'video'/m/'run.json')['normalization_p1_p99'] for m in depths}

    def video_panel(i, methods):
        images = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)]
        labels = [f'RGB frame {i:03d} t={i/source["fps"]:.3f}s']
        for method in methods:
            images.append(color(depths[method][i], *ranges[method]))
            labels.append(('DA-V2-S' if method == 'da2' else 'VDA-S')+' relative depth; bright=near')
        return panel(images, labels, width=640, height=360)

    for methods, name in [(['da2'], 'rgb_da2'), (['vda'], 'rgb_vda'),
                          (['da2', 'vda'], 'rgb_da2_vda')]:
        movie(root/'video'/f'{name}.mp4', (video_panel(i, methods) for i in range(len(frames))), source['fps'])
    for i in sorted(set(np.linspace(0, len(frames)-1, 5).astype(int))):
        save(root/'video/figures'/f'frame_{i:03d}.png', video_panel(i, ['da2', 'vda']))
    # Consecutive central frames expose temporal behavior without resampling time.
    start = max(0, len(frames)//2-2)
    strip = np.concatenate([video_panel(i, ['da2', 'vda']) for i in range(start, min(start+5, len(frames)))])
    save(root/'video/figures/consecutive_frames.png', strip)
    write_json(root/'video/normalization.json', {'method_ranges': ranges,
                                               'colormap': 'INFERNO', 'per_frame_normalization': False,
                                               'units': 'relative inverse; no shared metric color scale'})
    # Three course-source cases, with full context and zoomed center crops.
    stereo_rows = read(root/'stereo/summary.json')
    import csv
    with (root/'stereo/per_scene.csv').open() as stream:
        sr = [r for r in csv.DictReader(stream) if r['method'] == 'sgbm']
    hardest = max(sr, key=lambda r: float(r['mae_px']))
    cases = [(root/'stereo/sgbm'/hardest['scene']/'comparison.png', 'stereo'),
             (root/'nyu/figures'/f'{ordered[-1]["image_id"]:04d}.png', 'nyu'),
             (root/'video/figures/consecutive_frames.png', 'video')]
    for path, name in cases:
        image = cv2.imread(str(path))
        save(root/'failures'/f'{name}.png', image)
    write_json(root/'failures/cases.json', {
        'stereo': {'scene': hardest['scene'], 'mae_px': float(hardest['mae_px']),
                   'selection': 'largest SGBM scene MAE; not tuning'},
        'nyu': {'image_id': ordered[-1]['image_id'], 'abs_rel': ordered[-1]['abs_rel'],
                'selection': 'largest per-image metric AbsRel; not tuning'},
        'video': {'frames': list(range(start, min(start+5, len(frames)))),
                  'selection': 'central 5 consecutive source frames; no depth GT'}})


def verify(root, archive=True):
    from depth_project.stereo.geometry import depth_from_disparity, read_calibration, read_pfm
    records = []
    data = PROJECT/'data/middlebury/MiddEval3/trainingQ'
    for method in ['bm', 'sgbm']:
        for scene in sorted(data.glob('*/disp0GT.pfm')):
            folder = root/'stereo'/method/scene.parent.name
            with np.load(folder/'arrays.npz') as d:
                gt = read_pfm(scene)
                np.testing.assert_array_equal(d['gt_disparity'], gt)
                actual = stereo_score(d['disparity'], gt)
                saved = read(folder/'metrics.json')
                for key in actual:
                    np.testing.assert_allclose(actual[key], saved[key], rtol=1e-12, atol=1e-12)
                f, b, offset, _ = read_calibration(scene.parent/'calib.txt')
                expected = depth_from_disparity(d['disparity'], f, b, offset)
                np.testing.assert_allclose(expected, d['depth_m'], rtol=0, atol=0, equal_nan=True)
                records.append({'method': method, 'scene': scene.parent.name, **actual})
    if len(records) != 30:
        raise ValueError('Expected 30 stereo outputs')
    for summary in read(root/'stereo/summary.json'):
        subset = [r for r in records if r['method'] == summary['method']]
        np.testing.assert_allclose(summary['mae_px_pixel_weighted'],
                                   sum(r['absolute_error_sum'] for r in subset)/sum(r['valid_predictions'] for r in subset), rtol=1e-12)
        np.testing.assert_allclose(summary['coverage_pixel_weighted'],
                                   sum(r['valid_predictions'] for r in subset)/sum(r['gt_pixels'] for r in subset), rtol=1e-12)
    data = PROJECT/'data/nyuv2'
    ids = loadmat(data/'splits.mat')['testNdxs'].ravel().astype(int).tolist()
    saved = read(root/'nyu/metrics.json')
    if saved['test_ids'] != ids or saved['count'] != 654:
        raise ValueError('NYU official sample IDs differ')
    if len(list((root/'nyu/predictions').glob('*.npy'))) != 654:
        raise ValueError('NYU prediction coverage differs')
    checked = []
    for image_id, old in zip(ids, saved['per_image']):
        pred = np.load(root/'nyu/predictions'/f'{image_id:04d}.npy')
        if pred.dtype != np.float32:
            raise ValueError('Prediction must be FP32 storage')
        gt = np.load(data/'test/gt'/f'{image_id:04d}.npy')
        actual = nyu_score(pred, gt)
        if old['image_id'] != image_id:
            raise ValueError('NYU per-image order mismatch')
        for key in actual:
            np.testing.assert_allclose(actual[key], old[key], rtol=1e-12, atol=1e-12)
        checked.append(actual)
    for key, value in saved['mean'].items():
        np.testing.assert_allclose(value, np.mean([r[key] for r in checked]), rtol=1e-12)
    frames = np.load(root/'video/frames.npz')
    source = read(root/'video/source.json')
    count = min(100, source['decoded_source_frames'])
    np.testing.assert_array_equal(frames['source_indices'], np.arange(count))
    if file_hash(root/'source/davis_rollercoaster.mp4') != source['sha256']:
        raise ValueError('Source file changed')
    norms = read(root/'video/normalization.json')
    for model in ['da2', 'vda']:
        prediction = np.load(root/'video'/model/'predictions.npy', mmap_mode='r')
        if prediction.shape != frames['frames'].shape[:3] or not np.isfinite(prediction).all():
            raise ValueError('Incomplete video prediction')
        np.testing.assert_allclose(np.percentile(prediction, [1, 99]), norms['method_ranges'][model], rtol=1e-12)
    videos = {}
    for name, panels in [('rgb_da2', 2), ('rgb_vda', 2), ('rgb_da2_vda', 3)]:
        cap = cv2.VideoCapture(str(root/'video'/f'{name}.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        i, differences = 0, []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape != (404, panels*640, 3) or i >= count:
                raise ValueError('Wrong rendered video dimensions/count')
            rgb = cv2.cvtColor(image[44:, :640], cv2.COLOR_BGR2RGB)
            differences.append(float(np.abs(rgb.astype(float)-frames['frames'][i]).mean()))
            i += 1
        cap.release()
        if i != count or abs(fps-source['fps']) > 1e-6 or max(differences) > 10:
            raise ValueError(f'Video count/FPS/RGB correspondence failed: {name}')
        videos[name] = {'frames': i, 'fps': fps, 'max_rgb_compression_mae': max(differences)}
    for model in ['unidepthv2', 'da2', 'vda']:
        result = read(root/'efficiency'/f'{model}.json')
        if len(result['rounds']) != 3 or not result['observed_network_tensor_shapes']:
            raise ValueError('Incomplete efficiency record')
    result = {'passed': True, 'stereo_outputs': len(records), 'nyu_images': len(checked),
              'nyu_mean_recomputed': {k: float(np.mean([r[k] for r in checked])) for k in saved['mean']},
              'videos': videos, 'scope': 'Complete Part 1 and Part 2; not full course submission'}
    if archive:
        path = root.parent/'part1_part2_results.zip'
        with zipfile.ZipFile(path) as package:
            bad = package.testzip()
            if bad is not None:
                raise ValueError(f'Archive CRC failure: {bad}')
            for rel in ['README.md', 'nyu/metrics.json', 'stereo/summary.csv', 'video/rgb_da2_vda.mp4']:
                if 'part1_part2_results/'+rel not in package.namelist():
                    raise ValueError(f'Archive missing {rel}')
        result['archive'] = {'path': str(path), 'bytes': path.stat().st_size,
                             'sha256': file_hash(path), 'all_entries_crc_checked': True}
    write_json(root/'verification.json', result)
    print('VERIFIED', result, flush=True)
    return result


def report(root):
    render(root)
    efficiency = [read(root/'efficiency'/f'{m}.json') for m in ['unidepthv2', 'da2', 'vda']]
    csv_file(root/'efficiency/summary.csv', [
        {k: r[k] for k in ['model', 'precision', 'gpu', 'ms_per_frame_mean',
                          'ms_per_frame_std', 'peak_allocated_mib']} for r in efficiency])
    source = read(root/'video/source.json')
    nyu = read(root/'nyu/metrics.json')['mean']
    stereo = read(root/'stereo/summary.json')
    text = '# Part 1 / Part 2 正式结果\n\n'
    text += '本包包含全部正式预测、逐样本指标、可视化、效率记录与复现信息。\n\n'
    text += f'官方原链接重新下载后实际解码 {source["decoded_source_frames"]} 帧；使用前 {source["selected_frames"]} 帧、{source["fps"]:g} fps，无跳帧、插帧或补帧。PDF写前100帧；本次按用户确认规则保留实际可用帧数。\n\n'
    text += '## NYU：UniDepthV2 ViT-S/14\n\n'
    text += 'RGB-only FP32，全部654张官方测试图，GT读取filled depths；480×640，crop[45:471,41:601]，有限GT 0.001<z<10m，预测截断[0.001,10]m，无尺度对齐，逐图等权。原始预测未经截断保存。\n\n'
    text += f'| AbsRel | RMSE (m) | δ1 |\n| --- | --- | --- |\n| {nyu["abs_rel"]:.8f} | {nyu["rmse_m"]:.8f} | {nyu["delta1"]:.8f} |\n\n'
    text += '## Middlebury：15个quarter-resolution训练场景\n\n'
    text += 'GT域为finite & disparity>0，预测finite & disparity>=0；MAE只在有效预测交集上计算，coverage以全部有效GT为分母。主汇总按像素加权，另存场景等权。左右一致性不缩小主评价域。\n\n'
    text += '| 方法 | MAE (px) | 有效预测比例 |\n| --- | --- | --- |\n'
    for r in stereo:
        text += f'| {r["method"].upper()} | {r["mae_px_pixel_weighted"]:.6f} | {r["coverage_pixel_weighted"]:.6f} |\n'
    text += '\n深度公式：f × baseline_mm / (1000 × (disparity_px + doffs))。OpenCV视差先除16。\n\n'
    text += '## 指定视频\n\nDA-V2-Small逐帧、VDA-Small官方离线推理，均为FP16 autocast；video/保存三份H.264原速演示及全部浮点预测。相同INFERNO色图，每方法整段固定P1/P99，亮色表示近处，relative depth颜色不代表共享米制尺度。实际上下限见video/normalization.json。\n\n'
    text += '## 效率\n\n计时包含预处理、推理、输出恢复与CPU传输，不含模型加载／磁盘写入；每模型独立进程、本机GPU、无缓存或DGX。均为3轮，UniDepth使用32张官方开发图而非测试图，单图模型预热5帧，VDA预热32帧窗口。实际网络输入形状及版本见efficiency/*.json。\n\n'
    text += '| 模型 | ms/frame均值 | 标准差 | 峰值allocated MiB |\n| --- | --- | --- | --- |\n'
    for r in efficiency:
        text += f'| {r["model"]} | {r["ms_per_frame_mean"]:.3f} | {r["ms_per_frame_std"]:.3f} | {r["peak_allocated_mib"]:.1f} |\n'
    text += '\n立体CPU计时见stereo/per_scene.csv，包含左／右两次匹配；不与GPU模型耗时混排。\n\n'
    text += '## 文件与复现\n\n- stereo/：两种方法各15场景、数组、无效区及对照图。\n- nyu/：654张原始FP32米制预测、逐图表、总指标与代表图。\n- video/：同帧RGB、两种预测、三份原速比较视频及连续帧图。\n- failures/：三个课程素材案例；详见analysis.md。\n- efficiency/：独立测量的每轮记录与汇总表。\n- 每阶段identity.json：代码、输入、权重及设置身份。\n\n'
    text += '仓库根目录执行：\n\n```bash\nexport PYTHONPATH="$PWD/Proj/src${PYTHONPATH:+:$PYTHONPATH}"\npython -m depth_project.course run\npython -m depth_project.course verify\npython -m depth_project.course report\n```\n\n使用现有aiaa3201环境。先准备官方Middlebury训练场景、NYU labeled MAT与splits，以及三个官方模型权重；命令不下载原始数据或权重。官方视频缺失时自动按PDF链接下载。身份改变时使用新的--output目录。\n\n'
    text += '## 范围与来源\n\n本包只交付Part 1／2，不包含Part 3、简单应用或最终CVPR论文，不作为完整课程提交包。无GT的视频只支持定性比较；不从色图声称米制精度。\n\n公开代码：https://github.com/juruohejiarui/Homework-AIAA3201\n\n模型与数据来源：Middlebury Stereo v3；NYU Depth V2；Depth Anything V2（NeurIPS 2024）；Video Depth Anything（CVPR 2025）；UniDepthV2（arXiv:2502.20110）。模型使用项目third_party中的官方代码和本地官方权重，版本及权重哈希见各identity.json。\n'
    (root/'README.md').write_text(text)
    if not (root/'analysis.md').exists():
        (root/'analysis.md').write_text('# 结果与难例\n\n正式数值和视频已生成。人工视觉复核与具体分析记录将在验收时补充。\n')
    verify(root, archive=False)
    # Portable identity/metric files use the same relative hierarchy as the local run.
    path = root.parent/'part1_part2_results.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as package:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix not in ['.tmp', '.log']:
                package.write(p, 'part1_part2_results/'+p.relative_to(root).as_posix())
    print('PACKAGED', path, path.stat().st_size, flush=True)
