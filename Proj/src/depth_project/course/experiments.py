"""Full scene/image/video experiments; every prediction is retained."""
import subprocess
import time

import cv2
import h5py
import numpy as np
from scipy.io import loadmat

from depth_project.course.common import (URL, NYU_PROTOCOL, csv_file, frame_indices,
                                        identity, nyu_score, pin, read, stereo_score)
from depth_project.io import file_hash, read_rgb, write_json
from depth_project.paths import CHECKPOINTS, PROJECT


def prepare(root):
    source = root/'source/davis_rollercoaster.mp4'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        pending = source.with_suffix('.download')
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--output', str(pending), URL], check=True)
        pending.replace(source)
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames, decoded = [], 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded += 1
        if len(frames) < 100:
            h, w = image.shape[:2]
            scale = min(1., 640/max(h, w))
            size = (max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2))
            frames.append(cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB))
    capture.release()
    ids = frame_indices(decoded)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source FPS')
    root.joinpath('video').mkdir(exist_ok=True)
    frames = np.stack(frames)
    np.savez_compressed(root/'video/frames.npz', frames=frames, fps=fps, source_indices=ids)
    write_json(root/'video/source.json', {
        'download_url': URL, 'sha256': file_hash(source), 'decoded_source_frames': decoded,
        'selected_frames': len(ids), 'fps': fps, 'source_indices': ids.tolist(),
        'timestamps_s': (ids/fps).tolist(), 'output_hw': list(frames.shape[1:3]),
        'selection': 'first min(100, actual decoded frames); stride=1; original fps',
        'course_difference': 'Official source has fewer than 100 frames' if decoded < 100 else None})
    print('SOURCE', decoded, 'frames;', len(ids), 'selected;', fps, 'fps', flush=True)


def stereo(root):
    from depth_project.stereo.geometry import read_calibration
    from depth_project.stereo.matching import match_scene
    data = PROJECT/'data/middlebury/MiddEval3/trainingQ'
    scenes = sorted(p.parent for p in data.glob('*/disp0GT.pfm'))
    if len(scenes) != 15:
        raise ValueError('Expected exactly 15 labeled trainingQ scenes')
    folder = root/'stereo'
    files = [s/f for s in scenes for f in ['calib.txt', 'im0.png', 'im1.png', 'disp0GT.pfm']]
    pin(folder, identity(files, {'methods': ['bm', 'sgbm'], 'gt': 'finite >0',
                                'prediction': 'finite >=0', 'main_aggregate': 'pixel weighted'}))
    rows = []
    for method in ['bm', 'sgbm']:
        for scene in scenes:
            dest = folder/method/scene.name
            dest.mkdir(parents=True, exist_ok=True)
            if (dest/'arrays.npz').exists() and (dest/'metrics.json').exists():
                record = read(dest/'metrics.json')
            else:
                arrays, original = match_scene(scene, method)
                np.savez_compressed(dest/'arrays.npz', **arrays)
                f, b, offset, n = read_calibration(scene/'calib.txt')
                record = {'method': method, 'scene': scene.name,
                          **stereo_score(arrays['disparity'], arrays['gt_disparity']),
                          'seconds_left_and_right_matching': original['seconds'],
                          'focal_px': f, 'baseline_m': b, 'doffs_px': offset,
                          'num_disparities': (n+15)//16*16}
                write_json(dest/'metrics.json', record)
            rows.append(record)
            print('STEREO', method, scene.name, record['mae_px'], flush=True)
    csv_file(folder/'per_scene.csv', rows)
    summaries = []
    for method in ['bm', 'sgbm']:
        r = [x for x in rows if x['method'] == method]
        summaries.append({'method': method, 'scenes': len(r),
                          'mae_px_pixel_weighted': sum(x['absolute_error_sum'] for x in r)/sum(x['valid_predictions'] for x in r),
                          'coverage_pixel_weighted': sum(x['valid_predictions'] for x in r)/sum(x['gt_pixels'] for x in r),
                          'mae_px_scene_mean': float(np.mean([x['mae_px'] for x in r])),
                          'coverage_scene_mean': float(np.mean([x['coverage'] for x in r])),
                          'seconds_matching_total': sum(x['seconds_left_and_right_matching'] for x in r)})
    write_json(folder/'summary.json', summaries)
    csv_file(folder/'summary.csv', summaries)


def nyu(root):
    import torch
    from depth_project.models import load_model
    data = PROJECT/'data/nyuv2'
    weights = CHECKPOINTS/'unidepth-v2-vits14'
    ids = loadmat(data/'splits.mat')['testNdxs'].ravel().astype(int).tolist()
    train = loadmat(data/'splits.mat')['trainNdxs'].ravel().astype(int).tolist()
    if len(ids) != 654 or len(train) != 795 or set(ids) & set(train):
        raise ValueError('Invalid NYU split')
    folder = root/'nyu'
    pin(folder, identity([data/'splits.mat', data/'nyu_depth_v2_labeled.mat',
                          weights/'config.json', weights/'model.safetensors'],
                         {'protocol': NYU_PROTOCOL, 'model': 'unidepthv2-vits14',
                          'fp32': True, 'resolution_level': 9, 'input': 'RGB only'}))
    (folder/'predictions').mkdir(exist_ok=True)
    rows, adapter = [], None
    started = time.perf_counter()
    with h5py.File(data/'nyu_depth_v2_labeled.mat') as mat:
        for i, image_id in enumerate(ids):
            rgb = mat['images'][image_id-1].transpose(2, 1, 0).astype(np.uint8)
            gt = mat['depths'][image_id-1].T.astype(np.float32)
            name = f'{image_id:04d}'
            # Confirm the existing export against the authoritative MAT, not filenames alone.
            if not np.array_equal(rgb, read_rgb(data/'test/rgb'/f'{name}.png')):
                raise ValueError(f'RGB export differs from MAT: {name}')
            if not np.array_equal(gt, np.load(data/'test/gt'/f'{name}.npy')):
                raise ValueError(f'GT export differs from filled depths: {name}')
            dest = folder/'predictions'/f'{name}.npy'
            if dest.exists():
                prediction = np.load(dest)
            else:
                if adapter is None:
                    adapter = load_model('unidepthv2', weights, fp32=True)
                with torch.inference_mode():
                    prediction = adapter.predict(rgb).astype(np.float32)
                if prediction.shape != gt.shape:
                    prediction = cv2.resize(prediction, (640, 480), interpolation=cv2.INTER_LINEAR)
                if not np.isfinite(prediction).all():
                    raise ValueError(f'Nonfinite prediction: {name}')
                pending = dest.with_suffix('.part.npy')
                np.save(pending, prediction)
                pending.replace(dest)
            rows.append({'image_id': image_id, **nyu_score(prediction, gt)})
            if (i+1) % 25 == 0:
                print('NYU', i+1, '/654', flush=True)
    means = {key: float(np.mean([r[key] for r in rows])) for key in ['abs_rel', 'rmse_m', 'delta1']}
    write_json(folder/'metrics.json', {'count': len(rows), 'protocol': NYU_PROTOCOL,
                                      'test_ids': ids, 'mean': means, 'per_image': rows,
                                      'wall_s_including_io_and_load': time.perf_counter()-started})
    csv_file(folder/'per_image.csv', rows)
    print('NYU COMPLETE', means, flush=True)


def video(root, model):
    import torch
    from depth_project.models import load_model
    frames = np.load(root/'video/frames.npz')['frames']
    source = read(root/'video/source.json')
    weight = CHECKPOINTS/({'da2': 'depth_anything_v2_vits.pth',
                           'vda': 'video_depth_anything_vits.pth'}[model])
    folder = root/'video'/model
    pin(folder, identity([weight, root/'source/davis_rollercoaster.mp4'],
                         {'fp32': False, 'input_size': 518, 'source': source,
                          'algorithm': 'official infer_video_depth' if model == 'vda' else 'official infer_image'}))
    if (folder/'predictions.npy').exists() and (folder/'run.json').exists():
        return
    adapter = load_model(model, weight, fp32=False, input_size=518)
    started = time.perf_counter()
    sizes = [518] if model == 'da2' else [518, 392, 280]
    for size in sizes:
        try:
            with torch.inference_mode():
                if model == 'vda':
                    result, _ = adapter.model.infer_video_depth(frames, source['fps'],
                                                               input_size=size, device='cuda', fp32=False)
                else:
                    result = np.stack([adapter.predict(frame) for frame in frames])
            break
        except torch.cuda.OutOfMemoryError:
            if size == sizes[-1]:
                raise
            torch.cuda.empty_cache()
    result = np.asarray(result, np.float32)
    if result.shape != frames.shape[:3] or not np.isfinite(result).all():
        raise ValueError('Invalid video output')
    np.save(folder/'predictions.npy', result)
    lo, hi = np.percentile(result, [1, 99])
    write_json(folder/'run.json', {'model': model, 'representation': 'relative_inverse',
                                   'precision': 'fp16 autocast', 'input_size': size,
                                   'frames': len(frames), 'output_hw': list(result.shape[1:]),
                                   'fps': source['fps'], 'normalization_p1_p99': [float(lo), float(hi)],
                                   'wall_s_including_transfer': time.perf_counter()-started})
    print('VIDEO COMPLETE', model, result.shape, flush=True)
