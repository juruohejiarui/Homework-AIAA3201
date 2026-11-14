"""Isolated, synchronized full-pipeline efficiency measurements."""
import contextlib
import os
import time

import h5py
import numpy as np
from scipy.io import loadmat

from depth_project.course.common import csv_file, read
from depth_project.io import write_json
from depth_project.paths import CHECKPOINTS, PROJECT


def measure(root, name):
    import torch
    from depth_project.models import load_model
    source = read(root/'video/source.json')
    video = np.load(root/'video/frames.npz')['frames']
    if name == 'unidepthv2':
        data = PROJECT/'data/nyuv2'
        ids = sorted(loadmat(data/'splits.mat')['trainNdxs'].ravel().astype(int).tolist())[:32]
        with h5py.File(data/'nyu_depth_v2_labeled.mat') as mat:
            frames = np.stack([mat['images'][i-1].transpose(2, 1, 0) for i in ids])
        weights = CHECKPOINTS/'unidepth-v2-vits14'
        size = None
    else:
        ids, frames = list(range(len(video))), video
        weights = CHECKPOINTS/({'da2': 'depth_anything_v2_vits.pth',
                                'vda': 'video_depth_anything_vits.pth'}[name])
        size = read(root/'video'/name/'run.json')['input_size']
    adapter = load_model(name, weights, fp32=name == 'unidepthv2', input_size=size or 518)
    observed = set()

    def observe(module, args):
        for value in args:
            if isinstance(value, torch.Tensor):
                observed.add(tuple(value.shape))
    target = adapter.model.pixel_encoder if name == 'unidepthv2' else adapter.model
    # Official image/video helpers sometimes call self.forward directly, bypassing
    # nn.Module.__call__ and its pre-hooks. Wrap forward without changing tensors.
    original_forward = target.forward

    def recorded_forward(*args, **kwargs):
        observe(target, args)
        return original_forward(*args, **kwargs)

    target.forward = recorded_forward

    def execute(batch):
        with torch.inference_mode():
            if name == 'vda':
                output, _ = adapter.model.infer_video_depth(batch, source['fps'], input_size=size,
                                                           device='cuda', fp32=False)
                del output
            else:
                for image in batch:
                    output = adapter.predict(image)
                    del output

    execute(frames[:32] if name == 'vda' else frames[:5])
    rounds = []
    for i in range(3):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        execute(frames)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        rounds.append({'round': i+1, 'seconds': seconds,
                       'ms_per_frame': seconds*1000/len(frames),
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        print('EFFICIENCY', name, rounds[-1], flush=True)
    target.forward = original_forward
    folder = root/'efficiency'
    folder.mkdir(exist_ok=True)
    result = {'model': name, 'precision': 'fp32' if name == 'unidepthv2' else 'fp16 autocast',
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads(),
              'input_size': size, 'input_rgb_hw': list(frames.shape[1:3]),
              'observed_network_tensor_shapes': [list(s) for s in sorted(observed)],
              'frames_per_round': len(frames), 'frame_or_image_ids': ids,
              'warmup_frames': 32 if name == 'vda' else 5, 'rounds': rounds,
              'ms_per_frame_mean': float(np.mean([r['ms_per_frame'] for r in rounds])),
              'ms_per_frame_std': float(np.std([r['ms_per_frame'] for r in rounds])),
              'peak_allocated_mib': max(r['peak_allocated_bytes'] for r in rounds)/2**20,
              'timing': 'preprocess + inference + output resize + CPU transfer; excludes load and disk IO',
              'execution': 'single local GPU, no cache or remote inference'}
    write_json(folder/f'{name}.json', result)
    return result
