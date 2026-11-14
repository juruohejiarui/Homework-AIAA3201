"""Course protocol, provenance and array-derived metrics."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from depth_project.io import file_hash, write_json
from depth_project.paths import PROJECT

URL = ('https://github.com/DepthAnything/Video-Depth-Anything/raw/refs/heads/'
       'main/assets/example_videos/davis_rollercoaster.mp4')
NYU_PROTOCOL = '480x640; crop[45:471,41:601]; finite GT .001<z<10; clip [.001,10]; no alignment; image mean'


def read(path):
    return json.loads(Path(path).read_text())


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(paths, settings):
    source = {}
    for folder in ['course', 'models', 'stereo']:
        for p in sorted((PROJECT/'src/depth_project'/folder).glob('*.py')):
            source[str(p.relative_to(PROJECT))] = file_hash(p)
    for p in paths:
        source[str(p.resolve())] = file_hash(p)
    return {'files': source, 'settings': settings}


def pin(folder, record):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'identity.json'
    if path.exists() and read(path) != record:
        raise ValueError(f'Identity changed; choose a new output directory: {folder}')
    write_json(path, record)


def nyu_score(prediction, truth):
    if prediction.shape != (480, 640) or truth.shape != prediction.shape:
        raise ValueError('NYU shape must be 480x640')
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite NYU prediction')
    mask = np.zeros(truth.shape, bool)
    mask[45:471, 41:601] = True
    mask &= np.isfinite(truth) & (truth > .001) & (truth < 10)
    if not mask.any():
        raise ValueError('Empty NYU evaluation mask')
    p = np.clip(prediction.astype(np.float64)[mask], .001, 10)
    t = truth.astype(np.float64)[mask]
    return {'abs_rel': float(np.mean(np.abs(p-t)/t)),
            'rmse_m': float(np.sqrt(np.mean((p-t)**2))),
            'delta1': float(np.mean(np.maximum(p/t, t/p) < 1.25)),
            'valid_pixels': int(mask.sum())}


def stereo_score(disparity, truth):
    mask = np.isfinite(truth) & (truth > 0)
    valid = mask & np.isfinite(disparity) & (disparity >= 0)
    count = int(valid.sum())
    total = int(mask.sum())
    if not total or not count:
        raise ValueError('Empty stereo GT/prediction intersection')
    error = float(np.abs(disparity[valid].astype(np.float64)-truth[valid]).sum())
    return {'mae_px': error/count, 'coverage': count/total, 'valid_predictions': count,
            'gt_pixels': total, 'absolute_error_sum': error}


def frame_indices(count):
    if count < 2:
        raise ValueError('At least two decoded video frames required')
    return np.arange(min(count, 100), dtype=np.int64)


def color(value, low, high, inverse=False):
    finite = np.isfinite(value)
    scaled = np.clip((np.nan_to_num(value, nan=low)-low)/max(high-low, 1e-12), 0, 1)
    if inverse:
        scaled = 1-scaled
    result = cv2.applyColorMap(np.rint(255*scaled).astype(np.uint8), cv2.COLORMAP_INFERNO)
    result[~finite] = 0
    return result


def panel(pictures, labels, width=640, height=480):
    canvas = np.full((height+44, width*len(pictures), 3), 245, np.uint8)
    for i, (picture, label) in enumerate(zip(pictures, labels)):
        h, w = picture.shape[:2]
        scale = min(width/w, height/h)
        image = cv2.resize(picture, (round(w*scale), round(h*scale)))
        y = 44+(height-image.shape[0])//2
        x = i*width+(width-image.shape[1])//2
        canvas[y:y+image.shape[0], x:x+image.shape[1]] = image
        cv2.putText(canvas, label, (i*width+8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas
