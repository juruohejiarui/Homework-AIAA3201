"""Shared official NYUv2 labeled-image metric helpers."""
import importlib.util

import numpy as np
import torch

from depth_project.paths import THIRD_PARTY


def metric_function():
    path = THIRD_PARTY / "Depth-Anything-V2/metric_depth/util/metric.py"
    spec = importlib.util.spec_from_file_location("nyu_upstream_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_depth


def score(prediction, truth, metric):
    mask = np.zeros(truth.shape, bool)
    mask[45:471, 41:601] = True
    mask &= np.isfinite(truth) & (truth > .001) & (truth < 10)
    values = np.clip(prediction, .001, 10)
    return metric(torch.from_numpy(values[mask].astype(np.float64)),
                  torch.from_numpy(truth[mask].astype(np.float64)))


