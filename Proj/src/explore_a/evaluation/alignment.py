"""GT-only image/video alignment and deterministic raw-field ordinal diagnostics."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from depth_project.evaluation.metrics import depth_metrics


def numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value).squeeze()


def moments(prediction, truth):
    x, y = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    return np.array([x.size, x.sum(), y.sum(), (x*x).sum(), (x*y).sum()], np.float64)


def fit(stats):
    n, sx, sy, sxx, sxy = stats
    if n == 0:
        return 0., 0.
    variance = sxx-sx*sx/n
    a = max(0., float((sxy-sx*sy/n)/variance)) if variance > 1e-15 else 0.
    return a, float((sy-a*sx)/n)


def depth(prediction, mapping, limits):
    a, b = mapping
    return np.clip(1/np.maximum(a*prediction+b, 1e-6), *limits)


def named(values, suffix):
    return {f"{key}_{suffix}": values[source]
            for key, source in (("absrel", "abs_rel"), ("rmse", "rmse"), ("delta1", "d1"))}


def ordinal(prediction, truth, sample_id):
    """Sensor-GT derived pairs, shared by frame ID; this is not DA-2K's annotation set."""
    if len(truth) < 2:
        return {}
    seed = int.from_bytes(sha256(str(sample_id).encode()).digest()[:8], "little")
    points = np.random.default_rng(seed).integers(0, len(truth), size=(4096, 2))
    first, second = points.T
    ratio = np.maximum(truth[first]/truth[second], truth[second]/truth[first])
    points = points[ratio >= 1.1][:1024]
    if not len(points):
        return {}
    first, second = points.T
    correct = np.sign(prediction[first]-prediction[second]) == np.sign(truth[second]-truth[first])
    return {"ordinal_accuracy": float(correct.mean()), "ordinal_pairs": int(len(points))}


class RelativeScores:
    """Bounded RAM: spool valid FP32 pixels, reduce clip moments, score without rerunning GPU.

    Temporary arrays live on the system filesystem, never in the read-only prediction cache.
    """
    def __init__(self):
        self.directory = TemporaryDirectory(prefix="da-alignment-")
        self.stats = defaultdict(lambda: np.zeros(5, np.float64))
        self.frames = []

    def add(self, row, prediction, truth, valid, limits):
        p, g, m = numpy(prediction), numpy(truth), numpy(valid).astype(bool)
        m = m & np.isfinite(g) & (g > 0)
        if not np.isfinite(p[m]).all():
            raise ValueError("Nonfinite prediction in relative scoring")
        p, g = p[m].astype(np.float32), g[m].astype(np.float32)
        key = (row["dataset"], row["sequence"], row.get("camera", ""),
               row.get("evaluation_clip", "full-sequence"))
        record = {}
        if len(g):
            stats = moments(p, 1/g.astype(np.float64))
            self.stats[key] += stats
            values = depth_metrics(depth(p, fit(stats), limits), g)
            record = {**named(values, "img"), **ordinal(p, g, row["id"])}
            path = Path(self.directory.name)/f"{len(self.frames)}.npy"
            np.save(path, np.stack((p, g)), allow_pickle=False)
            self.frames.append((row, key, path, limits))
        return record

    def finish(self, baseline_stats=None):
        try:
            for row, key, path, limits in self.frames:
                p, g = np.load(path, allow_pickle=False)
                values = depth_metrics(depth(p, fit(self.stats[key]), limits), g)
                measured = named(values, "video")
                if baseline_stats is not None and key in baseline_stats:
                    fixed = depth_metrics(depth(p, fit(baseline_stats[key]), limits), g)
                    measured.update(named(fixed, "baseline_fixed_video"))
                yield row, measured, len(g)
        finally:
            self.close()

    def close(self):
        self.directory.cleanup()
