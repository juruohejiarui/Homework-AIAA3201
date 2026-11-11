"""Sintel image scoring reproduced from DepthAnything-AC, Table 5.

Upstream HVision-NKU/DepthAnythingAC f1209ed2e103344b346aef69a7e30c5e602e1ccd.
The existing video metric retains its own mask, range and aggregation.
"""
from pathlib import Path

import numpy as np
import torch

PROTOCOL = {
    "id": "depthanything-ac-sintel-final-v1",
    "paper": "https://arxiv.org/abs/2507.01634",
    "render_pass": "final", "frames": 1064, "sequences": 23,
    "gt_cap_m": 72, "prediction_limits_m": [1e-5, 80],
    "alignment": "per-image inverse-depth least squares, unconstrained scale",
    "valid": "GT > 0 after GT clamping", "metrics_space": "depth",
    "image_aggregation": "frame_mean",
    "video_aggregation": "sequence_macro", "video_max_depth_m": 70,
}
IMAGE_METRICS = ("absrel_img", "delta1_img", "rmse_img")


def prepare_manifest(manifest, root):
    """Apply the adopted cohort to every benchmark entry point."""
    if manifest["dataset"] != "sintel":
        return
    sequences = manifest["sequences"]
    if (len(sequences) != PROTOCOL["sequences"] or
            sum(len(s["frames"]) for s in sequences) != PROTOCOL["frames"]):
        raise ValueError("AC Sintel requires all 23 sequences / 1064 frames")
    manifest["config"] = {**manifest["config"], "image_scoring": PROTOCOL,
                          "crop": [0, 436, 0, 1024], "factor": 1, "max_depth": 70}
    for seq in sequences:
        for frame in seq["frames"]:
            frame["image"] = final_rgb(frame["image"], root)


def frame_mean(records, branch=None):
    """Aggregate AC image metrics across unequal scene lengths."""
    total = sum(r["frames"] for r in records)
    values = [r["branches"][branch] if branch else r for r in records]
    return {key: sum(v[key]*r["frames"] for r,v in zip(records, values))/total
            for key in IMAGE_METRICS}


def final_rgb(path, root=None):
    """Resolve the sibling final pass without silently falling back to clean."""
    path = Path(path)
    parts = list(path.parts)
    for i in range(len(parts) - 1):
        if parts[i] == "training" and parts[i + 1] in ("clean", "final"):
            parts[i + 1] = "final"
            result = Path(*parts)
            if root is not None:
                from explore_a.evaluation.part2_protocol import benchmark_path
                return benchmark_path(str(result), root)
            if result.is_file():
                return str(result)
            raise FileNotFoundError(f"AC Sintel requires final RGB: {result}")
    raise ValueError(f"Expected Sintel training/clean or training/final RGB: {path}")


def image_scores(prediction, truth):
    """Use the upstream FP32 normal equations, disparity floor and depth metrics.

    Do not discard far GT or constrain the fitted scale: both change this protocol.
    Reductions include the full masked HW field, as in the public implementation.
    """
    p = torch.from_numpy(np.array(prediction, dtype=np.float32, copy=True))
    g = torch.from_numpy(np.minimum(np.asarray(truth, np.float32), 72).copy())
    m = torch.isfinite(g) & (g > 0)
    if not m.any():
        return {}
    if not torch.isfinite(p[m]).all():
        raise ValueError("Nonfinite prediction in AC Sintel image scoring")
    p = torch.where(m, p, 0.)
    target = torch.zeros_like(g)
    target[m] = 1 / g[m]
    a00, a01, a11 = (m*p*p).sum(), (m*p).sum(), m.sum()
    b0, b1 = (m*p*target).sum(), (m*target).sum()
    det = a00*a11 - a01*a01
    if det > 0:
        scale = (a11*b0-a01*b1)/det
        shift = (-a01*b0+a00*b1)/det
    else:
        scale = shift = torch.zeros((), dtype=torch.float32)
    aligned = (scale*p+shift).clamp_min(1/80)
    depth = (1/aligned).clamp(1e-5, 80).numpy()[m.numpy()]
    gt = g.numpy()[m.numpy()]
    return {"absrel_img": float(np.mean(np.abs(depth-gt)/gt)),
            "delta1_img": float(np.mean(np.maximum(depth/gt, gt/depth) < 1.25)),
            "rmse_img": float(np.sqrt(np.mean((depth-gt)**2)))}
