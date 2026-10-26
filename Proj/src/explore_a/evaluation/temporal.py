"""Temporal diagnostics use fixed Farneback correspondences, independent of the experiment."""
import torch

from explore_a.geometry.warp import pair_geometry, warp


def flow_temporal(current, previous, data, settings):
    geometry = pair_geometry(data, settings)
    prior, _ = warp(previous, data["backward"])
    mask = geometry["rule"]
    result = {}
    if mask.any():
        result["flow_aligned_mae_m"] = float((current - prior).abs()[mask].mean())
        result["apparent_log_scale_step"] = float(
            (current.clamp_min(1e-6).log() - prior.clamp_min(1e-6).log())[mask].median())
    gt_previous, _ = warp(data["previous_gt"], data["backward"])
    gt_valid, _ = warp(data["previous_gt_valid"].float(), data["backward"])
    measured = mask & data["gt_valid"].bool() & (gt_valid > .999)
    if measured.any():
        # Subtract measured change: a camera moving toward a wall is not depth flicker.
        change = current - prior - (data["gt"] - gt_previous)
        result["gt_corrected_temporal_mae_m"] = float(change.abs()[measured].mean())
    return result


def geometric_temporal(current, previous, data):
    """GT defines static visible correspondences; compare prediction residual in current Z."""
    if "pose" not in data or "previous_pose" not in data or "K" not in data:
        return {}
    k = data["K"][0].float()
    height, width = current.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=current.device),
                            torch.arange(width, device=current.device), indexing="ij")
    rays = torch.linalg.solve(k, torch.stack([xx, yy, torch.ones_like(xx)]).float().reshape(3, -1))
    transform = torch.linalg.solve(data["pose"][0].float(), data["previous_pose"][0].float())
    ground = rays * data["previous_gt"].reshape(1, -1)
    reference = transform[:3, :3] @ ground + transform[:3, 3:4]
    predicted = transform[:3, :3] @ (rays * previous.reshape(1, -1)) + transform[:3, 3:4]
    uv = k @ reference
    valid = data["previous_gt_valid"].reshape(-1).bool() & (reference[2] > 0)
    coordinates = (uv[:2] / uv[2:3].clamp_min(1e-6)).round().long()
    x, y = coordinates
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    ids = torch.where(valid)[0]
    if not len(ids):
        return {}
    x, y = x[ids], y[ids]
    truth = data["gt"][0, 0, y, x]
    target_valid = data["gt_valid"][0, 0, y, x].bool()
    # A 5%/5cm visibility tolerance rejects occluded surfaces and nonrigid motion.
    tolerance = torch.maximum(truth * .05, truth.new_tensor(.05))
    target_valid &= (reference[2, ids] - truth).abs() < tolerance
    if not target_valid.any():
        return {}
    error = current[0, 0, y, x] - predicted[2, ids] - (truth - reference[2, ids])
    return {"static_geometric_gt_corrected_mae_m": float(error.abs()[target_valid].mean())}
