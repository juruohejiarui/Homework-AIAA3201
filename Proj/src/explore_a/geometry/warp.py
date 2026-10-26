"""Backward sampling and correspondence masks in the current frame's coordinate grid."""
import torch
import torch.nn.functional as functional


def warp(previous, backward):
    """Sample previous[B,C,H,W] at x + B(cur->prev); return values and bounds mask."""
    batch, _, height, width = backward.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=backward.device),
                            torch.arange(width, device=backward.device), indexing="ij")
    x = xx[None].float() + backward[:, 0].float()
    y = yy[None].float() + backward[:, 1].float()
    valid = torch.isfinite(x) & torch.isfinite(y)
    valid &= (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    grid = torch.stack([2 * x / max(width - 1, 1) - 1,
                        2 * y / max(height - 1, 1) - 1], -1)
    grid = torch.nan_to_num(grid, nan=2, posinf=2, neginf=-2)
    values = functional.grid_sample(previous.float(), grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True)
    return values, valid.reshape(batch, 1, height, width)


def pair_geometry(data, settings):
    forward, backward = data["forward"], data["backward"]
    previous_rgb = data["previous_rgb"].float() / 255
    current_rgb = data["current_rgb"].float() / 255
    historical, bounds = warp(data["previous_depth"], backward)
    previous_color, _ = warp(previous_rgb, backward)
    warped_flow, _ = warp(forward, backward)
    sampled_valid, _ = warp(data["image_valid"].float(), backward)
    historical_valid, _ = warp((torch.isfinite(data["previous_depth"])
                                & (data["previous_depth"] > 0)).float(), backward)
    hard = bounds & (sampled_valid > .999) & data["image_valid"].bool()
    hard &= (historical_valid > .999) & torch.isfinite(historical) & (historical > 0)
    residual2 = (backward + warped_flow).square().sum(1, keepdim=True)
    bound = settings["fb_relative"] * (backward.square() + warped_flow.square()).sum(
        1, keepdim=True) + settings["fb_absolute_px2"]
    rule = hard & (residual2 <= bound)
    return {"current_rgb": current_rgb, "previous_rgb": previous_rgb,
            "warped_rgb": previous_color, "historical": torch.where(hard, historical, 0),
            "hard": hard, "rule": rule, "fb_residual": residual2.sqrt(),
            "photo_residual": (current_rgb - previous_color).abs().mean(1, keepdim=True)}
