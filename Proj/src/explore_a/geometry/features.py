"""Fixed input channel definitions shared by all controlled experiment variants."""
import torch


def fusion_features(data, geometry, visibility, epsilon=1e-6):
    current = data["current_depth"].float().clamp_min(epsilon).log()
    historical = geometry["historical"].clamp_min(epsilon).log()
    historical = torch.where(geometry["hard"], historical, current)
    height, width = current.shape[-2:]
    return torch.cat([geometry["current_rgb"], geometry["warped_rgb"], current, historical,
                      (current - historical).abs(), visibility,
                      geometry["fb_residual"] / max(height, width),
                      geometry["photo_residual"]], 1)


def occlusion_features(data, geometry):
    scale = max(data["forward"].shape[-2:])
    return torch.cat([geometry["current_rgb"], geometry["previous_rgb"],
                      data["forward"] / scale, data["backward"] / scale,
                      geometry["fb_residual"] / scale, geometry["photo_residual"],
                      geometry["hard"].float()], 1)
