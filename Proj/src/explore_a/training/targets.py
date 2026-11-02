"""Train-only optimal blend targets; no GT-derived fields enter model inputs."""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FusionTargets:
    blend: Tensor
    reset: Tensor
    blend_valid: Tensor
    reset_valid: Tensor


@torch.no_grad()
def fusion_targets(output, data, settings):
    current, historical = data["current_depth"].float(), output["historical"].detach()
    truth = torch.where(data["gt_valid"].bool(), data["gt"].float(), 1)
    difference = historical - current
    valid = data["gt_valid"].bool() & (output["visibility"] > 0)
    scale = (output["field_scale"] if output.get("prediction_kind") == "relative_inverse"
             else current.abs().clamp_min(1e-6))
    separated = difference.abs() >= scale * settings["blend_min_gap"]
    eligible = valid & separated
    denominator = torch.where(eligible, difference, 1)
    blend = ((truth - current) / denominator).clamp(0, 1)
    best = current + blend * difference
    denominator = scale if output.get("prediction_kind") == "relative_inverse" else truth
    gain = ((current - truth).abs() - (best - truth).abs()) / denominator.clamp_min(1e-6)
    threshold = settings["reset_min_gain"]
    # Near the decision boundary the binary label is unstable; retain continuous blend loss.
    unambiguous = (gain <= threshold * .5) | (gain >= threshold * 1.5)
    return FusionTargets(blend, (gain >= threshold).float(), eligible,
                         eligible & unambiguous)
