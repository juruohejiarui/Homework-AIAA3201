"""Depth supervision and explicit, separately recorded fusion auxiliary losses."""
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as functional

from explore_a.training.targets import fusion_targets
from explore_a.training.correction import correction_losses


def masked_mean(value, mask):
    selected = torch.where(mask, value, 0)
    return selected.sum() / mask.sum().clamp_min(1)


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    depth: Tensor
    blend: Tensor
    reset: Tensor
    correction: Tensor
    global_correction: Tensor
    local_correction: Tensor
    temporal: Tensor | None = None
    gauge_fallback: Tensor | None = None
    gt_coverage: Tensor | None = None

    def scalars(self):
        return {"depth_loss": float(self.depth.detach()), "blend_loss": float(self.blend.detach()),
                "reset_loss": float(self.reset.detach()),
                "correction_loss": float(self.correction.detach()),
                "global_correction_loss": float(self.global_correction.detach()),
                "local_correction_loss": float(self.local_correction.detach()),
                "aux_depth_ratio": float((self.total.detach() - self.depth.detach())
                                         / self.depth.detach().clamp_min(1e-8)),
                "temporal_loss": 0. if self.temporal is None else float(self.temporal.detach()),
                "gauge_fallback_fraction": 0. if self.gauge_fallback is None
                else float(self.gauge_fallback.detach()),
                "gt_coverage": 0. if self.gt_coverage is None else float(self.gt_coverage.detach())}


def fusion_loss(output, data, settings):
    if output.get("prediction_kind") == "relative_inverse":
        from explore_a.training.relative import loss
        return loss(output, data, settings)
    valid = data["gt_valid"].bool()
    target = torch.where(valid, data["gt"], 1).clamp_min(1e-6)
    error = functional.smooth_l1_loss(output["depth"].clamp_min(1e-6).log(), target.log(),
                                     beta=settings["log_smooth_l1_beta"], reduction="none")
    depth = masked_mean(error, valid)
    blend, reset = depth * 0, depth * 0
    if "weight_logit" in output:
        targets = fusion_targets(output, data, settings)
        blend = masked_mean(functional.binary_cross_entropy_with_logits(
            output["weight_logit"], targets.blend, reduction="none"), targets.blend_valid)
        reset = masked_mean(functional.binary_cross_entropy_with_logits(
            output["reset_logit"], targets.reset, reduction="none"), targets.reset_valid)
    correction = masked_mean(output["correction"].square(), valid)
    global_loss, local_loss = correction_losses(output, data, settings)
    total = (depth + settings["blend_aux_weight"] * blend + settings["reset_aux_weight"] * reset
             + settings["correction_penalty"] * correction
             + settings.get("global_correction_weight", .01) * global_loss
             + settings.get("local_correction_weight", .01) * local_loss)
    return LossTerms(total, depth, blend, reset, correction, global_loss, local_loss)
