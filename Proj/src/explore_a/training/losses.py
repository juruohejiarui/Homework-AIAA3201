"""Metric supervision only on measured valid GT; no confidence labels are invented."""
import torch.nn.functional as functional


def masked_mean(value, mask):
    count = mask.sum()
    if count == 0:
        return value.sum() * 0
    return value[mask].mean()


def fusion_loss(output, data, settings):
    valid = data["gt_valid"].bool()
    target = data["gt"].clamp_min(1e-6)
    error = functional.smooth_l1_loss(output["depth"].log(), target.log(),
                                     beta=settings["log_smooth_l1_beta"], reduction="none")
    loss = masked_mean(error, valid)
    penalty = masked_mean(output["residual_fraction"].square(), valid)
    return loss + settings["residual_penalty"] * penalty


def student_loss(logits, data, geometry):
    target = data["teacher_visibility"].float().clamp(0, 1)
    error = functional.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return masked_mean(error, geometry["hard"])
