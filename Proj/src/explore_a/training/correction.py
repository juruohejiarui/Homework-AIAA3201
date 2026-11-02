"""Train-only decomposition targets; no GT participates in forward context pooling."""
import torch
import torch.nn.functional as functional


@torch.no_grad()
def correction_targets(output, data, fraction):
    mixed = output["mixed_depth"].detach().float()
    truth = data["gt"].float()
    valid = (data["gt_valid"].bool() & (output["visibility"] > 0)
             & torch.isfinite(truth) & (truth > 0) & torch.isfinite(mixed) & (mixed > 0))
    safe = torch.where(valid, truth, mixed)
    if output.get("prediction_kind") == "relative_inverse":
        target = ((safe-mixed)/output["field_scale"]).clamp(-fraction, fraction)
    else:
        target = (safe / mixed.clamp_min(1e-6) - 1).clamp(-fraction, fraction)
    target = torch.where(valid, target, 0)
    count = valid.sum((-2, -1), keepdim=True)
    global_target = target.sum((-2, -1), keepdim=True) / count.clamp_min(1)
    residual = target - global_target
    room = torch.where(residual >= 0, fraction - global_target, fraction + global_target)
    local_target = torch.where(valid, residual / room.clamp_min(1e-6), 0).clamp(-1, 1)
    return global_target, local_target, valid


def correction_losses(output, data, settings):
    zero = output["depth"].sum() * 0
    fraction = settings["correction_fraction"]
    if "global_correction" not in output or fraction == 0:
        return zero, zero
    global_target, local_target, valid = correction_targets(output, data, fraction)
    count = valid.sum((-2, -1), keepdim=True)
    frames = count > 0
    global_error = functional.smooth_l1_loss(
        output["global_correction"] / fraction, global_target / fraction,
        beta=.1, reduction="none")
    local_error = functional.smooth_l1_loss(output["local_control"], local_target,
                                           beta=.1, reduction="none")
    local_error = torch.where(valid, local_error, 0).sum((-2, -1), keepdim=True)
    global_loss = torch.where(frames, global_error, 0).sum() / frames.sum().clamp_min(1)
    local_loss = (local_error / count.clamp_min(1)).sum() / frames.sum().clamp_min(1)
    return global_loss, local_loss
