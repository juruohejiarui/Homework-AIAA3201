"""GT-only fixed baseline gauges, kept outside every forward/cache/RPC interface."""
import torch
import torch.nn.functional as F

from explore_a.geometry.relative import affine, statistics


@torch.no_grad()
def gauge(data):
    raw, gt = data["current_depth"].float(), data["gt"].float()
    valid = data["gt_valid"].bool() & torch.isfinite(gt) & (gt > 0)
    inverse = torch.where(valid, 1/gt.clamp_min(1e-6), 0)
    a, b, good = affine(raw, inverse, valid, minimum=2, robust=False)
    for i in range(len(raw)):
        if valid[i].any() and not good[i].item():
            a[i] = inverse[i][valid[i]].median()/raw[i][valid[i]].median().clamp_min(1e-6)
            b[i] = 0
    _, scale = statistics(inverse, valid)
    # Relative inverse error is normalized by the target magnitude, not predicted meters.
    for i in range(len(raw)):
        if valid[i].any():
            scale[i] = inverse[i][valid[i]].median().clamp_min(1e-6)
    return {"a": a, "b": b, "scale": scale, "inverse": inverse, "valid": valid,
            "fallback": ~good, "target": (inverse-b)/a.clamp_min(1e-12)}


def main_loss(prediction, fixed):
    error = (fixed["a"]*prediction.float()+fixed["b"]-fixed["inverse"])/fixed["scale"]
    values = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.1, reduction="none")
    mask = fixed["valid"]
    return torch.where(mask, values, 0).sum()/mask.sum().clamp_min(1)


def scoring_depth(prediction, target, valid):
    """Scoring-only per-image inverse SSI, never an inference-time metric conversion."""
    from depth_project.evaluation.metrics import align_inverse_depth
    values = prediction.detach().float().cpu().numpy()
    truth, mask = target.float().cpu().numpy(), valid.bool().cpu().numpy()
    if not mask.any():
        return prediction.detach().float()
    depth, _ = align_inverse_depth(values, truth, mask)
    return torch.as_tensor(depth, dtype=torch.float32, device=prediction.device)


def loss(output, data, settings):
    from explore_a.training.losses import LossTerms, masked_mean
    from explore_a.training.targets import fusion_targets
    from explore_a.training.correction import correction_losses

    fixed = gauge(data)
    main = main_loss(output["depth"], fixed)
    # This GT-derived native-field target is loss-local, never attached to FusionInputs.
    supervision = {**data, "gt": fixed["target"],
                   "gt_valid": fixed["valid"] & (fixed["target"] >= 0)}
    blend, reset = main*0, main*0
    if "weight_logit" in output:
        targets = fusion_targets(output, supervision, settings)
        blend = masked_mean(F.binary_cross_entropy_with_logits(
            output["weight_logit"], targets.blend, reduction="none"), targets.blend_valid)
        reset = masked_mean(F.binary_cross_entropy_with_logits(
            output["reset_logit"], targets.reset, reduction="none"), targets.reset_valid)
    penalty = masked_mean(output["correction"].square(), fixed["valid"])
    global_loss, local_loss = correction_losses(output, supervision, settings)
    total = (main + settings["blend_aux_weight"]*blend + settings["reset_aux_weight"]*reset
             + settings["correction_penalty"]*penalty
             + settings.get("global_correction_weight", .01)*global_loss
             + settings.get("local_correction_weight", .01)*local_loss)
    present = fixed["valid"].flatten(1).any(1)
    fallback = (fixed["fallback"].flatten() & present).float().sum()/present.sum().clamp_min(1)
    return LossTerms(total, main, blend, reset, penalty, global_loss, local_loss,
                     gauge_fallback=fallback, gt_coverage=fixed["valid"].float().mean())
