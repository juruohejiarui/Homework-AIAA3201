"""Bidirectional training on exactly the same two frames; no recurrent state or new I/O."""
from dataclasses import fields

import torch
import torch.nn.functional as functional

from explore_a.geometry.warp import warp
from explore_a.models.final_flow import FinalFlow
from explore_a.training.losses import LossTerms, fusion_loss
from explore_a.training.prediction import predict


def reverse_pair(data):
    reverse = dict(data)
    for current, previous in (("current_rgb", "previous_rgb"),
                              ("current_depth", "previous_depth"),
                              ("gt", "previous_gt"), ("gt_valid", "previous_gt_valid"),
                              ("forward", "backward"), ("pose", "previous_pose")):
        if current in data and previous in data:
            reverse[current], reverse[previous] = data[previous], data[current]
    reverse["image_valid"] = data.get("previous_image_valid", data["image_valid"])
    reverse["current_image_valid"] = reverse["image_valid"]
    reverse["previous_image_valid"] = data["image_valid"]
    # Visibility is directional. Reverse the frozen visibility by sampling, not by copying.
    if "teacher_visibility" in data:
        visibility, bounds = warp(data["teacher_visibility"], data["forward"])
        reverse["teacher_visibility"] = torch.where(bounds, visibility.clamp(0, 1), 0)
    return reverse


def temporal_direction(current, previous, data, geometry, relative=False):
    if relative:
        return relative_temporal(current, previous, data, geometry)
    valid_current = data["gt_valid"].bool() & torch.isfinite(data["gt"]) & (data["gt"] > 0)
    valid_previous = (data["previous_gt_valid"].bool()
                      & torch.isfinite(data["previous_gt"]) & (data["previous_gt"] > 0))
    previous_valid, _ = warp(valid_previous.float(), data["backward"].detach())
    valid = geometry["rule"] & valid_current & (previous_valid > .999)
    prior_log, _ = warp(previous["depth"].float().clamp_min(1e-6).log(),
                        data["backward"].detach())
    # Sanitize before log/warp: invalid GT must not contaminate neighboring valid samples.
    with torch.no_grad():
        gt = torch.where(valid_current, data["gt"], 1).float().log()
        prior_gt = torch.where(valid_previous, data["previous_gt"], 1).float().log()
        prior_gt, _ = warp(prior_gt, data["backward"].detach())
        target = gt - prior_gt
    delta = current["depth"].float().clamp_min(1e-6).log() - prior_log
    error = functional.smooth_l1_loss(delta, target, beta=.1, reduction="none")
    return torch.where(valid, error, 0).sum() / valid.sum().clamp_min(1)


def relative_temporal(current, previous, data, geometry):
    from explore_a.training.relative import gauge
    now, before = gauge(data), gauge(reverse_pair(data))
    sampled, _ = warp(before["valid"].float(), data["backward"].detach())
    valid = now["valid"] & geometry["rule"] & (sampled > .999)
    predicted_before = before["a"]*previous["depth"].float()+before["b"]
    predicted_before, _ = warp(predicted_before, data["backward"].detach())
    gt_before, _ = warp(before["inverse"], data["backward"].detach())
    delta = now["a"]*current["depth"].float()+now["b"]-predicted_before
    target = now["inverse"]-gt_before
    scale = (.5*(now["scale"]+before["scale"])).clamp_min(1e-6)
    error = functional.smooth_l1_loss((delta-target)/scale, torch.zeros_like(delta),
                                     beta=.1, reduction="none")
    return torch.where(valid, error, 0).sum()/valid.sum().clamp_min(1)


def supervised(model, data, config, occlusion, step):
    output, geometry = predict(model, data, config, occlusion)
    terms = fusion_loss(output, data, config["training"])
    if not isinstance(model, FinalFlow):
        return output, terms
    reverse = reverse_pair(data)
    backward, backward_geometry = predict(model, reverse, config, occlusion)
    backward_terms = fusion_loss(backward, reverse, config["training"])
    # A missing GT endpoint contributes neither a loss nor a denominator of two.
    present = data["gt_valid"].flatten(1).any(1).float().sum()
    before = reverse["gt_valid"].flatten(1).any(1).float().sum()
    denominator = (present + before).clamp_min(1)
    averaged = {field.name: (getattr(terms, field.name) * present
                            + getattr(backward_terms, field.name) * before) / denominator
                for field in fields(LossTerms) if field.name != "temporal"
                and getattr(terms, field.name) is not None}
    from explore_a.frontend import relative
    temporal = (temporal_direction(output, backward, data, geometry, relative(config))
                + temporal_direction(backward, output, reverse, backward_geometry,
                                     relative(config))) * .5
    warmup = config["training"].get("temporal_warmup_steps", 500)
    weight = config["training"].get("temporal_weight", .05) * min(1., step / max(warmup, 1))
    averaged["total"] = averaged["total"] + weight * temporal
    averaged["temporal"] = temporal
    return output, LossTerms(**averaged)
