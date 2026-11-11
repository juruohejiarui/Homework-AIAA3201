"""One GT-free prediction path shared by training and evaluation."""
import torch

from explore_a.geometry.warp import pair_geometry
from explore_a.models.flow_refine import FlowRefine
from explore_a.models.final_flow import FinalFlow
from explore_a.models.guided import GuidedUNet
from explore_a.models.inputs import FusionInputs


def make_fusion(name, config):
    epsilon = config["geometry"]["epsilon"]
    if name == "unet_guided":
        return GuidedUNet(epsilon)
    if name == "flow_refine":
        return FlowRefine(config["training"]["correction_fraction"], epsilon)
    if name == "final_flow":
        return FinalFlow(config["training"]["correction_fraction"], epsilon)
    raise ValueError(f"Unknown fusion model: {name}")


def predict(model, data, config, occlusion, geometry=None):
    geometry = fusion_geometry(data, config) if geometry is None else geometry
    if occlusion == "rule":
        visibility = torch.ones_like(data["current_depth"])
    elif occlusion == "mftiq":
        visibility = data["teacher_visibility"]
    else:
        raise ValueError(f"Unknown occlusion strategy: {occlusion}")
    inputs = FusionInputs.from_pair(data, geometry, visibility)
    from depth_project.models.api import ModelInput, fusion_forward
    result = fusion_forward(model, ModelInput(specific_in={"fusion_inputs": inputs}))
    return {"depth": result.prediction, "prediction_kind": result.representation,
            **result.specific_out}, geometry


def fusion_geometry(data, config):
    from explore_a.frontend import relative
    if not relative(config):
        return pair_geometry(data, config["geometry"])
    from explore_a.geometry.relative import affine, statistics
    geometry = pair_geometry(data, {**config["geometry"], "relative_inverse": True})
    center, scale = statistics(data["current_depth"], data["image_valid"])
    a, b, accepted = affine(geometry["historical"], data["current_depth"], geometry["rule"])
    history = a*geometry["historical"]+b
    valid = accepted & torch.isfinite(history) & (history >= 0)
    geometry.update(prediction_kind="relative_inverse", field_center=center, field_scale=scale,
                    alignment_scale=a, alignment_shift=b, alignment_accepted=accepted,
                    unaligned_historical=geometry["historical"],
                    unaligned_rule=geometry["rule"],
                    previous_field=torch.where(torch.isfinite(a*data["previous_depth"]+b),
                                               a*data["previous_depth"]+b, 0).clamp_min(0),
                    historical=torch.where(valid, history, 0),
                    hard=geometry["hard"] & valid, rule=geometry["rule"] & valid)
    return geometry


def project_to_interval(output, data):
    """Inference-only ablation: projection needs no GT or additional model execution."""
    lower = torch.minimum(data["current_depth"], output["historical"])
    upper = torch.maximum(data["current_depth"], output["historical"])
    depth = torch.maximum(lower, torch.minimum(output["depth"], upper))
    depth = torch.where(output["visibility"] > 0, depth, data["current_depth"])
    return {**output, "depth": depth}


def global_only(output, data):
    """Inference-only intervention on the same forward, with unchanged blend weights."""
    valid = output["visibility"] > 0
    correction = torch.where(valid, output["global_correction"], 0)
    depth = torch.where(valid, (output["mixed_depth"] + output["field_scale"]
                                * correction).clamp_min(0), data["current_depth"])
    return {**output, "depth": depth, "correction": correction,
            "local_correction": torch.zeros_like(correction)}
