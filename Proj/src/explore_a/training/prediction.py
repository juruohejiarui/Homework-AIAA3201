"""One common prediction path for train, development and final evaluation."""
from pathlib import Path

import torch

from explore_a.geometry.warp import pair_geometry
from explore_a.models.fusion import Fusion
from explore_a.models.occlusion import OcclusionStudent
from explore_a.training.checkpoint import load_checkpoint


def make_fusion(name, config):
    return Fusion(name, config["training"]["residual_fraction"], config["geometry"]["epsilon"])


def load_student(config, flow, device):
    student = OcclusionStudent().to(device)
    path = Path(config["paths"]["runs"]) / f"occlusion-{flow}" / "best.pt"
    state = load_checkpoint(path, student)
    expected = {"flow": flow, "occlusion": "distillation", "fusion": "student"}
    if state["identity"]["experiment"] != expected:
        raise ValueError(f"Student checkpoint does not match flow: {path}")
    from explore_a.online.identity import check_online, input_identity

    check_online(state["identity"])
    if state["identity"]["input"] != input_identity(config, flow):
        raise ValueError("Student was trained with different input/weight provenance")
    return student.eval().requires_grad_(False)


def visibility_for(data, geometry, occlusion, student=None):
    if occlusion == "rule":
        return geometry["rule"].float()
    if occlusion == "mftiq":
        return data["teacher_visibility"]
    if occlusion == "student":
        with torch.no_grad():
            return student(data, geometry).sigmoid()
    raise ValueError(occlusion)


def predict(model, data, config, occlusion, student=None):
    geometry = pair_geometry(data, config["geometry"])
    visibility = visibility_for(data, geometry, occlusion, student)
    return model(data, geometry, visibility), geometry
