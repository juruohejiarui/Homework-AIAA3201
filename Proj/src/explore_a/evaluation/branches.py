"""Load controlled experiment heads and validate their online input provenance."""
from itertools import product
from pathlib import Path

from explore_a.config import FLOWS, FUSIONS, OCCLUSIONS, experiment_name, file_identity
from explore_a.online.identity import check_online, input_identity, teacher_identity
from explore_a.training.checkpoint import load_checkpoint
from explore_a.training.prediction import load_student, make_fusion


def specification(flow, occlusion="rule", fusion="unet", baseline=None):
    name = f"baseline-{flow}-{baseline}" if baseline else experiment_name(flow, occlusion, fusion)
    return {"name": name, "flow": "farneback" if baseline == "raw" else flow,
            "occlusion": occlusion, "fusion": fusion, "baseline": baseline}


def matrix_specs():
    specs = [specification(*values) for values in product(FLOWS, OCCLUSIONS, FUSIONS)]
    specs.append(specification("farneback", baseline="raw"))
    specs.extend(specification(flow, baseline=baseline)
                 for flow, baseline in product(FLOWS, ("fixed", "rule-fixed")))
    return specs


def load_branch(config, spec, device):
    branch = {**spec, "model": None, "student": None}
    if spec["baseline"]:
        return branch
    model = make_fusion(spec["fusion"], config).to(device).eval()
    path = Path(config["paths"]["runs"]) / spec["name"] / "best.pt"
    state = load_checkpoint(path, model)
    identity = state["identity"]
    check_online(identity)
    expected = {key: spec[key] for key in ("flow", "occlusion", "fusion")}
    if identity["experiment"] != expected or identity["input"] != input_identity(
            config, spec["flow"]):
        raise ValueError(f"Checkpoint input/experiment differs: {path}")
    if spec["occlusion"] == "mftiq" and identity.get("teacher") != teacher_identity(config):
        raise ValueError(f"Teacher provenance differs: {path}")
    if spec["occlusion"] == "student":
        branch["student"] = load_student(config, spec["flow"], device)
        student_path = Path(config["paths"]["runs"]) / f"occlusion-{spec['flow']}/best.pt"
        if identity["student"] != file_identity(student_path):
            raise ValueError("Student weights changed after fusion training")
    branch["model"] = model
    return branch
