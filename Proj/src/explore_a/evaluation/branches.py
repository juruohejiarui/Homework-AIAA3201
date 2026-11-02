"""Load controlled experiment heads and validate their online input provenance."""
from itertools import product
from pathlib import Path
import copy
import torch

from explore_a.config import FLOWS, FUSIONS, OCCLUSIONS, experiment_name
from explore_a.training.checkpoint import load_model_state
from explore_a.training.prediction import make_fusion


def specification(flow, occlusion="rule", fusion="unet_guided", baseline=None,
                  project_interval=False, global_only=False, ablation="full"):
    from explore_a.models.final_flow import ABLATIONS
    if ablation not in ABLATIONS or (ablation != "full" and
            (fusion != "final_flow" or baseline or project_interval or global_only)):
        raise ValueError("Ablations require final_flow without other output interventions")
    if global_only and (baseline or fusion != "flow_refine" or project_interval):
        raise ValueError("Global-only requires flow_refine and cannot combine with projection")
    name = f"baseline-{flow}-{baseline}" if baseline else experiment_name(flow, occlusion, fusion)
    if project_interval:
        if baseline or fusion != "flow_refine":
            raise ValueError("Interval projection requires flow_refine without a baseline")
        name += "-projected"
    if global_only:
        name += "-global-only"
    if ablation != "full":
        name += "-" + ablation
    return {"name": name, "project_interval": project_interval, "global_only": global_only,
            "flow": flow,
            "occlusion": occlusion, "fusion": fusion, "baseline": baseline,
            "ablation": ablation}


def matrix_specs():
    specs = [specification("farneback", "rule", fusion) for fusion in FUSIONS]
    specs.append(specification("farneback", baseline="raw"))
    return specs


def load_branch(config, spec, device):
    branch = {**spec, "model": None}
    if spec["baseline"]:
        return branch
    model_name = experiment_name(spec["flow"], spec["occlusion"], spec["fusion"])
    path = spec.get("checkpoint") or Path(config["paths"]["runs"]) / model_name / "best.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    identity = state["identity"]
    from explore_a.training.identity import check_frontend
    check_frontend(identity, config)
    expected = {key: spec[key] for key in ("flow", "occlusion", "fusion")}
    if identity["experiment"] != expected:
        raise ValueError(f"Checkpoint model/task differs: {path}")
    inference_config = copy.deepcopy(config)
    inference_config["training"].update(identity.get("training", {}))
    model = make_fusion(spec["fusion"], inference_config).to(device).eval()
    load_model_state(model, state["model"])
    branch["model"] = model
    branch["checkpoint"] = str(Path(path).resolve())
    branch["checkpoint_step"] = state.get("step")
    return branch
