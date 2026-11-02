"""Explicit-checkpoint comparison with one shared frozen frontend and one final model."""
from pathlib import Path

import torch

from explore_a.evaluation.branches import specification
from explore_a.evaluation.engine import evaluate_specs
from explore_a.models.final_flow import ABLATIONS


def checkpoint_spec(fusion, checkpoint, **options):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    task = state['identity']['experiment']
    if task['fusion'] != fusion:
        raise ValueError(f"Expected {fusion} checkpoint: {checkpoint}")
    spec = specification(task['flow'], task['occlusion'], fusion, **options)
    spec['checkpoint'] = Path(checkpoint)
    return spec


def compare_pair(config, guided, final, split="dev", output=None, device="cuda",
                 ablations=False):
    from explore_a.data.prepare import load_manifest
    from explore_a.training.dev_subset import select_dev
    rows = [row for row in load_manifest(config)["samples"] if row["split"] == split]
    if split == "dev":
        rows = select_dev(rows, config["training"]["dev_sample_fraction"], config["data"]["seed"])
    specs = []
    for fusion, checkpoint in (("unet_guided", guided), ("final_flow", final)):
        spec = checkpoint_spec(fusion, checkpoint)
        specs.append(spec)
    if ablations:
        for mode in ABLATIONS[1:]:
            spec = checkpoint_spec("final_flow", final, ablation=mode)
            specs.append(spec)
    specs.append(specification("farneback", baseline="raw"))
    return evaluate_specs(config, specs, split, device, output, rows=rows)


def compare(config, guided, refine, final, split="test", output=None, device="cuda"):
    specs = []
    for fusion, checkpoint in (("unet_guided", guided), ("flow_refine", refine),
                               ("final_flow", final)):
        spec = checkpoint_spec(fusion, checkpoint)
        specs.append(spec)
    for ablation in ABLATIONS[1:]:
        spec = checkpoint_spec("final_flow", final, ablation=ablation)
        specs.append(spec)
    specs.append(specification("farneback", baseline="raw"))
    return evaluate_specs(config, specs, split, device, output)
