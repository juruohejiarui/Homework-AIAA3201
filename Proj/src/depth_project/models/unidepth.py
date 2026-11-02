"""Official UniDepthV2 ViT-S RGB-only inference, with fully local model loading."""
import json
import inspect
import sys

import numpy as np
import torch

from depth_project.paths import PROJECT, THIRD_PARTY

REPOSITORY = "lpiccinelli/unidepth-v2-vits14"
REVISION = "038c238f06c87b6c2f5b3749fd51fbf442b1f218"
DIRECTORY = "unidepth-v2-vits14"


def prepare_imports():
    sys.path.insert(0, str(THIRD_PARTY / "UniDepth"))
    # Optional upstream visualization imports wandb, even for RGB-only inference.
    # Keep those dependencies separate from the user's active training environment.
    dependencies = PROJECT / ".runtime/unidepth-deps"
    if dependencies.is_dir():
        sys.path.append(str(dependencies))


def build_local(weights):
    """Do not fetch redundant DINO weights or construct unused training losses."""
    from safetensors.torch import load_file

    prepare_imports()
    from unidepth.models import UniDepthV2

    config = json.loads((weights / "config.json").read_text())
    config["model"]["pixel_encoder"]["pretrained"] = None
    config["training"]["losses"] = {}
    model = UniDepthV2(config)
    model.load_state_dict(load_file(str(weights / "model.safetensors")), strict=True)
    # Fix the official inference resolution setting rather than relying on its warning default.
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"
    return model.eval().requires_grad_(False)


class UniDepthAdapter:
    kind = "metric_m"

    def __init__(self, weights, fp32=True):
        from pathlib import Path

        self.model = build_local(Path(weights)).cuda()
        self.fp32 = fp32
        # Upstream infer has an unconditional FP16 decorator. Unwrap only decorators,
        # keeping its body/preprocessing unchanged so our precision flag is truthful.
        self.infer = inspect.unwrap(type(self.model).infer)

    @torch.inference_mode()
    def predict(self, rgb):
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).cuda()
        with torch.autocast("cuda", dtype=torch.float16, enabled=not self.fp32):
            result = self.infer(self.model, image)["depth"]
        return result[0, 0].float().cpu().numpy()
