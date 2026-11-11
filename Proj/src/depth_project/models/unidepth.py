"""Official UniDepthV2 ViT-S RGB-only inference, with fully local model loading."""
import json
import inspect
import sys

import numpy as np
import torch

from depth_project.paths import PROJECT, THIRD_PARTY
from depth_project.models.api import ImageAdapter

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


class UniDepthAdapter(ImageAdapter):
    runtime_paths = ("third_party/UniDepth",)
    kind = "metric_m"

    def forward(self, request, **specific_param):
        from depth_project.models.api import ModelOutput, frame_array
        level = specific_param.pop("resolution_level", self.model.resolution_level)
        if specific_param or set(request.specific_in) - {"camera"}:
            raise ValueError("Unsupported UniDepth inputs/parameters")
        frames = request.rgb
        if isinstance(frames, (np.ndarray, torch.Tensor)) and frames.ndim == 3:
            frames = [frames]
        outputs = []
        camera = request.specific_in.get("camera")
        if isinstance(camera, (np.ndarray, torch.Tensor, list, tuple)):
            camera = torch.as_tensor(camera, device=next(self.model.parameters()).device)
        previous = self.model.resolution_level
        self.model.resolution_level = level
        try:
            for rgb in frames:
                image = torch.from_numpy(frame_array(rgb).transpose(2, 0, 1).copy()).to(
                    next(self.model.parameters()).device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=not self.fp32):
                    outputs.append(self.infer(self.model, image,
                                              camera=camera))
        finally:
            self.model.resolution_level = previous
        return ModelOutput(torch.stack([r["depth"][0, 0].float() for r in outputs]), self.kind,
                           request.frame_ids, {"frames": [{k: v for k, v in r.items()
                                                           if k != "depth"} for r in outputs]})

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
