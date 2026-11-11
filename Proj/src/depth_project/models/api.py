"""Shared differentiable model contract. Transport and scoring detach at their boundaries."""
from dataclasses import dataclass, field
from importlib import import_module

import numpy as np
import torch


@dataclass
class ModelInput:
    rgb: object = None
    frame_ids: tuple = ()
    timestamps: tuple = ()
    reset_history: tuple = ()
    specific_in: dict = field(default_factory=dict)


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    representation: str
    frame_ids: tuple = ()
    specific_out: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.prediction, torch.Tensor):
            raise TypeError("Model prediction must be a Tensor; detach only at I/O boundaries")
        if self.representation not in ("relative_inverse", "metric_m"):
            raise ValueError("Unknown depth representation")

    def numpy(self):
        return self.prediction.detach().float().cpu().numpy()


def frame_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
        if value.ndim == 3 and value.shape[0] == 3:
            value = value.transpose(1, 2, 0)
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("RGB frames must be uint8 HWC (or Tensor CHW)")
    return np.ascontiguousarray(value)


class ImageAdapter:
    capability = "image"

    def forward(self, request, **specific_param):
        if specific_param or request.specific_in:
            raise ValueError(f"Unsupported model-specific inputs/parameters: {specific_param.keys()}")
        frames = request.rgb
        if isinstance(frames, (np.ndarray, torch.Tensor)) and frames.ndim == 3:
            frames = [frames]
        values = [torch.as_tensor(self.predict(frame_array(rgb)), dtype=torch.float32)
                  for rgb in frames]
        # Frozen training submits one frame at a time. Stacking a single CPU map launches
        # a parallel copy for every prediction; a batch-axis view preserves the same graph.
        prediction = values[0].unsqueeze(0) if len(values) == 1 else torch.stack(values)
        return ModelOutput(prediction, self.kind, request.frame_ids)


def fusion_forward(model, request, **specific_param):
    if specific_param:
        raise ValueError(f"Unsupported fusion forward parameters: {specific_param.keys()}")
    inputs = request.specific_in["fusion_inputs"]
    output = model(inputs)
    return ModelOutput(output["depth"].float(), output["prediction_kind"], request.frame_ids,
                       {k: v for k, v in output.items() if k not in ("depth", "prediction_kind")})


def resolve_adapter(name):
    """Third-party adapters implement forward(ModelInput, **specific_param)."""
    module, symbol = name.split(":", 1)
    return getattr(import_module(module), symbol)
