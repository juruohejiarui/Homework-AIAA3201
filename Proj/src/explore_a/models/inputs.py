"""GT-free model inputs and common prediction fields for all fusion heads."""
from dataclasses import dataclass
from typing import NotRequired, TypedDict

import torch
from torch import Tensor


@dataclass(frozen=True)
class FusionInputs:
    current_rgb: Tensor
    previous_rgb: Tensor
    warped_rgb: Tensor
    current_depth: Tensor
    previous_depth: Tensor
    historical: Tensor
    forward: Tensor
    backward: Tensor
    hard: Tensor
    fb_mask: Tensor
    availability: Tensor
    fb_residual: Tensor
    photo_residual: Tensor
    current_valid: Tensor
    prediction_kind: str = "metric_m"
    center: Tensor | None = None
    scale: Tensor | None = None

    @classmethod
    def from_pair(cls, data, geometry, visibility):
        availability = geometry["hard"] & geometry["rule"]
        return cls(
            current_rgb=geometry["current_rgb"], previous_rgb=geometry["previous_rgb"],
            warped_rgb=geometry["warped_rgb"], current_depth=data["current_depth"].float(),
            previous_depth=geometry.get("previous_field", data["previous_depth"]).float(),
            historical=geometry["historical"].float(),
            forward=data["forward"].float(), backward=data["backward"].float(),
            hard=geometry["hard"], fb_mask=geometry["rule"],
            availability=availability.float() * visibility.float().clamp(0, 1),
            fb_residual=geometry["fb_residual"], photo_residual=geometry["photo_residual"],
            current_valid=data["image_valid"].bool() & (
                data["current_depth"] >= 0 if geometry.get("prediction_kind") == "relative_inverse"
                else data["current_depth"] > 0),
            prediction_kind=geometry.get("prediction_kind", "metric_m"),
            center=geometry.get("field_center"), scale=geometry.get("field_scale"),
        )

    def scalar(self, value, epsilon=1e-6):
        if self.prediction_kind == "relative_inverse":
            return (value.float()-self.center)/self.scale
        return value.clamp_min(epsilon).log()

    def features(self, epsilon=1e-6):
        current = self.scalar(self.current_depth, epsilon)
        historical = self.scalar(self.historical, epsilon)
        historical = torch.where(self.hard, historical, current)
        scale = max(current.shape[-2:])
        return torch.cat([self.current_rgb, self.warped_rgb, current, historical,
                          (current - historical).abs(), self.availability,
                          self.fb_residual / scale, self.photo_residual], 1)


class FusionOutput(TypedDict):
    prediction_kind: str
    field_scale: NotRequired[Tensor]
    depth: Tensor
    historical_weight: Tensor
    visibility: Tensor
    mixed_depth: Tensor
    historical: Tensor
    correction: Tensor
    weight_logit: NotRequired[Tensor]
    reset_logit: NotRequired[Tensor]
    global_correction: NotRequired[Tensor]
    local_control: NotRequired[Tensor]
    local_correction: NotRequired[Tensor]
    spatial_global_logit: NotRequired[Tensor]
    spatial_local_logit: NotRequired[Tensor]
    temporal_global_increment: NotRequired[Tensor]
    temporal_local_increment: NotRequired[Tensor]
    temporal_global_gate: NotRequired[Tensor]
    temporal_local_gate: NotRequired[Tensor]
    spatial_correction: NotRequired[Tensor]
    temporal_correction: NotRequired[Tensor]
    temporal_current_features: NotRequired[Tensor]
    temporal_prior_features: NotRequired[Tensor]
    temporal_reference_features: NotRequired[Tensor]


def assemble(inputs, weight, correction=None, **aux) -> FusionOutput:
    current = inputs.current_depth
    valid = inputs.availability > 0
    mixed = torch.where(valid, (1 - weight) * current + weight * inputs.historical, current)
    correction = torch.zeros_like(mixed) if correction is None else correction.float()
    correction = torch.where(valid, correction, 0)
    corrected = ((mixed + inputs.scale * correction).clamp_min(0)
                 if inputs.prediction_kind == "relative_inverse" else mixed * (1 + correction))
    depth = torch.where(valid, corrected, current)
    return {"depth": depth, "mixed_depth": mixed, "historical": inputs.historical,
            "historical_weight": weight, "visibility": inputs.availability,
            "correction": correction, "prediction_kind": inputs.prediction_kind,
            **({"field_scale": inputs.scale} if inputs.scale is not None else {}), **aux}
