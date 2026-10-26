"""Confidence gives relative mixture weights, not calibrated depth-correctness probabilities."""
import torch
from torch import nn

from explore_a.geometry.features import fusion_features
from explore_a.models.blocks import SmallUNet


class Fusion(nn.Module):
    def __init__(self, architecture, residual_fraction=.1, epsilon=1e-6):
        super().__init__()
        self.architecture = architecture
        self.residual_fraction, self.epsilon = residual_fraction, epsilon
        if architecture == "cnn":
            layers = []
            for inputs in (12, 32, 32):
                layers.extend([nn.Conv2d(inputs, 32, 3, padding=1), nn.ReLU()])
            self.network = nn.Sequential(*layers, nn.Conv2d(32, 2, 1))
        elif architecture in ("unet", "residual"):
            self.network = SmallUNet(12, 3 if architecture == "residual" else 2)
        else:
            raise ValueError(architecture)

    def forward(self, data, geometry, visibility):
        availability = geometry["hard"].float() * visibility.float().clamp(0, 1)
        features = fusion_features(data, geometry, availability, self.epsilon)
        output = self.network(features)
        # Log-space normalization avoids a 0/0 when history is absent and logits saturate.
        logits = output[:, :2].float()
        # log(0)=-inf gives exactly zero history weight; positive soft masks stay unmodified.
        historical_logit = logits[:, 1:2] + availability.log()
        weights = torch.softmax(torch.cat([logits[:, :1], historical_logit], 1), 1)
        weight = torch.where(availability > 0, weights[:, 1:2], 0)
        current = data["current_depth"].float()
        depth = (1 - weight) * current + weight * geometry["historical"]
        correction = torch.zeros_like(depth)
        if self.architecture == "residual":
            correction = self.residual_fraction * output[:, 2:3].float().tanh()
            # All variants, including the residual variant, exactly fall back without history.
            correction = correction * (availability > 0)
            depth = depth * (1 + correction)
        return {"depth": depth.clamp_min(self.epsilon), "historical_weight": weight,
                "confidence": logits.softmax(1), "residual_fraction": correction,
                "visibility": availability}
