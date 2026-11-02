"""GT-free global context and a bounded, independently controlled spatial residual."""
import torch
from torch import nn
import torch.nn.functional as functional


def bounded_correction(global_value, local_control, fraction):
    # where, rather than two ReLUs, retains a local derivative at q=0 initialization.
    room = torch.where(local_control >= 0, fraction - global_value,
                       fraction + global_value)
    return global_value + room * local_control


class CorrectionContext(nn.Module):
    """Coverage pooling ignores padding without requiring every coarse pixel to be valid."""
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.global_head = nn.Linear(32, 1)
        self.film = nn.Linear(32, 32)
        for layer in (self.global_head, self.film):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, memory, decoded, available, fraction):
        coverage = functional.adaptive_avg_pool2d(available.float(), memory.shape[-2:])
        pooled = (memory.float() * coverage).sum((-2, -1))
        pooled = pooled / coverage.sum((-2, -1)).clamp_min(1e-6)
        context = self.context(pooled)
        global_value = fraction * self.global_head(context).float().tanh()
        present = available.flatten(1).any(1).reshape(-1, 1, 1, 1)
        global_value = torch.where(present, global_value[..., None, None], 0)
        scale, bias = self.film(context).float().chunk(2, 1)
        decoded = decoded.float() * (1 + .1 * scale.tanh()[..., None, None])
        decoded = decoded + .1 * bias.tanh()[..., None, None]
        return global_value, decoded
