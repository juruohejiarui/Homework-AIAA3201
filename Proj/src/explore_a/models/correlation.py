"""Checkpoint-compatible kernel=1 local correlation, with an optional compiled backend."""
import torch
from torch import nn
import torch.nn.functional as functional

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler as CompiledSampler
except ImportError:
    CompiledSampler = None


class TorchCorrelation(nn.Module):
    def __init__(self, kernel_size=1, patch_size=7, stride=1, padding=0, dilation=1):
        super().__init__()
        if (kernel_size, stride, padding, dilation) != (1, 1, 0, 1) or patch_size % 2 != 1:
            raise ValueError("Fallback implements exactly MFTIQ's kernel=1, stride=1 correlation")
        self.patch_size = patch_size

    def forward(self, left, right):
        """Sum over channels, with dy-major/dx-minor offsets; output B,P,P,H,W."""
        radius = self.patch_size // 2
        height, width = left.shape[-2:]
        padded = functional.pad(right, (radius,) * 4)
        values = [(left * padded[:, :, dy:dy + height, dx:dx + width]).sum(1)
                  for dy in range(self.patch_size) for dx in range(self.patch_size)]
        return torch.stack(values, 1).reshape(left.shape[0], self.patch_size,
                                               self.patch_size, height, width)


SpatialCorrelationSampler = CompiledSampler or TorchCorrelation
