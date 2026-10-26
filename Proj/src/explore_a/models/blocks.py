"""Small convolution blocks; GroupNorm supports microbatch training."""
import torch
from torch import nn
import torch.nn.functional as functional


def double_conv(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs),
                         nn.ReLU(), nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(4, outputs), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.enc1 = double_conv(inputs, 16)
        self.enc2 = double_conv(16, 32)
        self.enc3 = double_conv(32, 64)
        self.dec2 = double_conv(96, 32)
        self.dec1 = double_conv(48, 16)
        self.head = nn.Conv2d(16, outputs, 1)

    def forward(self, value):
        first = self.enc1(value)
        second = self.enc2(functional.avg_pool2d(first, 2))
        third = self.enc3(functional.avg_pool2d(second, 2))
        up = functional.interpolate(third, size=second.shape[-2:], mode="bilinear",
                                    align_corners=False)
        up = self.dec2(torch.cat([up, second], 1))
        up = functional.interpolate(up, size=first.shape[-2:], mode="bilinear",
                                    align_corners=False)
        return self.head(self.dec1(torch.cat([up, first], 1)))
