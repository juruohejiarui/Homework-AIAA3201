"""Supervised blend/reset heads, inspired by Khan et al. and CODD fusion modules."""
import math

import torch
from torch import nn
import torch.nn.functional as functional

from explore_a.models.blocks import SmallUNet
from explore_a.models.inputs import FusionInputs, assemble


def initialize_gates(head):
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    with torch.no_grad():
        head.bias[0] = math.log(.05 / .95) - math.log(.9)
        head.bias[1] = math.log(.9 / .1)


def guided_output(inputs, logits, correction=None):
    blend, reset = logits[:, :1].float(), logits[:, 1:2].float()
    # Excluded pixels have no BCE supervision, so their effective logit can remain finite.
    available = inputs.availability > 0
    prior = torch.where(available, inputs.availability, 1)
    effective = blend + prior.log() + functional.logsigmoid(reset)
    weight = torch.where(available, effective.sigmoid(), 0)
    return assemble(inputs, weight, correction, weight_logit=effective, reset_logit=reset)


class GuidedUNet(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.network = SmallUNet(12, 2)
        initialize_gates(self.network.head)

    def forward(self, inputs: FusionInputs):
        return guided_output(inputs, self.network(inputs.features(self.epsilon)))
