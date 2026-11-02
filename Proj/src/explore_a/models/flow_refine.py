"""Pair-local FlowGRU-inspired feature aggregation with bounded depth repair."""
import torch
from torch import nn
import torch.nn.functional as functional

from explore_a.geometry.warp import warp
from explore_a.models.blocks import double_conv
from explore_a.models.correction import CorrectionContext, bounded_correction
from explore_a.models.guided import guided_output, initialize_gates
from explore_a.models.inputs import FusionInputs


def resize(value, size):
    return functional.interpolate(value, size=size, mode="bilinear", align_corners=False)


def align_features(previous, backward, valid):
    height, width = backward.shape[-2:]
    small_h, small_w = previous.shape[-2:]
    # Flow is in canvas pixels; resampling alone does not convert its displacement units.
    flow = resize(backward.float(), (small_h, small_w))
    flow = flow * flow.new_tensor([small_w / width, small_h / height])[None, :, None, None]
    aligned, bounds = warp(previous, flow)
    mask = functional.adaptive_avg_pool2d(valid.float(), (small_h, small_w)) > .999
    return torch.where(mask & bounds, aligned, 0)


class PairGRU(nn.Module):
    def __init__(self, inputs=69, hidden=64):
        super().__init__()
        self.gates = nn.Conv2d(inputs + hidden, hidden * 2, 3, padding=1)
        self.candidate = nn.Conv2d(inputs + hidden, hidden, 3, padding=1)

    def forward(self, current, prior):
        reset, update = self.gates(torch.cat([current, prior], 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat([current, reset * prior], 1)).tanh()
        return (1 - update) * prior + update * candidate


class FlowRefine(nn.Module):
    def __init__(self, correction_fraction=.1, epsilon=1e-6):
        super().__init__()
        self.correction_fraction, self.epsilon = correction_fraction, epsilon
        self.encoder = nn.ModuleList([double_conv(4, 16), double_conv(16, 32),
                                      double_conv(32, 64)])
        self.memory = PairGRU()
        self.decode4 = double_conv(96, 32)
        self.decode2 = double_conv(48, 16)
        self.decode1 = double_conv(20, 16)
        self.gate_head = nn.Conv2d(16, 2, 1)
        self.local_head = nn.Conv2d(16, 1, 1)
        self.correction_context = CorrectionContext()
        initialize_gates(self.gate_head)
        for head in (self.local_head,):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def encode(self, rgb, depth, scalar):
        value = torch.cat([rgb, scalar(depth, self.epsilon)], 1)
        levels = []
        for block in self.encoder:
            value = block(functional.avg_pool2d(value, 2, ceil_mode=True))
            levels.append(value)
        return levels

    def forward(self, inputs: FusionInputs):
        current = self.encode(inputs.current_rgb, inputs.current_depth, inputs.scalar)
        previous = self.encode(inputs.previous_rgb, inputs.previous_depth, inputs.scalar)
        prior = align_features(previous[-1], inputs.backward, inputs.availability > 0)
        log_current = inputs.scalar(inputs.current_depth, self.epsilon)
        log_history = inputs.scalar(inputs.historical, self.epsilon)
        valid = inputs.availability > 0
        delta = torch.where(valid, log_current - log_history, 0)
        cues = torch.cat([delta, delta.abs(), inputs.availability,
                          inputs.fb_residual / max(delta.shape[-2:]), inputs.photo_residual], 1)
        value = torch.cat([current[-1], resize(cues, current[-1].shape[-2:])], 1)
        # State lives only in this call. Random training pairs cannot leak across scenes.
        memory = self.memory(value, prior)
        up = self.decode4(torch.cat([resize(memory, current[1].shape[-2:]), current[1]], 1))
        up = self.decode2(torch.cat([resize(up, current[0].shape[-2:]), current[0]], 1))
        full = torch.cat([inputs.current_rgb, log_current], 1)
        up = self.decode1(torch.cat([resize(up, full.shape[-2:]), full], 1))
        global_value, local_features = self.correction_context(
            memory, up, valid, self.correction_fraction)
        control = self.local_head(local_features).float().tanh()
        correction = bounded_correction(global_value, control, self.correction_fraction)
        output = guided_output(inputs, self.gate_head(up), correction)
        output.update(global_correction=global_value, local_control=control,
                      local_correction=torch.where(valid, correction - global_value, 0))
        return output
