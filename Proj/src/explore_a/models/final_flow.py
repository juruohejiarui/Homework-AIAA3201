"""Strictly two-frame correction: current-only S plus anchored temporal increments T."""
import torch
from torch import nn
import torch.nn.functional as functional

from explore_a.models.blocks import double_conv
from explore_a.models.correction import CorrectionContext, bounded_correction
from explore_a.models.flow_refine import align_features, resize
from explore_a.models.guided import guided_output, initialize_gates

ABLATIONS = ("full", "no-temporal", "global-only", "fusion-only", "shuffled-history")


class FinalFlow(nn.Module):
    def __init__(self, correction_fraction=.1, epsilon=1e-6):
        super().__init__()
        self.correction_fraction, self.epsilon = correction_fraction, epsilon
        self.encoder = nn.ModuleList([double_conv(4, 16), double_conv(16, 32),
                                      double_conv(32, 64)])
        self.decode4 = double_conv(96, 32)
        self.decode2 = double_conv(48, 16)
        self.decode1 = double_conv(20, 16)
        self.context = CorrectionContext()
        self.spatial_local = nn.Conv2d(16, 1, 1)
        # 3*32 feature channels + log delta/abs + flow(2) + A + FB + photo = 103.
        self.temporal = double_conv(103, 32)
        self.fusion_head = nn.Conv2d(32, 2, 1)
        self.increment_global = nn.Linear(32, 1, bias=False)
        self.increment_local = nn.Conv2d(32, 1, 1, bias=False)
        self.reliability_global = nn.Linear(32, 1)
        self.reliability_local = nn.Conv2d(32, 1, 1)
        initialize_gates(self.fusion_head)
        for layer in (self.spatial_local, self.increment_global, self.increment_local,
                      self.reliability_global, self.reliability_local):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def encode(self, rgb, depth, scalar=None):
        value = torch.cat([rgb, depth.clamp_min(self.epsilon).log()
                           if scalar is None else scalar(depth, self.epsilon)], 1)
        levels = []
        for block in self.encoder:
            value = block(functional.avg_pool2d(value, 2, ceil_mode=True))
            levels.append(value)
        return levels

    @staticmethod
    def pool(features, mask):
        coverage = functional.adaptive_avg_pool2d(mask.float(), features.shape[-2:])
        return ((features.float() * coverage).sum((-2, -1))
                / coverage.sum((-2, -1)).clamp_min(1e-6))

    def spatial(self, current, inputs):
        up = self.decode4(torch.cat([resize(current[2], current[1].shape[-2:]), current[1]], 1))
        up = self.decode2(torch.cat([resize(up, current[0].shape[-2:]), current[0]], 1))
        log_depth = inputs.scalar(inputs.current_depth, self.epsilon)
        full = torch.cat([inputs.current_rgb, log_depth], 1)
        up = self.decode1(torch.cat([resize(up, full.shape[-2:]), full], 1))
        # S pools only current-frame support, never history availability or temporal cues.
        support = inputs.current_valid
        context = self.context.context(self.pool(current[2], support))
        global_logit = self.context.global_head(context).float()[..., None, None]
        scale, bias = self.context.film(context).float().chunk(2, 1)
        up = up.float() * (1 + .1 * scale.tanh()[..., None, None])
        up = up + .1 * bias.tanh()[..., None, None]
        return global_logit, self.spatial_local(up).float()

    def temporal_features(self, current, prior, inputs, self_reference=False):
        size = current.shape[-2:]
        valid = functional.adaptive_avg_pool2d((inputs.availability > 0).float(), size) > .999
        difference = torch.where(valid, current - prior, 0)
        delta = torch.where(inputs.availability > 0,
                            inputs.scalar(inputs.current_depth, self.epsilon)
                            - inputs.scalar(inputs.historical, self.epsilon), 0)
        flow = inputs.backward.float() / max(delta.shape[-2:])
        fb = inputs.fb_residual / max(delta.shape[-2:])
        photo = inputs.photo_residual
        if self_reference:
            prior = align_features(current, torch.zeros_like(inputs.backward),
                                   inputs.availability > 0)
            difference = torch.zeros_like(current)
            delta = torch.zeros_like(delta)
            flow, fb, photo = torch.zeros_like(flow), torch.zeros_like(fb), torch.zeros_like(photo)
        cues = torch.cat([delta, delta.abs(), flow, inputs.availability, fb, photo], 1)
        return self.temporal(torch.cat([current, prior, difference, resize(cues, size)], 1))

    def forward(self, inputs):
        current = self.encode(inputs.current_rgb, inputs.current_depth, inputs.scalar)
        previous = self.encode(inputs.previous_rgb, inputs.previous_depth, inputs.scalar)
        prior = align_features(previous[1], inputs.backward, inputs.availability > 0)
        real = self.temporal_features(current[1], prior, inputs)
        reference = self.temporal_features(current[1], current[1], inputs, True)
        pooled = self.pool(real, inputs.availability > 0)
        reference_pooled = self.pool(reference, inputs.availability > 0)
        dg = (self.increment_global(pooled) - self.increment_global(reference_pooled)).float()
        dl = (self.increment_local(real) - self.increment_local(reference)).float()
        rg = self.reliability_global(pooled).float().sigmoid()[..., None, None]
        rl = resize(self.reliability_local(real).float().sigmoid(), inputs.current_depth.shape[-2:])
        sg, sl = self.spatial(current, inputs)
        size = inputs.current_depth.shape[-2:]
        dg, dl = dg[..., None, None], resize(dl, size)
        global_value = self.correction_fraction * (sg + rg * dg).tanh()
        correction = bounded_correction(global_value, (sl + rl * dl).tanh(),
                                        self.correction_fraction)
        logits = resize(self.fusion_head(real).float(), size)
        output = guided_output(inputs, logits, correction)
        spatial_correction = bounded_correction(self.correction_fraction * sg.tanh(), sl.tanh(),
                                                self.correction_fraction)
        spatial_correction = torch.where(inputs.availability > 0, spatial_correction, 0)
        output.update(global_correction=global_value, local_control=(sl + rl * dl).tanh(),
                      local_correction=torch.where(inputs.availability > 0,
                                                   correction - global_value, 0),
                      spatial_global_logit=sg, spatial_local_logit=sl,
                      spatial_correction=spatial_correction,
                      temporal_correction=output["correction"] - spatial_correction,
                      temporal_global_increment=dg, temporal_local_increment=dl,
                      temporal_global_gate=rg, temporal_local_gate=rl)
        # Retained only for same-forward evaluation interventions, never cached or transmitted.
        output["temporal_current_features"] = current[1]
        output["temporal_prior_features"] = prior
        output["temporal_reference_features"] = reference
        return output

    def ablate(self, output, inputs, mode):
        if mode not in ABLATIONS:
            raise ValueError(f"Unknown final_flow ablation: {mode}")
        if mode == "full":
            return output
        sg, sl = output["spatial_global_logit"], output["spatial_local_logit"]
        dg, dl = output["temporal_global_increment"], output["temporal_local_increment"]
        rg, rl = output["temporal_global_gate"], output["temporal_local_gate"]
        if mode == "no-temporal":
            dg, dl = torch.zeros_like(dg), torch.zeros_like(dl)
        elif mode == "shuffled-history":
            prior = output["temporal_prior_features"]
            prior = torch.roll(prior, (prior.shape[-2] // 2, prior.shape[-1] // 2), (-2, -1))
            real = self.temporal_features(output["temporal_current_features"], prior, inputs)
            ref = output["temporal_reference_features"]
            pooled = self.pool(real, inputs.availability > 0)
            ref_pool = self.pool(ref, inputs.availability > 0)
            dg = (self.increment_global(pooled) - self.increment_global(ref_pool)).float()
            dg = dg[..., None, None]
            dl = resize((self.increment_local(real) - self.increment_local(ref)).float(),
                        inputs.current_depth.shape[-2:])
            rg = self.reliability_global(pooled).float().sigmoid()[..., None, None]
            rl = resize(self.reliability_local(real).float().sigmoid(), dl.shape[-2:])
        global_value = self.correction_fraction * (sg + rg * dg).tanh()
        control = (sl + rl * dl).tanh()
        if mode == "global-only":
            control = torch.zeros_like(control)
        correction = bounded_correction(global_value, control, self.correction_fraction)
        if mode == "fusion-only":
            correction = torch.zeros_like(correction)
            global_value, control = torch.zeros_like(global_value), torch.zeros_like(control)
        valid = inputs.availability > 0
        correction = torch.where(valid, correction, 0)
        value = ((output["mixed_depth"]+inputs.scale*correction).clamp_min(0)
                 if inputs.prediction_kind == "relative_inverse"
                 else output["mixed_depth"] * (1 + correction))
        depth = torch.where(valid, value, inputs.current_depth)
        spatial = output["spatial_correction"]
        if mode == "global-only":
            spatial = torch.where(valid, self.correction_fraction * sg.tanh(), 0)
        elif mode == "fusion-only":
            spatial = torch.zeros_like(spatial)
        return {**output, "depth": depth, "correction": correction,
                "spatial_correction": spatial, "temporal_correction": correction - spatial,
                "temporal_global_increment": dg, "temporal_local_increment": dl,
                "temporal_global_gate": rg, "temporal_local_gate": rl,
                "global_correction": global_value, "local_control": control,
                "local_correction": torch.where(valid, correction - global_value, 0)}
