"""On-demand frozen inference; only one immediately preceding raw depth survives streaming."""
import time

import numpy as np
import torch

from depth_project.models.depthpro import DepthProAdapter
from explore_a.data.reading import fit_image
from explore_a.models.flow import PairFlow
from explore_a.models.mftiq import MFTIQVisibility
from explore_a.models.weights import check_weights
from explore_a.online.identity import required_weights
from explore_a.observability.progress import message


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class FrozenRuntime:
    def __init__(self, config, flows, teacher=False, device="cuda"):
        check_weights(config, required_weights(flows, teacher))
        self.device = device
        message("Loading frozen Depth Pro")
        self.depth = DepthProAdapter(config["weights"]["depthpro"], fp32=False)
        self.depth.model.eval().requires_grad_(False)
        self.flows = {}
        for flow in flows:
            message("Loading flow backend:", flow)
            self.flows[flow] = PairFlow(flow, config, device)
        if teacher:
            message("Loading frozen MFTIQ and feature backbones")
        self.teacher = MFTIQVisibility(config, device) if teacher else None
        message("Frozen frontends ready")
        self.reset()

    def reset(self):
        self.previous_key, self.previous_depth = None, None

    @torch.no_grad()
    def depth_pair(self, raw, ordered=False):
        row = raw["row"]
        sequence = (row["dataset"], row["sequence"], row["camera"])
        data = {key: value.to(self.device, non_blocking=True)
                for key, value in raw["data"].items()}
        if not ordered:
            self.reset()
        depths = {}
        for previous, prefix in ((True, "previous"), (False, "current")):
            key = (*sequence, row["previous_id" if previous else "current_id"])
            if previous and ordered and key == self.previous_key:
                value = self.previous_depth
            else:
                predicted = self.depth.predict(raw["images"][previous])
                # The shared adapter's focal history is useful in short standalone runs;
                # online training must not accumulate one Python object per prediction.
                if hasattr(self.depth, "focal_predictions"):
                    self.depth.focal_predictions.clear()
                if not np.isfinite(predicted).all() or (predicted <= 0).any():
                    raise ValueError(f"Invalid frozen depth: {row['id']}")
                fitted, _ = fit_image(predicted.astype(np.float32), raw["canvas"])
                value = torch.from_numpy(fitted[None]).to(self.device)
            depths[previous] = value
            data[prefix + "_depth"] = value
        if ordered:
            self.previous_key = (*sequence, row["current_id"])
            self.previous_depth = depths[False]
        return data

    @torch.no_grad()
    def add_flow(self, raw, base, flow, teacher=False):
        data = dict(base)
        previous, current = raw["fitted"][True], raw["fitted"][False]
        forward, backward = self.flows[flow](previous, current)
        for key, value in (("forward", forward), ("backward", backward)):
            if not np.isfinite(value).all():
                raise ValueError(f"Invalid {flow} {key}: {raw['row']['id']}")
            data[key] = torch.from_numpy(value.astype(np.float32)).to(self.device)
        if teacher:
            output = self.teacher(previous, current, backward.astype(np.float32))
            visibility = output["visibility"]
            if not np.isfinite(visibility).all():
                raise ValueError(f"Invalid MFTIQ visibility: {raw['row']['id']}")
            data["teacher_visibility"] = torch.from_numpy(visibility.astype(np.float32)).to(
                self.device)
        return data

    def compute(self, raw, flow, teacher=False, ordered=False):
        return self.add_flow(raw, self.depth_pair(raw, ordered), flow, teacher)

    def timed_depth(self, raw):
        synchronize(self.device)
        began = time.perf_counter()
        data = self.depth_pair(raw, ordered=True)
        synchronize(self.device)
        return data, time.perf_counter() - began
