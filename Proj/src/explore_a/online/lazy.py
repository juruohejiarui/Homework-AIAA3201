"""Lazy frozen frontends for hybrid reads, with initialization RNG isolated from training."""
from contextlib import contextmanager
import random
import time

import numpy as np
import torch

from explore_a.models.attention import configure_depth_attention
from explore_a.online import frozen


@contextmanager
def preserve_rng(device):
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class LazyFrozen:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.models, self.load_seconds, self.attention = {}, 0., None

    def get(self, name):
        if name not in self.models:
            began = time.perf_counter()
            with preserve_rng(self.device), torch.no_grad():
                if name == "depth":
                    frozen.check_weights(self.config, ["depthpro"])
                    model = frozen.DepthProAdapter(self.config["weights"]["depthpro"], fp32=False)
                    model.model.eval().requires_grad_(False)
                    self.attention = configure_depth_attention(model.model)
                elif name == "teacher":
                    frozen.check_weights(self.config, ["mftiq", "resnet", "dinov2"])
                    model = frozen.MFTIQVisibility(self.config, self.device)
                else:
                    if name == "raft":
                        frozen.check_weights(self.config, ["raft"])
                    model = frozen.PairFlow(name, self.config, self.device)
            self.models[name] = model
            self.load_seconds += time.perf_counter() - began
        return self.models[name]

    @torch.no_grad()
    def depth(self, rgb):
        model = self.get("depth")
        value = model.predict(rgb)
        if hasattr(model, "focal_predictions"):
            model.focal_predictions.clear()
        if not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError("Invalid online Depth Pro depth")
        return value.astype(np.float32)

    @torch.no_grad()
    def flow(self, name, previous, current):
        return self.get(name)(previous, current)

    @torch.no_grad()
    def visibility(self, previous, current, backward):
        return self.get("teacher")(previous, current, backward)["visibility"]
