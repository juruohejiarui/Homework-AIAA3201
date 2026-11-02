"""Frozen prediction representation; DA-V2-S relative inverse depth."""
from pathlib import Path

import numpy as np


def name(config):
    value = config.get("frontend", {}).get("name", "da2")
    if value != "da2":
        raise ValueError(f"Unsupported frozen frontend: {value}")
    return value


def relative(config):
    return name(config) == "da2"


def kind(config):
    return "relative_inverse"


def field(config):
    return "inverse_depth"


def metadata(config):
    return {"name": name(config), "kind": kind(config), "precision": "fp16",
            "preprocessing": "da2-vda-wide-518-v1"}


class DA2Frontend:
    """Match one preprocessing rule on local, producer and memory-RPC paths."""
    def __init__(self, weights):
        from depth_project.models.da2 import DepthAnythingAdapter
        self.adapter = DepthAnythingAdapter(weights, fp32=False, backend="sdpa")
        self.model = self.adapter.model.eval().requires_grad_(False)

    def predict(self, rgb):
        ratio = max(rgb.shape[:2]) / min(rgb.shape[:2])
        self.adapter.input_size = (round(int(518 * 1.777 / ratio) / 14) * 14
                                   if ratio > 1.78 else 518)
        return self.adapter.predict(rgb).astype(np.float32)


def make(config):
    name(config)
    return DA2Frontend(config["weights"]["da2"])


def shared_root(root):
    root = Path(root)
    return root.parents[1] if root.name == "da2" and root.parent.name == "frontends" else root


def valid_prediction(value, config):
    return np.isfinite(value).all() and (value >= 0).all()
