"""Offline constructors used by the vendored MFTIQ feature extractor."""
from pathlib import Path

import torch
from torchvision.models import resnet50


def load_resnet(paths):
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(paths["resnet"], map_location="cpu", weights_only=True))
    return model


def load_dinov2(paths):
    source = Path(paths["dinov2_source"])
    if not (source / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 source is missing: {source}; see MODEL_DOWNLOADS.md")
    model = torch.hub.load(str(source), "dinov2_vits14_reg", source="local", pretrained=False)
    model.load_state_dict(torch.load(paths["dinov2"], map_location="cpu", weights_only=True))
    return model
