"""Depth Anything V2 Small adapter."""
import sys

import cv2
import torch

from depth_project.models.attention import configure
from depth_project.paths import SMALL, THIRD_PARTY


class DepthAnythingAdapter:
    kind = "relative_inverse"

    def __init__(self, weights, fp32=True, input_size=518, backend="upstream"):
        sys.path.insert(0, str(THIRD_PARTY / "Depth-Anything-V2"))
        from depth_anything_v2.dpt import DepthAnythingV2

        configure("depth_anything_v2", backend)
        self.model = DepthAnythingV2(**SMALL)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = self.model.cuda().eval()
        self.fp32 = fp32
        self.input_size = input_size

    def predict(self, rgb):
        with torch.autocast("cuda", dtype=torch.float16, enabled=not self.fp32):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return self.model.infer_image(bgr, input_size=self.input_size)
