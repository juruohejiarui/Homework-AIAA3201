"""Depth Anything V2 Small adapter."""
import sys

import cv2
import torch

from depth_project.models.attention import configure
from depth_project.paths import SMALL, THIRD_PARTY
from depth_project.models.api import ImageAdapter


class DepthAnythingAdapter(ImageAdapter):
    runtime_paths = ("third_party/Depth-Anything-V2",)
    kind = "relative_inverse"

    def forward(self, request, **specific_param):
        previous = self.input_size
        self.input_size = specific_param.pop("input_size", self.input_size)
        try:
            return super().forward(request, **specific_param)
        finally:
            self.input_size = previous

    def __init__(self, weights, fp32=True, input_size=518, backend="upstream", align_corners=True):
        sys.path.insert(0, str(THIRD_PARTY / "Depth-Anything-V2"))
        from depth_anything_v2.dpt import DepthAnythingV2

        configure("depth_anything_v2", backend)
        self.model = DepthAnythingV2(**SMALL)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = self.model.cuda().eval()
        self.fp32 = fp32
        self.input_size = input_size
        self.align_corners = align_corners

    @torch.no_grad()
    def predict(self, rgb):
        with torch.autocast("cuda", dtype=torch.float16, enabled=not self.fp32):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if self.align_corners:
                return self.model.infer_image(bgr, input_size=self.input_size)
            image, (h, w) = self.model.image2tensor(bgr, self.input_size)
            depth = self.model(image)
            return torch.nn.functional.interpolate(
                depth[:, None], (h, w), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
