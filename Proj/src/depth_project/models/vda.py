"""Video Depth Anything Small adapter; frames stay in their original order."""
import sys

import torch

from depth_project.models.attention import configure
from depth_project.paths import SMALL, THIRD_PARTY


class VideoDepthAdapter:
    def __init__(self, weights, fp32=True, input_size=518, backend="upstream", metric=False):
        sys.path.insert(0, str(THIRD_PARTY / "Video-Depth-Anything"))
        from video_depth_anything.video_depth import VideoDepthAnything

        configure("video_depth_anything", backend)
        self.model = VideoDepthAnything(**SMALL, metric=metric)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = self.model.cuda().eval()
        self.fp32 = fp32
        self.input_size = input_size
        self.kind = "metric_m" if metric else "relative_inverse"

    def predict_video(self, frames, fps):
        depths, _ = self.model.infer_video_depth(
            frames, fps, input_size=self.input_size, device="cuda", fp32=self.fp32,
        )
        return depths
