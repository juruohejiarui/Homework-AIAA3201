"""Video Depth Anything Small adapter; frames stay in their original order."""
import sys

import torch
from depth_project.models.api import ModelOutput, frame_array

from depth_project.models.attention import configure
from depth_project.paths import SMALL, THIRD_PARTY


class VideoDepthAdapter:
    runtime_paths = ("third_party/Video-Depth-Anything",)
    capability = "sequence"

    def forward(self, request, **specific_param):
        import numpy as np
        fps = specific_param.pop("fps", 1.)
        if specific_param or request.specific_in:
            raise ValueError("Unsupported VDA inputs/parameters")
        frames = (request.rgb if isinstance(request.rgb, np.ndarray) and request.rgb.ndim == 4
                  else np.stack([frame_array(frame) for frame in request.rgb]))
        values = self.predict_video(frames, fps)
        return ModelOutput(torch.as_tensor(values, dtype=torch.float32), self.kind,
                           request.frame_ids)

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
        from depth_project.models.video_stream import infer
        return infer(self, frames)
