"""Model-specific adapters. Every adapter accepts RGB and returns float depth maps."""
from depth_project.models.da2 import DepthAnythingAdapter
from depth_project.models.vda import VideoDepthAdapter


def load_model(name, weights, fp32=True, input_size=518, backend="upstream", metric=False):
    if name == "da2":
        return DepthAnythingAdapter(weights, fp32, input_size, backend)
    if name == "unidepthv2":
        from depth_project.models.unidepth import UniDepthAdapter
        return UniDepthAdapter(weights, fp32)
    if name == "vda":
        return VideoDepthAdapter(weights, fp32, input_size, backend, metric)
    raise ValueError(f"Unknown model: {name}")
