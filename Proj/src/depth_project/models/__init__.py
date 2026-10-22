"""Model-specific adapters. Every adapter accepts RGB and returns float depth maps."""
from depth_project.models.da2 import DepthAnythingAdapter
from depth_project.models.depthpro import DepthProAdapter
from depth_project.models.vda import VideoDepthAdapter


def load_model(name, weights, fp32=True, input_size=518, backend="upstream", metric=False, focal=None):
    if name == "da2":
        return DepthAnythingAdapter(weights, fp32, input_size, backend)
    if name == "depthpro":
        return DepthProAdapter(weights, fp32, focal)
    return VideoDepthAdapter(weights, fp32, input_size, backend, metric)
