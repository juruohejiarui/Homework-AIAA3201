"""Model-specific adapters. Every adapter accepts RGB and returns float depth maps."""


def __getattr__(name):
    # Preserve public class imports without loading adapters for metadata-only imports.
    if name == "DepthAnythingAdapter":
        return adapter_type("da2")
    if name == "VideoDepthAdapter":
        return adapter_type("vda")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def adapter_type(name):
    if ":" in name:
        from depth_project.models.api import resolve_adapter
        return resolve_adapter(name)
    if name == "da2":
        from depth_project.models.da2 import DepthAnythingAdapter
        return DepthAnythingAdapter
    if name == "vda":
        from depth_project.models.vda import VideoDepthAdapter
        return VideoDepthAdapter
    if name == "unidepthv2":
        from depth_project.models.unidepth import UniDepthAdapter
        return UniDepthAdapter
    raise ValueError(f"Unknown model: {name}")


def load_model(name, weights, fp32=True, input_size=518, backend="upstream", metric=False,
               **specific_param):
    if ":" in name:
        import inspect
        from depth_project.models.api import resolve_adapter
        adapter = resolve_adapter(name)
        parameters = inspect.signature(adapter).parameters
        common = dict(weights=weights, fp32=fp32, input_size=input_size,
                      backend=backend, metric=metric)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            common = {k:v for k,v in common.items() if k in parameters}
        return adapter(**common, **specific_param)
    align_corners = specific_param.pop("align_corners", True) if name == "da2" else True
    if specific_param:
        raise ValueError(f"Unsupported loader parameters for {name}: {specific_param.keys()}")
    if name == "da2":
        return adapter_type(name)(weights, fp32, input_size, backend, align_corners=align_corners)
    if name == "unidepthv2":
        from depth_project.models.unidepth import UniDepthAdapter
        return UniDepthAdapter(weights, fp32)
    if name == "vda":
        return adapter_type(name)(weights, fp32, input_size, backend, metric)
    raise ValueError(f"Unknown model: {name}")
