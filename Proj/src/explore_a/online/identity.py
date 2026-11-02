"""Numeric provenance for online experiments, independent of performance queue settings."""
from explore_a.config import file_identity


def required_weights(flows, teacher=False, config=None):
    from explore_a.frontend import name
    names = [name(config or {})]
    if "raft" in flows:
        names.append("raft")
    if teacher:
        names.extend(("mftiq", "resnet", "dinov2"))
    return names


def input_identity(config, flow):
    result = _input_identity(config, flow)
    from explore_a.frontend import metadata
    result.update(frontend=metadata(config), attention="sdpa",
                  precision={"da2": "fp16", "flow": "fp32"})
    return result


def _input_identity(config, flow):
    if config.get("execution", {}).get("backend") == "hybrid":
        from explore_a.cache.reader import cache_spec
        result = {"schema": 4, "execution": "hybrid_fp32_per_artifact_v1", "flow": flow,
                  "recipe": cache_spec(config) if _has_recipe(config) else None,
                  "manifest": file_identity(config["paths"]["manifest"]),
                  "online_weights": {name: file_identity(path)
                                     for name, path in config["weights"].items()
                                     if name in required_weights([flow], config=config)},
                  "attention": "flash_prefer", "write_missing_predictions": False}
        from explore_a.cache.read_policy import settings
        from explore_a.remote_online.policy import settings as remote_settings
        result.update(schema=6, execution="blocking_frozen_components_v2",
                      cache_read=settings(config), remote_online=remote_settings(config),
                      evaluation_backend="same", write_missing_predictions=False)
        return result
    if config.get("execution", {}).get("backend") == "cache":
        from explore_a.cache.reader import cache_spec
        return {"schema": 3, "execution": "dgx_cache_fp32", "flow": flow,
                "recipe": cache_spec(config),
                "manifest": file_identity(config["paths"]["manifest"])}
    return {"schema": 2, "execution": "online_raw_no_intermediate_quantization",
            "manifest": file_identity(config["paths"]["manifest"]), "data": config["data"],
            "geometry": config["geometry"], "flow": flow,
            "weights": {name: file_identity(config["weights"][name])
                        for name in required_weights([flow], config=config)},
            "precision": {"da2": "fp16", "flow": "fp32"}}


def teacher_identity(config):
    return {name: file_identity(config["weights"][name])
            for name in ("mftiq", "resnet", "dinov2")}


def _has_recipe(config):
    from explore_a.cache.reader import configured_root
    return (configured_root(config)/"recipe.json").is_file()
