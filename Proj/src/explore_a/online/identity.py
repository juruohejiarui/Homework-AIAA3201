"""Numeric provenance for online experiments, independent of performance queue settings."""
from explore_a.config import file_identity


def required_weights(flows, teacher=False):
    names = ["depthpro"]
    if "raft" in flows:
        names.append("raft")
    if teacher:
        names.extend(("mftiq", "resnet", "dinov2"))
    return names


def input_identity(config, flow):
    if config.get("execution", {}).get("backend") == "hybrid":
        from explore_a.cache.reader import cache_spec
        return {"schema": 4, "execution": "hybrid_fp32_per_artifact_v1", "flow": flow,
                "recipe": cache_spec(config), "manifest": file_identity(config["paths"]["manifest"]),
                "online_weights": {name: file_identity(path)
                                   for name, path in config["weights"].items()},
                "attention": "flash_prefer", "write_missing_predictions": False}
    if config.get("execution", {}).get("backend") == "cache":
        from explore_a.cache.reader import cache_spec
        return {"schema": 3, "execution": "dgx_cache_fp32", "flow": flow,
                "recipe": cache_spec(config), "manifest": file_identity(config["paths"]["manifest"])}
    return {"schema": 2, "execution": "online_raw_no_intermediate_quantization",
            "manifest": file_identity(config["paths"]["manifest"]), "data": config["data"],
            "geometry": config["geometry"], "flow": flow,
            "weights": {name: file_identity(config["weights"][name])
                        for name in required_weights([flow])},
            "precision": {"depthpro": "fp16", "flow": "fp32"},
            "depthpro_focal": "model_estimated"}


def teacher_identity(config):
    if config.get("execution", {}).get("backend") in ("cache", "hybrid"):
        from explore_a.cache.reader import cache_spec
        return cache_spec(config)["weights"]
    return {name: file_identity(config["weights"][name])
            for name in ("mftiq", "resnet", "dinov2")}


def check_online(identity):
    if "input" not in identity or identity["input"].get("schema") not in (2, 3, 4):
        raise ValueError("Legacy experiment protocol; choose a NEW runs directory for online mode")
