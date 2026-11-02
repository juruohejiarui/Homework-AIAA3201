"""Reuse existing cache weight-size identity and frozen numeric conventions."""
from pathlib import Path

from explore_a.online.identity import required_weights
from explore_a.remote_online.wire import VERSION


def specification(config, flows, teacher):
    from explore_a.frontend import metadata
    result = {"protocol": VERSION, "flows": sorted(flows), "teacher": teacher,
            "weights": {name: {"filename": Path(config["weights"][name]).name,
                               "bytes": Path(config["weights"][name]).stat().st_size}
                        for name in required_weights(flows, teacher, config)},
            "geometry": config["geometry"],
            "data": {key: config["data"][key] for key in ("canvas_kitti", "canvas_indoor")},
            "online": {"raft_bidirectional_batch": config["online"]["raft_bidirectional_batch"]},
            "attention": "sdpa", "frontend":metadata(config),
            "precision": {"da2": "fp16", "flow": "fp32", "mftiq": "fp32"}}
    return result


def portable_config(config, spec):
    from depth_project.paths import PROJECT
    weights = {}
    for name in spec["weights"]:
        try:
            weights[name] = str(Path(config["weights"][name]).relative_to(PROJECT))
        except ValueError as error:
            raise ValueError("Remote weights must be inside Proj; sync weights separately"
                             ) from error
    result = {"weights": weights, **{key: spec[key] for key in ("data", "geometry", "online")}}
    result["frontend"] = spec["frontend"]
    if spec["teacher"]:
        result["paths"] = {"dinov2_source": str(
            Path(config["paths"]["dinov2_source"]).relative_to(PROJECT))}
    return result
