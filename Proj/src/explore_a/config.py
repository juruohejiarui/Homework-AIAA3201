"""Explicit experiment settings and identities; importing never starts computation."""
import math
import copy
from pathlib import Path

import yaml

from depth_project.io import read_json, write_json
from depth_project.paths import PROJECT

DEFAULT_CONFIG = PROJECT / "configs/explore_a/mixed_h5_da2.yaml"
FLOWS = ("raft", "farneback")
OCCLUSIONS = ("rule", "mftiq")
FUSIONS = ("unet_guided", "flow_refine", "final_flow")


def read_config(path, parents=()):
    """Resolve small experiment overrides; generated worker configs remain self-contained."""
    path = Path(path).resolve()
    if path in parents:
        raise ValueError(f"Cyclic configuration inheritance: {path}")
    supplied = yaml.safe_load(path.read_text())
    base = supplied.pop("extends", None)
    result = read_config(path.parent / base, (*parents, path)) if base else {}

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    merge(result, supplied)
    return result


def load_config(path=DEFAULT_CONFIG):
    config = read_config(path)
    config.setdefault("frontend", {"name":"da2"})
    config.setdefault("cache", {}).setdefault("recipe", "glob-da2")
    # Keep the existing manifest's exact schema; null adapter options need no YAML entries.
    for key in ("scene_groups", "scannet_train_list", "scannet_val_list", "nyu_raw_root"):
        config["data"].setdefault(key, None)
    from explore_a.frontend import name
    name(config)
    sections = {"online": {"read_workers": 4, "prefetch": 8,
                           "raft_bidirectional_batch": True},
                "result_io": {"write_workers": 2, "write_queue": 8}}
    for section, defaults in sections.items():
        settings = config.setdefault(section, {})
        for key, value in defaults.items():
            settings.setdefault(key, value)
            if key != "raft_bidirectional_batch" and (
                    not isinstance(settings[key], int) or settings[key] < 1):
                raise ValueError(f"{section}.{key} must be a positive integer")
    if not isinstance(config["online"]["raft_bidirectional_batch"], bool):
        raise ValueError("online.raft_bidirectional_batch must be boolean")
    from explore_a.observability import DEFAULTS
    observed = config.setdefault("observability", {})
    for key, value in DEFAULTS.items():
        observed.setdefault(key, value)
    if observed["progress"] not in ("auto", "bar", "text", "off"):
        raise ValueError("observability.progress must be auto/bar/text/off")
    for key in ("tensorboard", "plots"):
        if not isinstance(observed[key], bool):
            raise ValueError(f"observability.{key} must be boolean")
    for key in ("plot_every", "refresh_seconds", "non_tty_every_s", "tb_flush_s"):
        if observed[key] <= 0:
            raise ValueError(f"observability.{key} must be positive")
    training = config["training"]
    training.setdefault("heldout_every", 0)
    # Name the existing fixed-probability baseline without changing its RNG stream.
    training.setdefault("sampling_mode", "fix")
    if training["sampling_mode"] not in ("fix", "video_sqrt"):
        raise ValueError("training.sampling_mode must be fix or video_sqrt")
    defaults = {"correction_fraction": .1, "correction_penalty": .01,
                "blend_aux_weight": .10, "reset_aux_weight": .05,
                "blend_min_gap": .01, "reset_min_gain": .001,
                "global_correction_weight": .01, "local_correction_weight": .01,
                "temporal_weight": .05, "temporal_warmup_steps": 500}
    for key, value in defaults.items():
        training.setdefault(key, value)
        if not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if training["correction_fraction"] >= 1:
        raise ValueError("correction_fraction must be smaller than one for positive depth")
    if training["blend_min_gap"] <= 0:
        raise ValueError("blend_min_gap must be positive")
    backend = config.setdefault("execution", {}).setdefault("backend", "online")
    if backend not in ("online", "cache", "hybrid"):
        raise ValueError("execution.backend must be online, cache or hybrid")
    from explore_a.cache.read_policy import settings as read_settings
    policy = read_settings(config)
    if policy is not None:
        config["cache_read"] = policy
    if "remote_online" in config:
        from explore_a.remote_online.policy import settings as remote_settings
        config["remote_online"] = remote_settings(config)
    limit = config.setdefault("cache_policy", {}).setdefault("max_rgb_frames", 4000)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("cache_policy.max_rgb_frames must be a positive integer")
    config["config_path"] = str(Path(path).resolve())
    for section in ("paths", "weights"):
        for key, value in config[section].items():
            target = Path(value).expanduser()
            config[section][key] = str(target if target.is_absolute() else PROJECT / target)
    for key in ("scene_groups", "scannet_train_list", "scannet_val_list", "nyu_raw_root",
                "nyu_h5_root"):
        if config["data"].get(key):
            target = Path(config["data"][key]).expanduser()
            config["data"][key] = str(target if target.is_absolute() else PROJECT / target)
    for key in ("microbatch", "effective_batch", "steps", "validate_every", "save_every",
                "log_every"):
        if config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if config["training"]["effective_batch"] % config["training"]["microbatch"]:
        raise ValueError("effective_batch must be divisible by microbatch")
    if config["training"]["heldout_every"] < 0:
        raise ValueError("training.heldout_every must be nonnegative (0 disables monitoring)")
    if not 0 < config["data"]["dev_fraction"] < 1:
        raise ValueError("dev_fraction must be between zero and one")
    weights = config["training"].get("source_weights")
    if weights is not None:
        if set(weights) != set(config["data"]["datasets"]) or any(
                not isinstance(v, (int, float)) or isinstance(v, bool)
                or not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("training.source_weights must give every dataset a positive weight")
    if config["data"].get("nyu_h5_root") and config["data"].get("nyu_raw_root"):
        raise ValueError("Select either nyu_h5_root or nyu_raw_root")
    return config


def bind_identity(path, identity):
    """Refuse to reuse artifacts from a different split, model or numeric recipe."""
    path = Path(path)
    if path.exists():
        previous = read_json(path)
        if "experiment" in previous and "experiment" in identity:
            from explore_a.training.identity import assert_same_task, record_change
            assert_same_task(previous, identity)
            if previous != identity:
                record_change(path, previous, identity)
                write_json(path, identity)
                print("Run config updated; continuing existing checkpoint:", path, flush=True)
            return
        if "input" in identity and "input" not in previous:
            raise ValueError(
                f"Legacy protocol; choose a NEW runs directory for online mode: {path}")
        if previous != identity:
            raise ValueError(f"Settings changed; choose a new runs/output directory: {path}")
    else:
        write_json(path, identity)


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def experiment_name(flow, occlusion, fusion):
    return f"{flow}-{occlusion}-{fusion}"
