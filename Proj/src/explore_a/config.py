"""Explicit experiment settings and identities; importing never starts computation."""
from pathlib import Path

import yaml

from depth_project.io import read_json, write_json
from depth_project.paths import PROJECT

DEFAULT_CONFIG = PROJECT / "configs/explore_a/mixed.yaml"
FLOWS = ("raft", "farneback")
OCCLUSIONS = ("rule", "mftiq", "student")
FUSIONS = ("cnn", "unet", "residual")


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
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
    config["training"].setdefault("heldout_every", 0)
    backend = config.setdefault("execution", {}).setdefault("backend", "online")
    if backend not in ("online", "cache", "hybrid"):
        raise ValueError("execution.backend must be online, cache or hybrid")
    limit = config.setdefault("cache_policy", {}).setdefault("max_rgb_frames", 4000)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("cache_policy.max_rgb_frames must be a positive integer")
    config["config_path"] = str(Path(path).resolve())
    for section in ("paths", "weights"):
        for key, value in config[section].items():
            target = Path(value).expanduser()
            config[section][key] = str(target if target.is_absolute() else PROJECT / target)
    for key in ("scene_groups", "scannet_train_list", "scannet_val_list", "nyu_raw_root"):
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
    return config


def bind_identity(path, identity):
    """Refuse to reuse artifacts from a different split, model or numeric recipe."""
    path = Path(path)
    if path.exists():
        previous = read_json(path)
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
