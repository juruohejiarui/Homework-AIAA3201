"""Immutable per-phase allocation; ticket phase owns routing even during nested evaluation."""
import math

from explore_a.cache.read_policy import settings as cache_settings
from explore_a.remote_online.policy import settings as remote_settings


def profiles(config):
    cache, remote = cache_settings(config) or {}, remote_settings(config)
    base = {"cache": cache.get("probability", 0.), "remote": remote["probability"]}
    supplied = config.get("execution", {}).get("routing", {})
    if not isinstance(supplied, dict):
        raise ValueError("execution.routing must be a mapping")
    if set(supplied)-{"train", "evaluation"}:
        raise ValueError("execution.routing accepts train and evaluation only")
    result = {}
    for phase in ("train", "evaluation"):
        overrides = supplied.get(phase, {})
        if not isinstance(overrides, dict) or set(overrides)-set(base):
            raise ValueError("Each execution.routing profile accepts cache and remote only")
        value = {**base, **overrides}
        for key, probability in value.items():
            if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError(f"Invalid execution.routing.{phase}.{key}")
        if sum(value.values()) > 1+1e-12:
            raise ValueError(f"execution.routing.{phase} cache+remote must be <= 1")
        backend = config.get("execution", {}).get("backend", "online")
        if backend != "hybrid":
            value = {"cache": float(backend == "cache"), "remote": 0.}
        elif not remote["enabled"]:
            value["remote"] = 0.
        result[phase] = value
    return result


def for_phase(policies, phase):
    return policies["train" if phase == "train" else "evaluation"]
