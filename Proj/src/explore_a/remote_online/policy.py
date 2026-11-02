"""Absolute three-way probabilities use a random stream independent of training."""
import math
import random


DEFAULTS = {"enabled": False, "probability": 0., "host": "Dgx-e",
            "project": "/home/hjr/Documents/Projects/Homework-AIAA3201",
            "python": "/home/hjr/miniconda3/envs/aiaa3201/bin/python",
            "max_inflight": 12, "startup_timeout_s": 120,
            "send_ahead": 2}


def settings(config):
    supplied = config.get("remote_online", {})
    if set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown remote_online settings")
    value = {**DEFAULTS, **supplied}
    if not isinstance(value["enabled"], bool):
        raise ValueError("remote_online.enabled must be boolean")
    for key in ("probability", "startup_timeout_s"):
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number < 0):
            raise ValueError(f"remote_online.{key} must be finite and nonnegative")
    if value["probability"] > 1:
        raise ValueError("remote_online.probability must be <= 1")
    for key in ("max_inflight", "send_ahead"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"remote_online.{key} must be a positive integer")
    for key in ("startup_timeout_s",):
        if value[key] <= 0:
            raise ValueError(f"remote_online.{key} must be positive")
    for key in ("host", "project", "python"):
        if not isinstance(value[key], str) or not value[key] or value[key].startswith("-"):
            raise ValueError(f"Invalid remote_online.{key}")
    # Choosing online/cache is sufficient; stale optional DGX settings stay inactive.
    if config.get("execution", {}).get("backend") != "hybrid":
        value["enabled"] = False
    if value["enabled"]:
        if config.get("cache_read", {}).get("mode") != "blocking":
            raise ValueError("remote_online requires blocking hybrid")
        if config["cache_read"]["probability"] + value["probability"] > 1:
            raise ValueError("Cache and DGX absolute probabilities must sum to <= 1")
    return value


def enabled(config):
    policy = settings(config)
    return policy["enabled"] and policy["probability"] > 0


def route(seed, occurrence, batch, cache_probability, remote_probability):
    step, slot = divmod(occurrence, batch)
    value = random.Random(f"frontend-route:{seed}:{step}:{slot}").random()
    if value < cache_probability:
        return "cache"
    return "remote" if value < cache_probability + remote_probability else "local"
