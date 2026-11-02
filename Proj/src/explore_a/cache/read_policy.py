"""Cache allocation settings; availability is resolved before task submission."""
import math

DEFAULTS = {"mode":"blocking", "probability":.2, "max_inflight":4}


def settings(config):
    supplied = config.get("cache_read")
    if supplied is None:
        return None
    if set(supplied)-set(DEFAULTS):
        raise ValueError("Unknown cache_read settings")
    value = {**DEFAULTS, **supplied}
    if value["mode"] != "blocking":
        raise ValueError("cache_read.mode must be blocking")
    p = value["probability"]
    if isinstance(p, bool) or not isinstance(p, (int,float)) or not math.isfinite(p) or not 0<=p<=1:
        raise ValueError("Invalid cache_read.probability")
    if type(value["max_inflight"]) is not int or value["max_inflight"] < 1:
        raise ValueError("Invalid cache_read.max_inflight")
    return value


def enabled(config):
    return config.get("execution", {}).get("backend") == "hybrid"
