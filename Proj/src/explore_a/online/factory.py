"""Backend selection never silently fills missing cache entries with online inference."""


def make_runtime(config, flows, teacher=False, device="cuda"):
    if config.get("execution", {}).get("backend") == "hybrid":
        from explore_a.cache.hybrid import HybridRuntime
        return HybridRuntime(config, flows, teacher, device)
    if config.get("execution", {}).get("backend", "online") == "cache":
        from explore_a.cache.reader import CachedRuntime
        return CachedRuntime(config, flows, teacher, device)
    from explore_a.online.frozen import FrozenRuntime
    return FrozenRuntime(config, flows, teacher, device)
