"""Scalar runtime deltas shared by training and evaluation without extra inference."""


def snapshot(runtime):
    return runtime.statistics() if hasattr(runtime, "statistics") else {}


def difference(runtime, before):
    return {key: value - before.get(key, 0) for key, value in snapshot(runtime).items()}
