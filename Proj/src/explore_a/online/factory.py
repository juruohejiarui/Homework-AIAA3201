"""All execution modes use the same frozen task scheduler."""
from explore_a.online.runtime import FrozenPipeline


def make_runtime(config, flows, teacher=False, device='cuda'):
    return FrozenPipeline(config, flows, teacher, device)
