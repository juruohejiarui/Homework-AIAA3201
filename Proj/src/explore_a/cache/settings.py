"""Both endpoints share one transport recipe; host-specific absolute paths stay outside it."""
from pathlib import Path

from depth_project.paths import PROJECT

LOCAL_ROOT = Path.home() / "Documents/ResourcesLow/cache/Homework-AIAA3201"
REMOTE_ROOT = "/home/hjr/Documents/Resources/cache/Homework-AIAA3201"
REMOTE_PROJECT = "/home/hjr/Documents/Projects/Homework-AIAA3201"
DEFAULT_CONFIG = PROJECT / "configs/explore_a/kitti_tum.yaml"
RECIPE = "kitti_tum-fp32-flash-v1"


def root_path(base, recipe=RECIPE):
    from explore_a.cache.paths import component
    return Path(base).expanduser() / "explore_a" / component(recipe)
