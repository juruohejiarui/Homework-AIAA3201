"""Both endpoints share one transport recipe; host-specific absolute paths stay outside it."""
from pathlib import Path

from explore_a.config import DEFAULT_CONFIG

LOCAL_ROOT = Path.home() / "Documents/ResourcesLow/cache/Homework-AIAA3201"
REMOTE_ROOT = "/home/hjr/Documents/Resources/cache/Homework-AIAA3201"
REMOTE_PROJECT = "/home/hjr/Documents/Projects/Homework-AIAA3201"
RECIPE = "glob"


def root_path(base, recipe=RECIPE):
    from explore_a.cache.paths import component
    if recipe == "glob-da2":
        return Path(base).expanduser() / "explore_a/glob/frontends/da2"
    return Path(base).expanduser() / "explore_a" / component(recipe)


def ready_files(base, filename):
    """Only supported recipe locations, including the separate DA prediction namespace."""
    directory = Path(base) / "explore_a"
    patterns = (f"*/jobs/*/{filename}", f"glob/frontends/da2/jobs/*/{filename}")
    return sorted(path for pattern in patterns for path in directory.glob(pattern))


def root_recipe(root):
    root = Path(root)
    return "glob-da2" if root.name == "da2" and root.parent.name == "frontends" else root.name
