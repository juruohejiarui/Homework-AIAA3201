"""Read actual past evaluation records; aliases never turn a used drive into new data."""
import re
from pathlib import Path

from depth_project.io import read_json
from depth_project.paths import PROJECT

DRIVE = re.compile(r"20\d{2}_\d{2}_\d{2}_drive_\d{4}(?:_sync)?")


def drive_id(value):
    match = DRIVE.search(str(value))
    if not match:
        raise ValueError(f"Not a KITTI drive: {value}")
    return match.group().removesuffix("_sync") + "_sync"


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def evaluation_history(roots=None):
    """Conservatively exclude every drive named in past evaluation identities/results."""
    sources = {}
    if roots is None:
        # This evidence snapshot protects a code-only transfer or archived old outputs too.
        snapshot = PROJECT / "configs/explore_a/historical_kitti.json"
        if snapshot.exists():
            sources = {name: set(paths) for name, paths in read_json(snapshot).items()}
        roots = [PROJECT / "outputs", PROJECT / "data/kitti/prepared"]
    for root in roots:
        if not Path(root).exists():
            continue
        for path in sorted(Path(root).rglob("*.json")):
            if path.is_relative_to(PROJECT / "outputs/explore_a"):
                continue
            if path.name not in {"manifest.json", "run.json", "metrics.json", "complete.json"}:
                continue
            data = read_json(path)
            if isinstance(data, dict) and "experiment" in data and "temporal_note" in data:
                continue  # This implementation's development evaluations are not old Part 1/2.
            for text in strings(data):
                for match in DRIVE.finditer(text):
                    sources.setdefault(drive_id(match.group()), set()).add(str(path.resolve()))
    if roots and not sources:
        raise ValueError("No historical KITTI records found; refusing an unprotected split")
    return {name: sorted(paths) for name, paths in sorted(sources.items())}


def assert_history_current(manifest):
    current = evaluation_history()
    known = set(manifest["history"])
    added = set(current) - known
    if added:
        raise ValueError("New historical evaluation drives found: " + ", ".join(sorted(added))
                         + ". Prepare a NEW manifest and output directory before continuing.")


def historical_scenes(dataset):
    """Collect scene names from existing benchmark manifests, excluding new A runs."""
    found = set()
    for root in [PROJECT / "outputs", PROJECT / "data" / dataset / "prepared"]:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.is_relative_to(PROJECT / "outputs/explore_a"):
                continue
            if path.name not in {"manifest.json", "run.json"}:
                continue
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            data = data.get("manifest", data)
            if isinstance(data, dict) and data.get("dataset") == dataset:
                found.update(s["name"] for s in data.get("sequences", []))
    return found
