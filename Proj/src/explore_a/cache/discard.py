"""Explicit length-policy deletion of one remote managed task; never local source data."""
import shutil

from depth_project.io import read_json, write_json
from explore_a.cache.paths import component, expected
from explore_a.cache.policy import skipped
from explore_a.cache.storage import safe_relative


def discard_length(root, name):
    """Caller must hold the remote worker lock; deletion is explicitly user-requested."""
    component(name)
    if not skipped(root, name):
        raise ValueError(f"Refusing deletion of a cache-eligible video: {name}")
    folder = root / "jobs" / name
    job = read_json(folder / "job.json")
    if job["job"] != name or job["recipe"] != root.name:
        raise ValueError("Remote task identity differs from requested deletion")
    paths = []
    for relative in expected(job["rows"]):
        path = root / safe_relative(relative)
        for candidate in (path, path.with_suffix(".tmp")):
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Deletion escapes managed remote output: {candidate}")
            if candidate.exists():
                if not candidate.is_file():
                    raise ValueError(f"Output is not a file: {candidate}")
                paths.append(candidate)
    staging = root / ".staging" / name
    if staging.is_symlink() or not staging.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unsafe remote staging directory")
    output_bytes = sum(path.stat().st_size for path in paths)
    input_bytes = sum(path.stat().st_size for path in staging.rglob("*")
                      if path.is_file() and not path.is_symlink()) if staging.exists() else 0
    for path in paths:
        path.unlink()
    if staging.exists():
        shutil.rmtree(staging)
    report = {"job": name, "state": "skipped_length", "output_files": len(paths),
              "output_bytes_deleted": output_bytes, "managed_input_bytes_deleted": input_bytes}
    write_json(folder / "discarded_length.json", report)
    write_json(folder / "skipped_length.json", {"reason": "length_policy"})
    return report
