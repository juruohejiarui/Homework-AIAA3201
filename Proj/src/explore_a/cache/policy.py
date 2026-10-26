"""Cache scheduling policy, separate from immutable samples and numeric recipes."""
import os
from pathlib import Path

from depth_project.io import read_json, write_json


def count_rgb(job, sources):
    """Count the complete original RGB sequence; KITTI's two color cameras are summed."""
    source = Path(next(iter(sources.values()))["source"])
    if job["rows"][0]["dataset"] == "kitti":
        directories = [source.parents[2] / camera / "data" for camera in ("image_02", "image_03")]
    else:
        directories = [source.parent]
    total = 0
    for directory in directories:
        # A missing camera cannot silently turn an unknown length into a short video.
        if not directory.is_dir():
            raise ValueError(f"Cannot count original RGB frames: {directory}")
        with os.scandir(directory) as entries:
            total += sum(entry.name.lower().endswith((".png", ".jpg", ".jpeg"))
                         and entry.is_file() for entry in entries)
    return total


def refresh_policy(root, config):
    limit = config.get("cache_policy", {}).get("max_rgb_frames", 4000)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("cache_policy.max_rgb_frames must be a positive integer")
    videos = {}
    for item in read_json(root / "plan.json")["jobs"]:
        name = item["job"]
        folder = root / "jobs" / name
        job = read_json(folder / "job.json")
        total = count_rgb(job, read_json(folder / "sources.json"))
        videos[name] = {"rgb_frames": total, "eligible": total <= limit,
                        "reason": "" if total <= limit else "skipped_length"}
        marker = folder / "skipped_length.json"
        if total > limit:
            write_json(marker, {"rgb_frames": total, "max_rgb_frames": limit})
        else:
            marker.unlink(missing_ok=True)
    result = {"max_rgb_frames": limit, "videos": videos}
    write_json(root / "cache_policy.json", result)
    return result


def skipped(root, name):
    if (root / "jobs" / name / "skipped_length.json").exists():
        return True
    path = root / "cache_policy.json"
    if not path.exists():
        return False
    return not read_json(path)["videos"].get(name, {"eligible": True})["eligible"]


def ensure_eligible(root, name):
    if skipped(root, name):
        raise ValueError(f"Cache generation/transfer disabled by length policy: {name}")


def eligible_names(root, names):
    path = root / "cache_policy.json"
    videos = read_json(path)["videos"] if path.exists() else {}
    return [name for name in names if videos.get(name, {"eligible": True})["eligible"]]
