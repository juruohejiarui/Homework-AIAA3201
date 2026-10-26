"""Human-readable video cache coverage; readers still validate each requested NPZ."""
from collections import defaultdict
import csv
import fcntl
import os
from pathlib import Path

from depth_project.io import read_json, write_json


def coverage(root, job):
    parents = defaultdict(set)
    for name in job["outputs"]:
        path = Path(name)
        parents[path.parent].add(path.name)
    count = 0
    for parent, names in parents.items():
        folder = root / parent
        if folder.is_dir():
            with os.scandir(folder) as entries:
                count += sum(entry.name in names and entry.is_file(follow_symlinks=False)
                             for entry in entries)
    return count


def refresh_catalog(root, remote_states=None):
    policy = root / "cache_policy.json"
    if not policy.exists():
        return []
    rows = []
    for name, video in read_json(policy)["videos"].items():
        folder = root / "jobs" / name
        job = read_json(folder / "job.json")
        available, expected = coverage(root, job), len(job["outputs"])
        state = "none" if not available else "partial"
        if available == expected and (folder / "local.complete.json").exists():
            state = "complete"
        row = {"job": name, **video, "local_cache": state, "available_files": available,
               "expected_files": expected,
               "remote_state": (remote_states or {}).get(name, "unknown")}
        if (folder / "discarded_length.json").exists():
            row["remote_state"] = "discarded_length"
        rows.append(row)
    with (root / "catalog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(root / "cache_status.json", rows)
        temporary = root / "cache_status.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["job"])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "cache_status.csv")
    return rows
