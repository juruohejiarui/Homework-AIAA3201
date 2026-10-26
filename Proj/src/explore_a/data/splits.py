"""Fixed group splits and canonical frame checks, before any predictions are generated."""
import random
from pathlib import Path

from explore_a.data.history import drive_id


def development_groups(groups, seed, fraction):
    groups = sorted(set(groups))
    if len(groups) < 2:
        raise ValueError("At least two independent training groups are required")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    return set(groups[:count])


def validate_rows(rows, history):
    frames, groups, ids = {}, {}, set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate canonical sample: {row['id']}")
        ids.add(row["id"])
        split = row["split"]
        group = (row["dataset"], row["group"])
        if group in groups and groups[group] != split:
            raise ValueError(f"Physical group crosses splits: {group}")
        groups[group] = split
        if row["dataset"] == "kitti" and split != "test" and row["sequence"] in history:
            raise ValueError(f"Historical evaluation drive leaked: {row['sequence']}")
        if row["dataset"] == "kitti":
            if int(row["current_id"]) != int(row["previous_id"]) + 1:
                raise ValueError("KITTI pair is not consecutive Raw RGB")
            for prefix in ("previous", "current"):
                path = Path(row[f"{prefix}_rgb"])
                if drive_id(path) != row["sequence"] or path.stem != row[f"{prefix}_id"] \
                        or row["camera"] not in path.parts:
                    raise ValueError(f"KITTI path disagrees with canonical frame identity: {path}")
        for frame in (row["previous_id"], row["current_id"]):
            key = (row["dataset"], row["sequence"], row["camera"], str(frame))
            if key in frames and frames[key] != split:
                raise ValueError(f"RGB frame crosses splits: {key}")
            frames[key] = split
    for split in ("train", "dev", "test"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"Empty {split} split")
