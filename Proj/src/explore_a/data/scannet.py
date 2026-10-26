"""ScanNet .sens adapters with official and historical scene-space exclusions."""
from collections import Counter
from pathlib import Path

import numpy as np

from explore_a.data.history import historical_scenes
from explore_a.data.sens import Sensor
from explore_a.data.splits import development_groups


def read_scenes(path):
    if not path:
        raise ValueError("Supply the official ScanNet train and val list paths in YAML")
    return set(Path(path).expanduser().read_text().split())


def prepare_scannet(root, settings, aliases):
    train = read_scenes(settings["scannet_train_list"])
    val = read_scenes(settings["scannet_val_list"])
    group = lambda name: aliases.get(name, name.rsplit("_", 1)[0])
    historical = historical_scenes("scannet")
    protected = {group(name) for name in val | historical}
    files = sorted((root / "v1/scans").glob("*/*.sens"))
    if not files:
        raise FileNotFoundError(f"No completed v1 .sens files: {root}")
    eligible = [p for p in files if p.parent.name in train
                and group(p.parent.name) not in protected]
    dev = development_groups([group(p.parent.name) for p in eligible],
                             settings["seed"], settings["dev_fraction"])
    rows, rejected = [], Counter()
    for path in files:
        name = path.parent.name
        if group(name) in protected:
            split = "test"
        elif name in train:
            split = "dev" if group(name) in dev else "train"
        else:
            rejected["outside_official_splits"] += 1
            continue
        source = Sensor(str(path))
        for index in range(1, len(source.frames)):
            prev, cur = source.frames[index - 1:index + 1]
            if not 0 < cur[3] - prev[3] <= settings["max_frame_gap_s"]:
                rejected["rgb_gap"] += 1
                continue
            if abs(cur[3] - cur[4]) > settings["sync_tolerance_s"]:
                rejected["depth_sync"] += 1
                continue
            if not np.isfinite(cur[5]).all() or not np.isfinite(prev[5]).all():
                rejected["invalid_pose"] += 1
                continue
            if abs(np.linalg.det(cur[5][:3, :3])) < .5 \
                    or abs(np.linalg.det(prev[5][:3, :3])) < .5:
                rejected["singular_pose"] += 1
                continue
            rows.append({"id": f"scannet:{name}:rgb:{index:06d}", "dataset": "scannet",
                         "sequence": name, "camera": "rgb", "group": group(name),
                         "split": split, "previous_id": str(index - 1),
                         "current_id": str(index), "sens": str(path.resolve()),
                         "depth_encoding": "sens", "K": source.color_k[:3, :3].tolist(),
                         "pose": (cur[5] @ source.color_extrinsic).tolist(),
                         "previous_pose": (prev[5] @ source.color_extrinsic).tolist()})
    return rows, dict(rejected)
