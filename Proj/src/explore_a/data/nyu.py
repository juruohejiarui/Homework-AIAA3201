"""NYUv2 raw pairing with labeled test-room exclusion and sensor projection at read time."""
from collections import Counter
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.io import loadmat

from depth_project.data.nyu_projection import RGB_INTRINSICS
from depth_project.data.nyuv2_raw import timestamp
from depth_project.data.protocol import NYU_SCENES
from explore_a.data.history import historical_scenes
from explore_a.data.splits import development_groups


def room_id(name):
    # basement_0001a/b are repeated views of one room, not independent scenes.
    return re.sub(r"(?<=\d)[a-z]$", "", name)


def test_rooms(root):
    ids = loadmat(root / "splits.mat")["testNdxs"].ravel().astype(int) - 1
    with h5py.File(root / "nyu_depth_v2_labeled.mat") as source:
        refs = source["scenes"][:].ravel()
        names = ["".join(chr(int(v)) for v in source[refs[i]][:].ravel()) for i in ids]
    return {room_id(name) for name in names + NYU_SCENES + list(historical_scenes("nyuv2"))}


def prepare_nyu(root, settings, aliases):
    raw = settings.get("nyu_raw_root")
    raw = Path(raw).expanduser() if raw else root / "raw"
    scenes = sorted({p.parent for p in raw.rglob("r-*.ppm")})
    if not scenes:
        raise FileNotFoundError(f"No NYUv2 Raw RGB found: {raw}")
    reserved = test_rooms(root)
    group = lambda name: aliases.get(name, room_id(name))
    protected = reserved | {group(s.name) for s in scenes if room_id(s.name) in reserved}
    eligible = [s for s in scenes if room_id(s.name) not in reserved
                and group(s.name) not in protected]
    dev = development_groups([group(s.name) for s in eligible],
                             settings["seed"], settings["dev_fraction"])
    fx, fy, cx, cy = RGB_INTRINSICS
    rows, rejected = [], Counter({"reserved_scenes": len(scenes) - len(eligible),
                                  "raw_scenes_discovered": len(scenes),
                                  "eligible_training_development_scenes": len(eligible)})
    for scene in scenes:
        split = "test" if room_id(scene.name) in reserved or group(scene.name) in protected \
            else "dev" if group(scene.name) in dev else "train"
        images = sorted(scene.glob("r-*.ppm"), key=timestamp)
        rejected["rgb_frames_discovered"] += len(images)
        depths = sorted(scene.glob("d-*.pgm"), key=timestamp)
        times = np.array([timestamp(p) for p in depths])
        if not len(times):
            rejected["no_sensor_depth"] += 1
            continue
        def match(image):
            t = timestamp(image)
            i = int(np.searchsorted(times, t))
            candidates = [j for j in (i - 1, i) if 0 <= j < len(depths)]
            nearest = min(candidates, key=lambda j: abs(times[j] - t))
            return depths[nearest] if abs(times[nearest] - t) <= settings["sync_tolerance_s"] \
                else None
        for index in range(1, len(images)):
            previous, current = images[index - 1:index + 1]
            if not 0 < timestamp(current) - timestamp(previous) <= settings["max_frame_gap_s"]:
                rejected["rgb_gap"] += 1
                continue
            gt, previous_gt = match(current), match(previous)
            if gt is None:
                rejected["depth_sync"] += 1
                continue
            rows.append({"id": f"nyuv2:{scene.name}:rgb:{current.stem}", "dataset": "nyuv2",
                         "sequence": scene.name, "camera": "rgb", "group": group(scene.name),
                         "split": split,
                         "previous_id": previous.stem, "current_id": current.stem,
                         "previous_rgb": str(previous.resolve()),
                         "current_rgb": str(current.resolve()), "gt": str(gt.resolve()),
                         "previous_gt": str(previous_gt.resolve()) if previous_gt else None,
                         "depth_encoding": "nyu_raw", "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                         "pose": None})
    report = dict(rejected)
    report["source"] = "official_raw"
    report["selection"] = "all eligible adjacent Raw RGB pairs; no labeled-only or frame limit"
    for split in ("train", "dev", "test"):
        selected = [row for row in rows if row["split"] == split]
        report[split + "_pairs"] = len(selected)
        report[split + "_groups"] = len({row["group"] for row in selected})
    return rows, report
