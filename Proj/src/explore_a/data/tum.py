"""TUM: pair consecutive RGB first, then associate each RGB to sensor depth."""
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from depth_project.data.association import associate_tum, read_timestamps

INTRINSICS = {
    "freiburg1": [517.3, 516.5, 318.6, 255.3],
    "freiburg2": [520.9, 521.0, 325.1, 249.7],
    "freiburg3": [535.4, 539.2, 320.1, 247.6],
}


def prepare_tum(root, settings, aliases):
    rows, rejected = [], Counter()
    scenes = sorted(root.glob("rgbd_dataset_freiburg*"))
    scenes += sorted((root / "sequences").glob("rgbd_dataset_freiburg*"))
    seen = set()
    for scene in scenes:
        if not scene.is_dir() or scene.resolve() in seen:
            continue
        seen.add(scene.resolve())
        name = scene.name
        if any(word in name for word in ("validation", "calibration", "checkerboard")):
            rejected["validation_or_calibration"] += 1
            continue
        if not (scene / "rgb.txt").is_file() or not (scene / "depth.txt").is_file():
            rejected["missing_rgb_or_depth_list"] += 1
            continue
        family = next(f for f in INTRINSICS if f in name)
        split = {"freiburg1": "dev", "freiburg2": "train", "freiburg3": "test"}[family]
        images = read_timestamps(scene / "rgb.txt")
        poses = trajectory(scene)
        matched = associate_tum(scene, settings["sync_tolerance_s"])
        by_image = {str(image): (depth, meta) for image, depth, meta in matched}
        fx, fy, cx, cy = INTRINSICS[family]
        matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        for index in range(1, len(images)):
            time, current = images[index]
            previous_time, previous = images[index - 1]
            if not 0 < time - previous_time <= settings["max_frame_gap_s"]:
                rejected["rgb_gap"] += 1
                continue
            if str(current) not in by_image:
                rejected["depth_sync"] += 1
                continue
            gt, metadata = by_image[str(current)]
            previous_gt = by_image.get(str(previous), (None,))[0]
            if not all(p.is_file() for p in (current, previous, gt)):
                rejected["missing_file"] += 1
                continue
            row = {"id": f"tum:{name}:rgb:{current.stem}", "dataset": "tum",
                         "sequence": name, "camera": "rgb", "group": aliases.get(name, name),
                         "split": split, "previous_id": previous.stem,
                         "current_id": current.stem, "previous_rgb": str(previous.resolve()),
                         "current_rgb": str(current.resolve()), "gt": str(gt.resolve()),
                         "previous_gt": str(previous_gt.resolve()) if previous_gt else None,
                         "depth_encoding": "tum_png", "K": matrix, "pose": None,
                         "rgb_timestamp": time, "previous_timestamp": previous_time,
                         "sync_error_s": metadata["sync_error_s"]}
            current_pose = interpolate_pose(poses, time, settings["sync_tolerance_s"])
            previous_pose = interpolate_pose(poses, previous_time, settings["sync_tolerance_s"])
            if current_pose is not None and previous_pose is not None:
                row.update(pose=current_pose.tolist(), previous_pose=previous_pose.tolist())
            rows.append(row)
    # Same place aliases override per-release defaults conservatively: test > dev > train.
    priority = {"train": 0, "dev": 1, "test": 2}
    group_split = {}
    for row in rows:
        old = group_split.get(row["group"], "train")
        group_split[row["group"]] = max((old, row["split"]), key=priority.get)
    for row in rows:
        row["split"] = group_split[row["group"]]
    return rows, dict(rejected)


def trajectory(scene):
    """Optional camera-to-world poses; absent trajectory never fabricates ground truth."""
    path = scene / "groundtruth.txt"
    return np.atleast_2d(np.loadtxt(path)) if path.exists() else None


def interpolate_pose(poses, time, tolerance):
    """TUM poses are camera-to-world, quaternion order x,y,z,w; interpolate only nearby GT."""
    if poses is None or len(poses) < 2:
        return None
    after = int(np.searchsorted(poses[:, 0], time, side="right"))
    if after == 0 or after == len(poses):
        return None
    before = after - 1
    times = poses[[before, after], 0]
    if max(time - times[0], times[1] - time) > tolerance:
        return None
    fraction = (time - times[0]) / (times[1] - times[0])
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = Slerp(times, Rotation.from_quat(poses[[before, after], 4:8]))(time).as_matrix()
    matrix[:3, 3] = poses[before, 1:4] * (1 - fraction) + poses[after, 1:4] * fraction
    return matrix
