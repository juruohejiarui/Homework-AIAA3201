"""Explicit frame-ID and TUM timestamp association."""
import re

import numpy as np


def natural_order(paths):
    def key(path):
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]
    return sorted(paths, key=key)


def pair_frame_ids(images, depths):
    depth_by_id = {path.stem: path for path in depths}
    return [(image, depth_by_id[image.stem]) for image in natural_order(images)]


def read_timestamps(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        timestamp, relative = line.split()[:2]
        rows.append((float(timestamp), path.parent / relative))
    return sorted(rows)


def associate_tum(scene, tolerance=0.02):
    images = read_timestamps(scene / "rgb.txt")
    depths = read_timestamps(scene / "depth.txt")
    depth_times = np.array([time for time, path in depths])
    candidates = []
    for image_index, (time, path) in enumerate(images):
        lower, upper = np.searchsorted(depth_times, [time - tolerance, time + tolerance])
        for depth_index in range(lower, upper):
            candidates.append((abs(time - depth_times[depth_index]), image_index, depth_index))
    used_images = set()
    used_depths = set()
    matches = []
    for error, image_index, depth_index in sorted(candidates):
        if image_index in used_images or depth_index in used_depths:
            continue
        used_images.add(image_index)
        used_depths.add(depth_index)
        matches.append((image_index, depth_index, error))
    return [(images[i][1], depths[j][1], {
        "source_index": i, "rgb_timestamp": images[i][0],
        "depth_timestamp": depths[j][0], "sync_error_s": error,
    }) for i, j, error in sorted(matches)]
