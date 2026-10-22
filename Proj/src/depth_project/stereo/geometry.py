"""Middlebury calibration and pixel-disparity geometry."""
from pathlib import Path

import numpy as np


def read_pfm(path):
    with Path(path).open("rb") as stream:
        tag = stream.readline().decode().strip()
        line = stream.readline().decode().strip()
        while line.startswith("#"):
            line = stream.readline().decode().strip()
        width, height = map(int, line.split())
        scale = float(stream.readline())
        values = np.fromfile(stream, "<f4" if scale < 0 else ">f4")
    shape = (height, width, 3) if tag == "PF" else (height, width)
    # Middlebury stores pixel disparities directly; the scale sign is endianness.
    return np.flipud(values.reshape(shape)).copy()


def read_calibration(path):
    fields = dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)
    camera = fields["cam0"].strip("[]").replace(";", " ").split()
    return float(camera[0]), float(fields["baseline"]) / 1000, float(fields["doffs"]), int(fields["ndisp"])


def depth_from_disparity(disparity, focal, baseline, offset):
    depth = np.full(disparity.shape, np.nan, np.float32)
    valid = np.isfinite(disparity) & (disparity >= 0) & (disparity + offset > 0)
    depth[valid] = focal * baseline / (disparity[valid] + offset)
    return depth


def sensitivity(disparity, focal, baseline, offset):
    depth = depth_from_disparity(disparity, focal, baseline, offset)
    rows = []
    for lower, upper in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1000)]:
        mask = np.isfinite(depth) & (depth >= lower) & (depth < upper)
        if not mask.any():
            continue
        row = {"range_m": [lower, upper], "pixels": int(mask.sum())}
        for delta in [-1, -0.5, 0.5, 1]:
            changed = depth_from_disparity(disparity + delta, focal, baseline, offset)
            valid = mask & np.isfinite(changed)
            row[str(delta)] = {
                "valid_pixels": int(valid.sum()),
                "mae_m": float(np.abs(changed[valid] - depth[valid]).mean()) if valid.any() else None,
            }
        rows.append(row)
    return rows
