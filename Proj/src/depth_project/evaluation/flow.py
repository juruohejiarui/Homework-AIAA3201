"""Optical-flow MAE diagnostics. This is not the paper's 3D TAE."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_project.io import write_json


def warp_and_valid(previous, current):
    def flow(source, target):
        return cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), cv2.cvtColor(target, cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )

    forward = flow(previous, current)
    backward = flow(current, previous)
    height, width = backward.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx + backward[:, :, 0]
    map_y = yy + backward[:, :, 1]

    def warp(values):
        return cv2.remap(values.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    warped_forward = warp(forward)
    residual = np.sum((backward + warped_forward) ** 2, axis=2)
    bound = 0.01 * (np.sum(backward ** 2, axis=2) + np.sum(warped_forward ** 2, axis=2)) + 0.5
    inside = (map_x >= 1) & (map_x < width - 2) & (map_y >= 1) & (map_y < height - 2)
    return warp, inside & (residual <= bound)


def evaluate_flow(frames, depths, kind):
    if depths.shape != frames.shape[:3]:
        raise ValueError("Frames and predictions must have matching shapes")
    lower, upper = None, None
    if kind == "relative_inverse":
        lower, upper = np.percentile(depths, [1, 99])
        if upper - lower < 1e-6:
            raise ValueError("Constant relative prediction")
        normalized = (depths - lower) / (upper - lower)
        proxy = 1 / np.maximum(depths, 1e-6)
        unit = "fraction of whole-clip P99-P1 range"
    else:
        normalized = depths
        proxy = np.maximum(depths, 1e-6)
        unit = "m"
    rows = []
    first_median = float(np.median(proxy[0]))
    for index in range(1, len(frames)):
        warp, mask = warp_and_valid(frames[index - 1], frames[index])
        if not mask.any():
            raise ValueError(f"No valid flow correspondences at frame {index}")
        difference = np.abs(normalized[index] - warp(normalized[index - 1]))
        log_ratio = np.log(np.maximum(proxy[index], 1e-6))
        log_ratio -= np.log(np.maximum(warp(proxy[index - 1]), 1e-6))
        rows.append({"frame": index, "valid_fraction": float(mask.mean()),
                     "flow_aligned_mae": float(difference[mask].mean()),
                     "apparent_log_scale_step": float(np.median(log_ratio[mask])),
                     "median_scale_ratio_to_first": float(np.median(proxy[index]) / first_median)})
    return {"kind": kind, "unit": unit, "gauge": [lower, upper], "per_frame": rows,
            "mean_flow_aligned_mae": float(np.mean([row["flow_aligned_mae"] for row in rows])),
            "note": "Moving-camera scale proxies include real geometry changes; this is not 3D TAE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.frames) as source:
        frames = source["frames"]
    with np.load(args.predictions) as source:
        result = evaluate_flow(frames, source["depths"], str(source["kind"]))
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
