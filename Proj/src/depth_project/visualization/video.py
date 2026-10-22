"""Render RGB and depth comparisons with a fixed scale across each whole clip."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_project.io import write_json
from depth_project.visualization.colors import colorize


def render_video(frames_path, prediction_paths, output, minimum=0.1, maximum=10):
    with np.load(frames_path) as source:
        frames = source["frames"]
        fps = float(source["fps"])
    predictions = []
    ranges = []
    kinds = []
    for path in prediction_paths:
        with np.load(path) as source:
            depths = source["depths"]
            kind = str(source["kind"])
        if depths.shape != frames.shape[:3]:
            raise ValueError(f"Frame count or resolution mismatch: {path}")
        lower, upper = (minimum, maximum) if kind == "metric_m" else np.percentile(depths, [1, 99])
        predictions.append(depths)
        ranges.append([float(lower), float(upper)])
        kinds.append(kind)
    height, width = frames.shape[1:3]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width * (len(predictions) + 1), height + 40))
    if not writer.isOpened():
        raise OSError(f"Cannot open video output: {output}")
    try:
        for index, rgb in enumerate(frames):
            panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
            labels = ["RGB"]
            for path, depths, kind, (lower, upper) in zip(prediction_paths, predictions, kinds, ranges):
                panels.append(colorize(depths[index], lower, upper, reverse=kind == "metric_m"))
                labels.append(f"{path.parent.name}: {kind}")
            canvas = np.zeros((height + 40, width * len(panels), 3), np.uint8)
            for column, (panel, label) in enumerate(zip(panels, labels)):
                canvas[40:, column * width:(column + 1) * width] = panel
                cv2.putText(canvas, label, (column * width + 8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            writer.write(canvas)
    finally:
        writer.release()
    write_json(output.with_suffix(".json"), {"ranges": ranges, "kinds": kinds,
                                           "note": "One fixed scale per whole clip, no per-frame normalization"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=float, default=0.1)
    parser.add_argument("--maximum", type=float, default=10)
    args = parser.parse_args()
    render_video(args.frames, args.predictions, args.output, args.minimum, args.maximum)
    print(args.output)


if __name__ == "__main__":
    main()
