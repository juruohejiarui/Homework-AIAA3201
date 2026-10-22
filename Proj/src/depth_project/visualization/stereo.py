"""Visualize stored stereo arrays without rerunning matching."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_project.visualization.colors import colorize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.arrays) as source:
        disparity = source["disparity"]
        panels = [colorize(np.where(disparity >= 0, disparity, np.nan), 0, np.nanmax(disparity)),
                  colorize(source["depth_m"], 0.1, 10, reverse=True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.concatenate(panels, axis=1))
    print(args.output)


if __name__ == "__main__":
    main()
