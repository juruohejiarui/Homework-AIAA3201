"""Run stereo matching on a Middlebury directory or one scene."""
import argparse
from pathlib import Path

import numpy as np

from depth_project.io import write_json
from depth_project.stereo.matching import match_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["bm", "sgbm"], default="sgbm")
    args = parser.parse_args()
    rows = []
    for calibration in sorted(args.data.rglob("calib.txt")):
        scene = calibration.parent
        if not (scene / "disp0GT.pfm").exists():
            continue
        arrays, metrics = match_scene(scene, args.method)
        destination = args.output / scene.name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "stereo.npz", **arrays)
        rows.append(metrics)
    if not rows:
        raise FileNotFoundError(f"No labeled stereo scenes in {args.data}")
    write_json(args.output / "metrics.json", rows)
    print(f"Saved {len(rows)} scenes to {args.output}")


if __name__ == "__main__":
    main()
