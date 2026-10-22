"""Plot per-frame diagnostics from previously saved JSON, without writing a report."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_project.io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for path in args.metrics:
        rows = read_json(path)["per_frame"]
        frames = [row["frame"] for row in rows]
        axes[0].plot(frames, [row["flow_aligned_mae"] for row in rows], label=path.parent.name)
        axes[1].plot(frames, [row["median_scale_ratio_to_first"] for row in rows])
    axes[0].set_ylabel("Flow-aligned MAE")
    axes[0].legend()
    axes[1].set_ylabel("Median scale proxy")
    axes[1].set_xlabel("Frame")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
