"""Evaluate metric or oracle-aligned predictions on NYUv2 images."""
import argparse
from pathlib import Path

import numpy as np

from depth_project.evaluation.metrics import align_inverse_depth, boundary_score, depth_metrics
from depth_project.io import write_json


def evaluate_images(predictions, ground_truth, alignment="none", crop="eigen", minimum=0.001, maximum=10):
    rows = []
    for path in sorted(Path(ground_truth).glob("*.npy")):
        target = np.load(path).astype(np.float64)
        prediction = np.load(Path(predictions) / path.name).astype(np.float64)
        if target.shape != prediction.shape:
            raise ValueError(f"RGB/GT resolution mismatch: {path.name}")
        mask = np.isfinite(target) & (target > minimum) & (target < maximum)
        if crop == "eigen":
            cropped = np.zeros_like(mask)
            cropped[45:471, 41:601] = True
            mask &= cropped
        diagnostic = None
        if alignment == "inverse-ssi":
            prediction, diagnostic = align_inverse_depth(prediction, target, mask)
        elif alignment == "inverse-median":
            prediction = 1 / np.maximum(prediction, 1e-6)
            prediction *= np.median(target[mask]) / np.median(prediction[mask])
        if not np.isfinite(prediction[mask]).all() or np.any(prediction[mask] <= 0):
            raise ValueError(f"Invalid depth prediction: {path.name}")
        prediction = np.clip(prediction, minimum, maximum)
        rows.append({"image": path.stem, **depth_metrics(prediction[mask], target[mask]),
                     "alignment": diagnostic, "boundary": boundary_score(prediction, target, mask)})
    if not rows:
        raise FileNotFoundError(f"No GT maps in {ground_truth}")
    keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["boundary"] = {key: float(np.mean([row["boundary"][key] for row in rows]))
                           for key in ["precision", "recall", "f1"]}
    summary["nonpositive_ssi_slope_count"] = sum(
        bool(row["alignment"] and row["alignment"]["constrained_to_constant"]) for row in rows
    )
    return {"summary": summary, "per_image": rows, "alignment": alignment,
            "crop": crop, "depth_bounds_m": [minimum, maximum],
            "protocol": "Development image evaluation; constrained SSI is not the official video protocol"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment", choices=["none", "inverse-ssi", "inverse-median"], default="none")
    parser.add_argument("--crop", choices=["eigen", "full"], default="eigen")
    parser.add_argument("--minimum", type=float, default=0.001)
    parser.add_argument("--maximum", type=float, default=10)
    args = parser.parse_args()
    result = evaluate_images(args.predictions, args.ground_truth, args.alignment,
                             args.crop, args.minimum, args.maximum)
    write_json(args.output, result)
    print(result["summary"])


if __name__ == "__main__":
    main()
