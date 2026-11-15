"""Fit one NYUv2 development scale and compare it against uncalibrated test depth."""

import argparse
from pathlib import Path

import numpy as np

from depth_project.data.nyuv2 import load_standard_splits
from depth_project.evaluation.images import evaluate_images, evaluation_mask
from depth_project.io import file_hash, write_json


SHAPE = (480, 640)
MINIMUM = 0.001
MAXIMUM = 10.0
METRICS = ("abs_rel", "rmse", "d1")


def require_maps(directory, names):
    directory = Path(directory)
    expected = set(names)
    actual = {path.name for path in directory.glob("*.npy") if path.is_file()}
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{directory}: missing {len(missing)} maps {missing[:3]}, "
                         f"unexpected {len(extra)} maps {extra[:3]}")


def fit_scale(predictions, ground_truth, names):
    ratios = []
    for name in names:
        depth = np.load(ground_truth / name).astype(np.float64)
        prediction = np.load(predictions / name).astype(np.float64)
        if depth.shape != SHAPE or prediction.shape != SHAPE:
            raise ValueError(f"{name}: expected {SHAPE}, got {depth.shape} and {prediction.shape}")
        valid = evaluation_mask(depth, "eigen", MINIMUM, MAXIMUM)
        valid &= np.isfinite(prediction) & (prediction > 0)
        if not valid.any():
            raise ValueError(f"{name}: no valid development pixels for scale fitting")
        ratios.append(float(np.median(depth[valid] / prediction[valid])))
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid fitted scale: {scale}")
    return scale


def compare(split, dev_predictions, dev_gt, test_predictions, test_gt, output):
    splits = load_standard_splits(split)
    development = [f"{index:04d}.npy" for index in splits["development"]]
    test = [f"{index:04d}.npy" for index in splits["test"]]
    require_maps(dev_predictions, development)
    require_maps(dev_gt, development)
    scale = fit_scale(dev_predictions, dev_gt, development)
    calibration = {
        "scale": scale,
        "fit": "median of per-image median ground-truth / raw prediction ratios",
        "development_count": len(development),
        "split_sha256": file_hash(split),
        "crop_rows_columns": [[45, 471], [41, 601]],
        "depth_bounds_m": [MINIMUM, MAXIMUM],
    }
    write_json(output / "scale.json", calibration)

    require_maps(test_predictions, test)
    require_maps(test_gt, test)
    for name in test:
        if np.load(test_gt / name, mmap_mode="r").shape != SHAPE:
            raise ValueError(f"{name}: expected a {SHAPE} test depth map")
    baseline = evaluate_images(test_predictions, test_gt)
    calibrated = evaluate_images(test_predictions, test_gt, scale=scale)
    before = {key: baseline["summary"][key] for key in METRICS}
    after = {key: calibrated["summary"][key] for key in METRICS}
    result = {
        **calibration,
        "test_count": len(test),
        "aggregation": "equal weight per test image",
        "baseline": before,
        "calibrated": after,
        "calibrated_minus_baseline": {key: after[key] - before[key] for key in METRICS},
        "per_image": [
            {"image": raw["image"], "baseline": {key: raw[key] for key in METRICS},
             "calibrated": {key: adjusted[key] for key in METRICS}}
            for raw, adjusted in zip(baseline["per_image"], calibrated["per_image"], strict=True)
        ],
    }
    write_json(output / "comparison.json", result)
    return {key: value for key, value in result.items() if key != "per_image"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("split", "dev-predictions", "dev-gt", "test-predictions", "test-gt", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.split, args.dev_predictions, args.dev_gt, args.test_predictions,
                     args.test_gt, args.output)
    print(f"Development scale: {result['scale']:.10f}")
    for label in ("baseline", "calibrated", "calibrated_minus_baseline"):
        print(label, result[label])


if __name__ == "__main__":
    main()
