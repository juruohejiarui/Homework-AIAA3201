"""Export the standard NYUv2 labeled development or test pairs."""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.io import loadmat

from depth_project.io import write_json


SUBSETS = {"development": ("trainNdxs", 795), "test": ("testNdxs", 654)}


def load_standard_splits(split_path):
    source = loadmat(split_path)
    splits = {}
    for subset, (key, count) in SUBSETS.items():
        values = source[key].ravel()
        if (len(values) != count or not np.isfinite(values).all()
                or np.any(values < 1) or np.any(values > 1449)
                or np.any(values != np.floor(values)) or len(np.unique(values)) != count):
            raise ValueError(f"Invalid NYUv2 {subset} split: expected {count} distinct image IDs")
        splits[subset] = values.astype(int)
    if np.intersect1d(splits["development"], splits["test"]).size:
        raise ValueError("NYUv2 development and test splits overlap")
    return splits


def export_subset(mat_path, split_path, output, subset):
    image_ids = load_standard_splits(split_path)[subset]
    output = Path(output)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "gt").mkdir(exist_ok=True)
    with h5py.File(mat_path) as dataset:
        for image_id in image_ids:
            index = image_id - 1
            rgb = dataset["images"][index].transpose(2, 1, 0).astype(np.uint8)
            depth = dataset["depths"][index].T.astype(np.float32)
            name = f"{image_id:04d}"
            if not cv2.imwrite(str(output / "rgb" / f"{name}.png"), rgb[:, :, ::-1]):
                raise OSError(f"Cannot write image {name}")
            np.save(output / "gt" / f"{name}.npy", depth)
    write_json(output / "manifest.json", {
        ("test_ids" if subset == "test" else "development_ids"): image_ids.tolist(),
        "gt": "depths",
        "unit": "m",
        "count": len(image_ids),
    })


def export_test_set(mat_path, split_path, output):
    export_subset(mat_path, split_path, output, "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=SUBSETS, default="test")
    args = parser.parse_args()
    export_subset(args.mat, args.split, args.output, args.subset)
    print(args.output)


if __name__ == "__main__":
    main()
