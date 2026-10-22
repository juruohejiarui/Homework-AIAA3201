"""Export the standard 654 NYUv2 labeled test pairs."""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.io import loadmat

from depth_project.io import write_json


def export_test_set(mat_path, split_path, output):
    test_ids = loadmat(split_path)["testNdxs"].ravel().astype(int)
    output = Path(output)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "gt").mkdir(exist_ok=True)
    with h5py.File(mat_path) as dataset:
        for image_id in test_ids:
            index = image_id - 1
            rgb = dataset["images"][index].transpose(2, 1, 0).astype(np.uint8)
            depth = dataset["depths"][index].T.astype(np.float32)
            name = f"{image_id:04d}"
            if not cv2.imwrite(str(output / "rgb" / f"{name}.png"), rgb[:, :, ::-1]):
                raise OSError(f"Cannot write image {name}")
            np.save(output / "gt" / f"{name}.npy", depth)
    write_json(output / "manifest.json", {
        "test_ids": test_ids.tolist(),
        "gt": "depths",
        "unit": "m",
        "count": len(test_ids),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_test_set(args.mat, args.split, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
