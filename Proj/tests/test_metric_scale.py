"""Regression checks for held-out scale evaluation."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from depth_project.data.nyuv2 import load_standard_splits
from depth_project.evaluation.metric_scale import compare


class MetricScaleTests(TestCase):
    def test_fits_on_development_and_scales_before_clipping(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dev_pred, dev_gt, test_pred, test_gt = (
                root / name for name in ("dev_pred", "dev_gt", "test_pred", "test_gt")
            )
            for folder, name, depth in (
                (dev_pred, "0001", 8), (dev_gt, "0001", 4),
                (test_pred, "0002", 12), (test_gt, "0002", 7),
            ):
                folder.mkdir()
                np.save(folder / name, np.full((480, 640), depth, dtype=np.float32))
            split = root / "split.mat"
            split.write_bytes(b"test fixture")
            with patch("depth_project.evaluation.metric_scale.load_standard_splits",
                       return_value={"development": [1], "test": [2]}):
                result = compare(split, dev_pred, dev_gt, test_pred, test_gt, root / "output")
            self.assertEqual(result["scale"], 0.5)
            self.assertEqual(result["baseline"]["rmse"], 3.0)
            self.assertEqual(result["calibrated"]["rmse"], 1.0)
            self.assertEqual(result["calibrated"]["d1"], 1.0)

    def test_rejects_a_split_with_test_ids_in_development(self):
        with TemporaryDirectory() as directory:
            split = Path(directory) / "splits.mat"
            savemat(split, {"trainNdxs": np.arange(1, 796), "testNdxs": np.arange(795, 1449)})
            with self.assertRaisesRegex(ValueError, "splits overlap"):
                load_standard_splits(split)
