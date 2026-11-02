"""Regression for OpenCV drawing on transposed RGB and reversed channel views."""
import cv2
import numpy as np
import torch

from explore_a.evaluation.visualization import save_panel


def test_save_panel_with_real_chw_rgb_layout(tmp_path):
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[:] = (40, 100, 200)
    chw = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None]
    depth = torch.full((1, 1, 48, 64), 2.)
    valid = torch.ones_like(depth, dtype=torch.bool)
    data = {"current_rgb": chw, "image_valid": valid, "gt": depth,
            "current_depth": depth, "gt_valid": valid}
    output = {"depth": depth, "historical_weight": depth * .1,
              "visibility": valid.float()}
    path = tmp_path / "visualizations/sample.jpg"
    save_panel(path, data, output, 10.)
    saved = cv2.imread(str(path))
    assert saved.shape == (48, 64 * 6, 3)
    np.testing.assert_allclose(saved[40, 10], [200, 100, 40], atol=3)
