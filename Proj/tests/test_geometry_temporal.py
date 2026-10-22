import cv2
import numpy as np

from depth_project.evaluation.flow import warp_and_valid
from depth_project.evaluation.metrics import align_inverse_depth, depth_metrics
from depth_project.stereo.geometry import depth_from_disparity, read_pfm


def test_middlebury_doffs_and_meters():
    disparity = np.array([8.0, 18.0, -1.0, np.inf], np.float32)
    depth = depth_from_disparity(disparity, 100, 0.1, 2)
    np.testing.assert_allclose(depth[:2], [1, 0.5])
    assert np.isnan(depth[2:]).all()


def test_known_depth_metrics():
    metrics = depth_metrics(np.array([2.0, 4.0]), np.array([1.0, 2.0]))
    actual = [metrics[key] for key in ["abs_rel", "sq_rel", "rmse", "rmse_log"]]
    np.testing.assert_allclose(actual, [1, 1.5, np.sqrt(2.5), np.log(2)])
    assert metrics["d1"] == metrics["d2"] == metrics["d3"] == 0


def test_pfm_byte_order_and_vertical_axis(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "depth.pfm"
    path.write_bytes(b"Pf\n4 3\n-1.0\n" + np.flipud(values).astype("<f4").tobytes())
    np.testing.assert_array_equal(read_pfm(path), values)


def test_backward_warp_direction():
    random = np.random.default_rng(5)
    previous = random.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 2], [0, 1, 0]]), (128, 96))
    warp, valid = warp_and_valid(previous, current)
    ramp = np.broadcast_to(np.arange(128, dtype=np.float32), (96, 128))
    yy, xx = np.indices(ramp.shape)
    mask = valid & (xx > 12) & (xx < 116) & (yy > 12) & (yy < 84)
    assert mask.sum() > 4000
    assert np.median(np.abs(warp(ramp)[mask] - (ramp - 2)[mask])) < 0.15


def test_inverse_ssi_negative_slope_is_retained():
    prediction = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[1.0, 2.0, 4.0]])
    aligned, diagnostic = align_inverse_depth(prediction, target, np.ones_like(target, dtype=bool))
    assert diagnostic["constrained_to_constant"]
    assert diagnostic["unconstrained_scale"] < 0
    np.testing.assert_allclose(aligned, 1 / np.mean(1 / target))


def test_middlebury_pfm_nonunit_display_scale(tmp_path):
    values = np.array([[16.0, 32.0], [48.0, 64.0]], dtype=np.float32)
    path = tmp_path / "middlebury.pfm"
    path.write_bytes(b"Pf\n2 2\n-0.003922\n" + np.flipud(values).astype("<f4").tobytes())
    np.testing.assert_array_equal(read_pfm(path), values)
