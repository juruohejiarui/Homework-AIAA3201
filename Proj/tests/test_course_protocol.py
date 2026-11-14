"""Course-critical boundary and aggregation checks, independent of model inference."""
import numpy as np
import pytest

from depth_project.course.common import frame_indices, nyu_score, stereo_score
from depth_project.stereo.geometry import depth_from_disparity


def test_stereo_positive_gt_and_invalid_predictions():
    gt = np.array([0, -1, np.inf, 2, 4, 6], np.float32)
    prediction = np.array([99, 99, 99, 0, -1, 8], np.float32)
    result = stereo_score(prediction, gt)
    assert result['gt_pixels'] == 3
    assert result['valid_predictions'] == 2
    assert result['coverage'] == 2/3
    assert result['mae_px'] == 2


def test_nyu_crop_range_and_clipping():
    gt = np.full((480, 640), np.nan, np.float32)
    pred = np.full_like(gt, 100)
    gt[45, 41] = 1
    pred[45, 41] = 1
    gt[470, 600] = 2
    pred[470, 600] = 2
    gt[44, 41] = 1  # outside crop
    gt[471, 41] = 1
    gt[46, 42] = 10  # strict depth boundary
    gt[46, 43] = np.float32(.001) - np.float32(1e-7)
    result = nyu_score(pred, gt)
    assert result == {'abs_rel': 0., 'rmse_m': 0., 'delta1': 1., 'valid_pixels': 2}
    pred[45, 41] = 20
    assert nyu_score(pred, gt)['abs_rel'] == 4.5
    pred[45, 41] = np.inf
    with pytest.raises(ValueError, match='Nonfinite'):
        nyu_score(pred, gt)


def test_video_keeps_real_frames_and_first_100_only():
    np.testing.assert_array_equal(frame_indices(70), np.arange(70))
    np.testing.assert_array_equal(frame_indices(150), np.arange(100))


def test_calibrated_depth_keeps_pixel_disparity_and_meters():
    cv_fixed = np.array([128, 288], np.float32)
    actual = depth_from_disparity(cv_fixed/16, 100, 100/1000, 2)
    np.testing.assert_allclose(actual, [1, .5])
