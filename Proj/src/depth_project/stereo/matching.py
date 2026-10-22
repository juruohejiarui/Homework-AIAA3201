"""BM/SGBM estimation and left-right consistency, without visualization."""
import time

import cv2
import numpy as np

from depth_project.evaluation.metrics import depth_metrics
from depth_project.stereo.geometry import depth_from_disparity, read_calibration, read_pfm, sensitivity


def create_matcher(method, disparities, minimum=0):
    if method == "bm":
        matcher = cv2.StereoBM_create(disparities, 15)
        matcher.setMinDisparity(minimum)
        return matcher
    return cv2.StereoSGBM_create(
        minDisparity=minimum, numDisparities=disparities, blockSize=5,
        P1=8 * 25, P2=32 * 25, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def match_scene(scene, method):
    left = cv2.imread(str(scene / "im0.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(scene / "im1.png"), cv2.IMREAD_GRAYSCALE)
    focal, baseline, offset, maximum = read_calibration(scene / "calib.txt")
    count = (maximum + 15) // 16 * 16
    started = time.perf_counter()
    disparity = create_matcher(method, count).compute(left, right).astype(np.float32) / 16
    minimum_right = -count + 1
    reverse = create_matcher(method, count, minimum_right).compute(right, left)
    reverse = reverse.astype(np.float32) / 16
    seconds = time.perf_counter() - started
    yy, xx = np.indices(disparity.shape)
    matched_x = np.rint(xx - disparity).astype(int)
    inside = (matched_x >= 0) & (matched_x < left.shape[1])
    sampled = reverse[yy, np.clip(matched_x, 0, left.shape[1] - 1)]
    valid = (disparity >= 0) & inside & (sampled > minimum_right - 1)
    consistent = valid & (np.abs(disparity + sampled) <= 1)
    ground_truth = read_pfm(scene / "disp0GT.pfm")
    gt_valid = np.isfinite(ground_truth) & (ground_truth >= 0) & (ground_truth + offset > 0)
    error = np.abs(disparity - ground_truth)
    evaluated = gt_valid & (disparity >= 0)
    metrics = {
        "scene": scene.name,
        "method": method,
        "seconds": seconds,
        "bad2_all_gt": float(((disparity < 0) | (error > 2))[gt_valid].mean()),
        "epe_valid_prediction": float(error[evaluated].mean()) if evaluated.any() else None,
        "prediction_coverage": float((disparity >= 0)[gt_valid].mean()),
        "lr_coverage": float(consistent[gt_valid].mean()),
        "sensitivity": sensitivity(ground_truth, focal, baseline, offset),
    }
    arrays = {
        "disparity": disparity,
        "right_disparity": reverse,
        "depth_m": depth_from_disparity(disparity, focal, baseline, offset),
        "lr_valid": consistent,
        "gt_disparity": ground_truth,
    }
    depth_mask = gt_valid & consistent & np.isfinite(arrays["depth_m"])
    truth_depth = depth_from_disparity(ground_truth, focal, baseline, offset)
    metrics["depth_metrics_lr_valid"] = None
    if depth_mask.any():
        metrics["depth_metrics_lr_valid"] = depth_metrics(arrays["depth_m"][depth_mask], truth_depth[depth_mask])
    nonoccluded_path = scene / "mask0nocc.png"
    if nonoccluded_path.exists():
        nonoccluded = cv2.imread(str(nonoccluded_path), 0) == 255
        mask = gt_valid & nonoccluded
        if mask.any():
            metrics["bad2_nocc"] = float(((disparity < 0) | (error > 2))[mask].mean())
    return arrays, metrics
