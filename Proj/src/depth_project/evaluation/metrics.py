"""Pixel metrics and the explicitly constrained development SSI alignment."""
import cv2
import numpy as np


def depth_metrics(prediction, target):
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "abs_rel": float(np.mean(np.abs(prediction - target) / target)),
        "sq_rel": float(np.mean((prediction - target) ** 2 / target)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(prediction) - np.log(target)) ** 2))),
        "d1": float(np.mean(ratio < 1.25)),
        "d2": float(np.mean(ratio < 1.25 ** 2)),
        "d3": float(np.mean(ratio < 1.25 ** 3)),
    }


def align_inverse_depth(prediction, target, mask):
    inverse = prediction[mask].astype(np.float64)
    truth = 1 / target[mask].astype(np.float64)
    centered = inverse - inverse.mean()
    variance = float(np.mean(centered ** 2))
    raw_scale = float(np.mean(centered * (truth - truth.mean())) / variance) if variance > 1e-15 else 0
    scale = max(raw_scale, 0)
    shift = float(truth.mean() - scale * inverse.mean())
    return 1 / np.maximum(scale * prediction + shift, 1e-6), {
        "unconstrained_scale": raw_scale,
        "scale": scale,
        "shift": shift,
        "constrained_to_constant": bool(raw_scale <= 0 or variance <= 1e-15),
    }


def boundary_score(prediction, target, valid, threshold=0.05, tolerance=1):
    """Custom log-depth edge score, not Depth Pro's SI-boundary F1."""
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    def edges(depth):
        logarithm = np.log(np.clip(np.nan_to_num(depth, nan=1), 1e-6, None)).astype(np.float32)
        horizontal = cv2.Sobel(logarithm, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
        vertical = cv2.Sobel(logarithm, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
        return (np.hypot(horizontal, vertical) > threshold) & valid

    predicted = edges(prediction)
    actual = edges(target)
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    nearby_actual = cv2.dilate(actual.astype(np.uint8), kernel) > 0
    nearby_predicted = cv2.dilate(predicted.astype(np.uint8), kernel) > 0
    precision = float((predicted & nearby_actual).sum() / max(predicted.sum(), 1))
    recall = float((actual & nearby_predicted).sum() / max(actual.sum(), 1))
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
