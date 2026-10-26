"""Decode metric depth and preserve sparse observations when fitting a padded canvas."""
import cv2
import numpy as np

from depth_project.data.nyu_projection import project_depth, undistort
from depth_project.io import read_rgb
from explore_a.data.sens import sensor


def read_frame(row, previous=False):
    prefix = "previous" if previous else "current"
    if row["depth_encoding"] == "sens":
        return sensor(row["sens"]).read(int(row[f"{prefix}_id"]))
    rgb = read_rgb(row[f"{prefix}_rgb"])
    path = row.get("previous_gt") if previous else row["gt"]
    if row["depth_encoding"] == "nyu_raw":
        if path:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise OSError(path)
            depth, rgb = project_depth(raw, rgb)
            return rgb, depth
        rgb = np.stack([undistort(rgb[:, :, c].astype(float), "rgb") for c in range(3)], -1)
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8), None
    if path is None:
        return rgb, None
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(path)
    factor = {"kitti_png": 256.0, "tum_png": 5000.0}[row["depth_encoding"]]
    return rgb, raw.astype(np.float32) / factor


def canvas_transform(shape, canvas):
    height, width = shape[:2]
    scale = min(canvas[0] / height, canvas[1] / width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    left, top = (canvas[1] - size[0]) // 2, (canvas[0] - size[1]) // 2
    return size, left, top


def fit_image(image, canvas):
    size, left, top = canvas_transform(image.shape, canvas)
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    output = np.zeros((*canvas, *image.shape[2:]), image.dtype)
    output[top:top + size[1], left:left + size[0]] = resized
    valid = np.zeros(canvas, bool)
    valid[top:top + size[1], left:left + size[0]] = True
    return output, valid


def fit_depth(depth, canvas):
    """Forward-splat measured pixels; nearest surface wins, missing GT stays missing."""
    size, left, top = canvas_transform(depth.shape, canvas)
    y, x = np.where(np.isfinite(depth) & (depth > 0))
    target_x = np.clip(np.floor((x + .5) * size[0] / depth.shape[1]).astype(int), 0, size[0] - 1)
    target_y = np.clip(np.floor((y + .5) * size[1] / depth.shape[0]).astype(int), 0, size[1] - 1)
    output = np.full(np.prod(canvas), np.inf, np.float32)
    np.minimum.at(output, (target_y + top) * canvas[1] + target_x + left, depth[y, x])
    output[~np.isfinite(output)] = 0
    return output.reshape(canvas)


def resize_intrinsics(matrix, shape, canvas):
    size, left, top = canvas_transform(shape, canvas)
    sx, sy = size[0] / shape[1], size[1] / shape[0]
    # OpenCV resize uses pixel centers: x_new = (x_old + .5) * scale - .5 + pad.
    affine = np.array([[sx, 0, left + .5 * sx - .5],
                       [0, sy, top + .5 * sy - .5], [0, 0, 1]])
    return affine @ np.asarray(matrix)


def depth_range(dataset, settings):
    maximum = settings["kitti_max_depth_m" if dataset == "kitti" else "indoor_max_depth_m"]
    return settings["min_depth_m"], maximum
