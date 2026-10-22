"""NYUv2 toolbox projection port, without hole filling; MATLAB parity pending."""
from functools import lru_cache

import numpy as np

RGB_INTRINSICS = (518.85790117450188, 519.46961112127485, 325.58244941119034, 253.73616633400465)
DEPTH_INTRINSICS = (582.62448167737955, 582.69103270988637, 313.04475870804731, 238.44389626620386)
RGB_DISTORTION = (.20796615318809061, -.58613825163911781, .00072231363135888329,
                  .0010479627195765181, .49856986684705107)
DEPTH_DISTORTION = (-.099897236553084481, .39065324602765344, .0019290592870229277,
                    -.0019422022475975055, -.51031725053400578)
ROTATION_VALUES = [.99997798940829263, .0050518419386157446, .0043011152014118693,
                   -.0050359919480810989, .99998051861143999, -.0036879781309514218,
                   -.0043196624923060242, .0036662365748484798, .99998394948385538]
ROTATION = np.linalg.inv(-np.array(ROTATION_VALUES).reshape(3, 3, order="F").T)
TRANSLATION = np.array([.025031875059141302, -.00029342312935846411, .00066238747008330102])


@lru_cache(maxsize=2)
def interpolation(camera):
    fx, fy, cx, cy = RGB_INTRINSICS if camera == "rgb" else DEPTH_INTRINSICS
    k1, k2, p1, p2, k3 = RGB_DISTORTION if camera == "rgb" else DEPTH_DISTORTION
    yy, xx = np.mgrid[:480, :640]
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    radius = x * x + y * y
    radial = 1 + k1 * radius + k2 * radius ** 2 + k3 * radius ** 3
    projected_x = fx * (x * radial + 2 * p1 * x * y + p2 * (radius + 2 * x * x)) + cx
    projected_y = fy * (y * radial + p1 * (radius + 2 * y * y) + 2 * p2 * x * y) + cy
    left = np.floor(projected_x).astype(int)
    top = np.floor(projected_y).astype(int)
    valid = (left >= 0) & (left <= 638) & (top >= 0) & (top <= 478)
    dx = (projected_x - left)[valid]
    dy = (projected_y - top)[valid]
    left = left[valid]
    top = top[valid]
    weights = np.array([(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx])
    neighbors = [(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)]
    return valid, neighbors, weights


def undistort(image, camera, noise=None):
    valid, neighbors, weights = interpolation(camera)
    weights = weights.copy()
    if noise is not None:
        for index, neighbor in enumerate(neighbors):
            weights[index] *= ~noise[neighbor]
        total = weights.sum(0)
        weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    output = np.full((480, 640), 255.0 if noise is None else 0.0)
    output[valid] = sum(weights[index] * image[neighbor] for index, neighbor in enumerate(neighbors))
    return output


def project_depth(raw, rgb):
    if raw.shape != (480, 640) or rgb.shape != (480, 640, 3):
        raise ValueError("NYUv2 raw frames must be 480x640")
    raw = raw.byteswap().astype(np.float64)
    color = np.stack([undistort(rgb[:, :, channel].astype(float), "rgb") for channel in range(3)], axis=2)
    color = np.clip(np.floor(color + 0.5), 0, 255).astype(np.uint8)
    noise = undistort((raw == raw.max()).astype(float) * 255, "depth") > 0
    disparity = undistort(raw, "depth", noise)
    disparity[(disparity < 600) | noise] = 2047
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.clip(351.3 / (1092.5 - disparity), 0, 10)
    yy, xx = np.mgrid[1:481, 1:641]
    fx, fy, cx, cy = DEPTH_INTRINSICS
    points = np.stack([(xx - cx) * depth / fx, (yy - cy) * depth / fy, depth], axis=2).reshape(-1, 3)
    points = points @ ROTATION.T + TRANSLATION
    fx, fy, cx, cy = RGB_INTRINSICS
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_x = np.floor(points[:, 0] * fx / points[:, 2] + cx + 0.5)
        projected_y = np.floor(points[:, 1] * fy / points[:, 2] + cy + 0.5)
    valid = (projected_x > 0) & (projected_x < 640) & (projected_y > 0) & (projected_y < 480)
    indices = (projected_y[valid].astype(int) - 1) * 640 + projected_x[valid].astype(int) - 1
    # Preserve the toolbox's 1-based projection and original camera Z buffer.
    output = np.full(480 * 640, np.inf)
    np.minimum.at(output, indices, depth.ravel()[valid])
    output[~np.isfinite(output)] = 0
    return output.reshape(480, 640).astype(np.float32), color
