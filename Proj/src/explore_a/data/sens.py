"""Read ScanNet .sens v4 on demand; keep RGB/depth/pose from the same sensor frame."""
from functools import lru_cache
from pathlib import Path
import struct
import zlib

import cv2
import numpy as np


def unpack(stream, fmt):
    size = struct.calcsize("<" + fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Incomplete ScanNet .sens file")
    return struct.unpack("<" + fmt, data)


class Sensor:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as stream:
            if unpack(stream, "I")[0] != 4:
                raise ValueError("Only ScanNet .sens version 4 is supported")
            stream.read(unpack(stream, "Q")[0])
            matrices = [np.array(unpack(stream, "16f")).reshape(4, 4) for _ in range(4)]
            self.color_k, self.color_extrinsic, self.depth_k, self.depth_extrinsic = matrices
            self.color_type, self.depth_type = unpack(stream, "2i")
            self.cw, self.ch, self.dw, self.dh = unpack(stream, "4I")
            self.depth_shift = unpack(stream, "f")[0]
            count = unpack(stream, "Q")[0]
            self.frames = []
            for _ in range(count):
                pose = np.array(unpack(stream, "16f"), np.float32).reshape(4, 4)
                color_time, depth_time, color_size, depth_size = unpack(stream, "4Q")
                offset = stream.tell()
                self.frames.append((offset, color_size, depth_size, color_time / 1e6,
                                    depth_time / 1e6, pose))
                stream.seek(color_size + depth_size, 1)
            if stream.tell() > Path(path).stat().st_size:
                raise ValueError(f"Truncated .sens payload: {path}")
        if self.color_type not in (1, 2) or self.depth_type not in (0, 1):
            raise ValueError("Supported .sens codecs: PNG/JPEG RGB and raw/zlib ushort depth")

    def read(self, index):
        """Return uint8 RGB and depth Z in meters, projected to the RGB pixel grid."""
        offset, color_size, depth_size, *_ = self.frames[index]
        with open(self.path, "rb") as stream:
            stream.seek(offset)
            color = stream.read(color_size)
            packed_depth = stream.read(depth_size)
        image = cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Invalid .sens color frame {index}: {self.path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        decoded = zlib.decompress(packed_depth) if self.depth_type == 1 else packed_depth
        depth = np.frombuffer(decoded, "<u2").reshape(self.dh, self.dw).astype(np.float32)
        depth /= self.depth_shift
        yy, xx = np.indices(depth.shape)
        rays = np.linalg.inv(self.depth_k[:3, :3]) @ np.stack(
            [xx.ravel(), yy.ravel(), np.ones(depth.size)])
        points = rays * depth.ravel()
        # Extrinsics map camera coordinates into the shared sensor frame.
        relative = np.linalg.inv(self.color_extrinsic) @ self.depth_extrinsic
        points = relative[:3, :3] @ points + relative[:3, 3:4]
        projected = self.color_k[:3, :3] @ points
        z = points[2]
        valid = (depth.ravel() > 0) & np.isfinite(z) & (z > 0)
        uv = np.zeros((2, depth.size), dtype=np.int64)
        uv[:, valid] = np.rint(projected[:2, valid] / projected[2:3, valid]).astype(np.int64)
        x, y = uv
        valid &= (x >= 0) & (x < self.cw) & (y >= 0) & (y < self.ch)
        aligned = np.full(self.ch * self.cw, np.inf, np.float32)
        # Z-buffer keeps the visible surface when several depth pixels collide.
        np.minimum.at(aligned, y[valid] * self.cw + x[valid], z[valid])
        aligned[~np.isfinite(aligned)] = 0
        return rgb, aligned.reshape(self.ch, self.cw)


@lru_cache(maxsize=2)
def sensor(path):
    return Sensor(path)
