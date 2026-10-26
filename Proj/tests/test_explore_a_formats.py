"""Known sensor units and archive boundaries, using tiny locally generated inputs."""
import io
import struct
import tarfile
import zlib

import cv2
import numpy as np
import pytest

from depth_project.io import write_json
from explore_a.data.reading import read_frame
from explore_a.data.sens import Sensor
from explore_a.data.tum import interpolate_pose
from explore_a.download.tum import complete, extract


def test_scannet_v4_units_and_rgb_zbuffer(tmp_path):
    identity = np.eye(4, dtype=np.float32).tobytes()
    rgb = np.full((4, 4, 3), [10, 20, 30], np.uint8)
    color = cv2.imencode(".png", rgb)[1].tobytes()
    depth = zlib.compress(np.array([[1000, 2000], [0, 4000]], "<u2").tobytes())
    path = tmp_path / "scene0000_00.sens"
    with path.open("wb") as stream:
        stream.write(struct.pack("<IQ", 4, 4) + b"test")
        stream.write(identity * 4)
        stream.write(struct.pack("<2i4IfQ", 1, 1, 4, 4, 2, 2, 1000., 2))
        for time in (0, 33333):
            stream.write(identity)
            stream.write(struct.pack("<4Q", time, time, len(color), len(depth)))
            stream.write(color + depth)
    source = Sensor(str(path))
    image, meters = source.read(1)
    np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
    np.testing.assert_allclose(meters[:2, :2], [[1, 2], [0, 4]])
    assert meters[2:].sum() == 0
    assert source.frames[1][3] == .033333


def test_tum_depth_factor_and_camera_to_world_interpolation(tmp_path):
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    cv2.imwrite(str(rgb), np.zeros((2, 2, 3), np.uint8))
    cv2.imwrite(str(depth), np.array([[0, 5000], [10000, 40000]], np.uint16))
    _, target = read_frame({"depth_encoding": "tum_png", "current_rgb": str(rgb), "gt": str(depth)})
    np.testing.assert_allclose(target, [[0, 1], [2, 8]])
    poses = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [.01, 1, 0, 0, 0, 0, 0, 1]])
    pose = interpolate_pose(poses, .005, .02)
    np.testing.assert_allclose(pose[:3, 3], [.5, 0, 0])
    assert interpolate_pose(poses, 1., .02) is None


def test_archive_control_file_and_path_escape(tmp_path):
    archive = tmp_path / "archives/rgbd_dataset_freiburg1_xyz.tgz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    entry = {"name": archive.name, "bytes": archive.stat().st_size}
    assert complete(archive, entry)
    control = archive.with_name(archive.name + ".aria2")
    control.touch()
    assert not complete(archive, entry)
    control.unlink()
    plan = tmp_path / "plan.json"
    write_json(plan, {"output": str(tmp_path), "files": [entry]})
    with pytest.raises(ValueError, match="Unsafe/unexpected"):
        extract(plan)
    assert not (tmp_path / "escaped.txt").exists()
