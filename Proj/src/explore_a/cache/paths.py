"""Canonical paths independent of host dataset roots; no RGB or supervision is cached."""
from pathlib import Path

KINDS = ("frames", "flow/raft", "flow/farneback", "teacher/raft", "teacher/farneback")


def component(value):
    if not value or value in (".", "..") or any(c in value for c in ("/", "\\", ":", "\0")):
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def frame_key(row, previous=False):
    return ":".join((row["dataset"], row["sequence"], row["camera"],
                     row["previous_id" if previous else "current_id"]))


def relative(kind, key):
    if kind not in KINDS:
        raise ValueError(f"Unknown cache kind: {kind}")
    dataset, sequence, camera, frame = [component(part) for part in key.split(":")]
    if dataset == "kitti":
        if camera not in ("image_02", "image_03"):
            raise ValueError(camera)
        return Path(kind) / dataset / sequence[:10] / sequence / camera / "data" / (frame + ".npz")
    if dataset == "tum" and camera == "rgb":
        return Path(kind) / dataset / "sequences" / sequence / "rgb" / (frame + ".npz")
    if dataset == "nyuv2" and camera == "h5_rgb":
        return Path(kind) / dataset / "h5" / sequence / "rgb" / (frame + ".npz")
    raise ValueError("The initial distributed cache recipe supports KITTI and TUM only")


def job_id(row):
    if row["dataset"] == "nyuv2":
        if row["camera"] != "h5_rgb":
            raise ValueError("Distributed NYUv2 caching currently requires the H5 adapter")
        return "nyuv2_h5__" + component(row["sequence"])
    return component(row["dataset"]) + "__" + component(row["sequence"])


def artifact(root, kind, row, previous=False):
    if kind != "frames":
        from explore_a.frontend import shared_root
        root = shared_root(root)
    return Path(root) / relative(kind, frame_key(row, previous))


def expected(rows, flows=("raft", "farneback"), teacher=True, frames=True):
    names = set()
    for row in rows:
        if frames:
            for previous in (True, False):
                names.add(str(relative("frames", frame_key(row, previous))))
        for flow in flows:
            names.add(str(relative(f"flow/{flow}", row["id"])))
            if teacher:
                names.add(str(relative(f"teacher/{flow}", row["id"])))
    return sorted(names)
