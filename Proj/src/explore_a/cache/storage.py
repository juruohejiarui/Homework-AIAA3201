"""Atomic, maximum-level lossless NPZ and cheap transfer inventories; no full-file hashes."""
from pathlib import Path
import shutil
import zipfile

import numpy as np


def write_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for key, value in values.items():
                value = np.asarray(value)
                if value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"Expected finite FP32 model output: {path}/{key}")
                with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, value, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"Invalid cached FP32 output: {path}")
    return values


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe inventory path: {name}")
    return path


def inventory(root, names):
    root = Path(root)
    result = {}
    for name in names:
        path = root / safe_relative(name)
        if (path.is_symlink() or not path.is_file() or
                not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f"Missing/nonregular artifact: {path}")
        result[name] = path.stat().st_size
    return result


def check_inventory(root, sizes):
    actual = inventory(root, sizes)
    if actual != sizes:
        raise ValueError("Incomplete or changed transfer: inventory sizes differ")


def require_space(root, minimum_gib, extra_bytes=0):
    path = Path(root)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < minimum_gib * 2**30 + extra_bytes:
        raise OSError(f"Space pause: {root} must retain {minimum_gib} GiB free")
