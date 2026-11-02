"""Temporary RGB-only transport for H5 inputs; supervision never leaves the local dataset."""
from pathlib import Path

import cv2

from depth_project.io import read_json, write_json
from explore_a.data.nyu_h5 import read_h5


def export_rgb(path, source):
    """Resume only files claimed by this exporter; validate the source's size and timestamp."""
    path, source = Path(path), Path(source)
    identity = {"source": str(source), "bytes": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns}
    receipt = path.with_suffix(".rgb.json")
    if path.exists() and receipt.exists():
        prior = read_json(receipt)
        if prior["source"] == identity and prior["bytes"] == path.stat().st_size:
            return path.stat().st_size
        raise ValueError(f"Changed RGB export: {path}")
    if path.exists():
        raise ValueError(f"Unclaimed RGB export: {path}")
    rgb, _ = read_h5(source, depth=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    if temporary.is_symlink():
        raise ValueError(f"RGB export temporary file is a symlink: {temporary}")
    if not cv2.imwrite(str(temporary), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not export RGB: {path}")
    size = temporary.stat().st_size
    # Receipt first: a crash before rename leaves no published transport file.
    write_json(receipt, {"source": identity, "bytes": size})
    temporary.replace(path)
    return size
