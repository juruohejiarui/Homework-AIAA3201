"""Sintel DPT decoding; retain float32 meters instead of quantizing to uint16."""
import numpy as np

from depth_project.data.association import pair_frame_ids


def read_dpt(path):
    with path.open("rb") as stream:
        tag = np.fromfile(stream, "<f4", count=1)[0]
        width, height = np.fromfile(stream, "<i4", count=2)
        values = np.fromfile(stream, "<f4")
    if tag != 202021.25:
        raise ValueError(f"Invalid Sintel depth header: {path}")
    return values.reshape(int(height), int(width))


def prepare_sintel(source, output):
    clean, = [path for path in source.rglob("clean") if path.is_dir()]
    depth_root, = [path for path in source.rglob("depth") if any(path.glob("*/*.dpt"))]
    sequences = []
    for scene in sorted(path for path in clean.iterdir() if path.is_dir()):
        pairs = pair_frame_ids(scene.glob("*.png"), (depth_root / scene.name).glob("*.dpt"))
        frames = []
        for index, (image, depth) in enumerate(pairs[:100]):
            target = output / "gt_m" / scene.name / (depth.stem + ".npy")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                np.save(target, read_dpt(depth))
            frames.append({"image": str(image.resolve()), "gt": str(target.resolve()),
                           "source_index": index, "raw_depth": str(depth)})
        sequences.append({"name": scene.name, "frames": frames})
    return sequences
