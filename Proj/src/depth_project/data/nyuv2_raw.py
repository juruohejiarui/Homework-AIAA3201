"""Export eight benchmark scenes from the full raw NYUv2 collection."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_project.data.nyu_projection import project_depth
from depth_project.data.protocol import NYU_SCENES
from depth_project.io import read_rgb, write_json
from depth_project.storage.mount import require_dataset_volume


def timestamp(path):
    return float(path.name[2:].split("-")[0])


def export_scene(source, output, count):
    images = sorted(source.glob("r-*.ppm"), key=timestamp)
    depths = sorted(source.glob("d-*.pgm"), key=timestamp)[:count]
    if len(depths) < count or not images:
        raise FileNotFoundError(f"Incomplete raw scene: {source}")
    times = np.array([timestamp(path) for path in images])
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(exist_ok=True)
    associations = []
    for index, depth_path in enumerate(depths):
        time = timestamp(depth_path)
        insertion = int(np.searchsorted(times, time))
        candidates = [candidate for candidate in [insertion - 1, insertion] if 0 <= candidate < len(images)]
        nearest = min(candidates, key=lambda candidate: (abs(times[candidate] - time), -candidate))
        image_path = images[nearest]
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth, rgb = project_depth(raw, read_rgb(image_path))
        encoded = np.clip(np.floor(depth * 6000 + 0.5), 0, 65535).astype(np.uint16)
        image_output = output / "rgb" / f"{index:06d}.png"
        depth_output = output / "depth" / f"{index:06d}.png"
        if not cv2.imwrite(str(image_output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(image_output)
        if not cv2.imwrite(str(depth_output), encoded):
            raise OSError(depth_output)
        associations.append({"index": index, "raw_rgb": str(image_path), "raw_depth": str(depth_path),
                             "rgb_timestamp": float(times[nearest]), "depth_timestamp": time})
    write_json(output / "association.json", {"frames": associations, "depth_factor": 6000,
                                             "unit": "m", "fill": "none", "matlab_parity": "pending"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=510)
    args = parser.parse_args()
    require_dataset_volume()
    for name in NYU_SCENES:
        scene, = [path for path in args.source.rglob(name) if path.is_dir()]
        destination = args.output / name
        if not (destination / "association.json").exists():
            export_scene(scene, destination, args.frames)
        print(name, flush=True)


if __name__ == "__main__":
    main()
