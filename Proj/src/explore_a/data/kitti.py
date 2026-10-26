"""KITTI pairs use consecutive Raw RGB IDs, never consecutive annotated filenames."""
from collections import Counter
import zipfile

import numpy as np

from explore_a.data.history import drive_id
from explore_a.data.splits import development_groups


def extract_training(root):
    """Only extract missing train PNGs; existing val files and RGB remain untouched."""
    archive = root / "archives/data_depth_annotated.zip"
    with zipfile.ZipFile(archive) as source:
        entries = [i for i in source.infolist() if i.filename.startswith("train/")
                   and i.filename.endswith(".png")]
        for info in entries:
            destination = root / "depth" / info.filename
            if not destination.resolve().is_relative_to((root / "depth").resolve()):
                raise ValueError(f"Unsafe ZIP path: {info.filename}")
            if destination.exists():
                if destination.stat().st_size != info.file_size:
                    raise ValueError(f"Existing annotation has unexpected size: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".part")
            temporary.write_bytes(source.read(info))
            temporary.replace(destination)


def camera_matrix(root, sequence, camera):
    calibration = root / "raw" / sequence[:10] / "calib_cam_to_cam.txt"
    if not calibration.exists():
        return None
    key = f"P_rect_{camera[-2:]}:"
    for line in calibration.read_text().splitlines():
        if line.startswith(key):
            return np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)[:, :3].tolist()
    return None


def prepare_kitti(root, settings, history, aliases):
    candidates = sorted(p for p in (root / "depth/train").glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError("KITTI depth/train is absent; run prepare --extract-kitti")
    group = lambda name: aliases.get(name, name)
    protected_groups = {group(name) for name in history}
    eligible = [p for p in candidates if group(drive_id(p.name)) not in protected_groups]
    dev = development_groups([group(drive_id(p.name)) for p in eligible],
                             settings["seed"], settings["dev_fraction"])
    drives = [(p, "dev" if group(drive_id(p.name)) in dev else "train") for p in eligible]
    for name in history:
        validation = root / "depth/val" / name
        drives.append((validation if validation.exists() else root / "depth/train" / name, "test"))
    rows, excluded = [], Counter()
    for scene, split in drives:
        sequence = drive_id(scene.name)
        for camera in ("image_02", "image_03"):
            image_root = root / "raw" / sequence[:10] / sequence / camera / "data"
            depth_root = scene / "proj_depth/groundtruth" / camera
            for target in sorted(depth_root.glob("*.png")):
                frame = int(target.stem)
                previous = image_root / f"{frame - 1:010d}.png"
                current = image_root / target.name
                if frame == 0 or not previous.is_file() or not current.is_file():
                    excluded["missing_consecutive_rgb"] += 1
                    continue
                rows.append({"id": f"kitti:{sequence}:{camera}:{frame:010d}",
                             "dataset": "kitti", "sequence": sequence, "camera": camera,
                             "group": group(sequence), "split": split,
                             "previous_id": f"{frame - 1:010d}", "current_id": target.stem,
                             "previous_rgb": str(previous.resolve()),
                             "current_rgb": str(current.resolve()),
                             "gt": str(target.resolve()), "depth_encoding": "kitti_png",
                             "previous_gt": str((depth_root / previous.name).resolve())
                             if (depth_root / previous.name).is_file() else None,
                             "K": camera_matrix(root, sequence, camera), "pose": None})
    excluded["historical_or_same_group_drives"] = len(candidates) - len(eligible)
    return rows, dict(excluded)
