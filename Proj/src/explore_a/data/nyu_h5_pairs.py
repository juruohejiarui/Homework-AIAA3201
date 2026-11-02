"""Pair adjacent retained H5 frames, with room-level isolation and unknown elapsed time."""
from collections import Counter
from pathlib import Path

from explore_a.data.nyu import room_id, test_rooms
from explore_a.data.splits import development_groups


def prepare_h5(root, settings, aliases):
    folder = Path(settings["nyu_h5_root"])
    scenes = sorted(p for p in (folder / "train").iterdir() if p.is_dir())
    if not scenes:
        raise FileNotFoundError(f"No NYUv2 H5 training sequences: {folder}")
    group = lambda name: aliases.get(name, room_id(name))
    reserved = test_rooms(root)
    protected = reserved | {group(s.name) for s in scenes if room_id(s.name) in reserved}
    eligible = [s for s in scenes if room_id(s.name) not in reserved
                and group(s.name) not in protected]
    dev = development_groups([group(s.name) for s in eligible], settings["seed"],
                             settings["dev_fraction"])
    rows, counts = [], Counter()
    for scene in scenes:
        split = "test" if room_id(scene.name) in reserved or group(scene.name) in protected \
            else "dev" if group(scene.name) in dev else "train"
        files = sorted(scene.glob("*.h5"), key=lambda p: int(p.stem))
        counts["rgb_frames_discovered"] += len(files)
        for previous, current in zip(files, files[1:]):
            gap = int(current.stem) - int(previous.stem)
            if gap <= 0:
                raise ValueError(f"Duplicate/nonincreasing frame IDs: {scene}")
            rows.append({"id": f"nyuv2:{scene.name}:h5_rgb:{current.stem}",
                         "dataset": "nyuv2", "sequence": scene.name, "camera": "h5_rgb",
                         "group": group(scene.name), "split": split,
                         "previous_id": previous.stem, "current_id": current.stem,
                         "previous_rgb": str(previous.resolve()),
                         "current_rgb": str(current.resolve()), "gt": str(current.resolve()),
                         "previous_gt": str(previous.resolve()), "depth_encoding": "nyu_h5",
                         "frame_index_gap": gap, "time_gap_s": None,
                         "pairing": "adjacent_retained_h5", "K": None, "pose": None})
    counts.update({f"{split}_pairs": sum(r["split"] == split for r in rows)
                   for split in ("train", "dev", "test")})
    return rows, {**counts, "source": "fastdepth_h5", "time_gap_known": False,
                  "selection": "adjacent retained frames within each training sequence",
                  "official_validation_unpaired": len(list((folder / "val").rglob("*.h5"))),
                  "temporal_test_available": counts["test_pairs"] > 0}
