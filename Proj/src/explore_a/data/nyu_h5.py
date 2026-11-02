"""FastDepth H5 RGB-D reading and resumable extraction; depth is already RGB-aligned, in m."""
import argparse
from collections import Counter
from pathlib import Path, PurePosixPath
import tarfile

import h5py
import numpy as np


def read_h5(path, depth=True):
    """Return uint8 HWC RGB and optional FP32 H×W metric depth, without Raw reprojection."""
    with h5py.File(path, "r") as source:
        rgb = source["rgb"][:]
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"Invalid CHW uint8 RGB: {path}")
        rgb = np.ascontiguousarray(rgb.transpose(1, 2, 0))
        gt = np.asarray(source["depth"][:], dtype=np.float32) if depth else None
    if gt is not None and (gt.shape != rgb.shape[:2] or not np.isfinite(gt).all()
                           or (gt < 0).any()):
        raise ValueError(f"Invalid aligned metric depth: {path}")
    return rgb, gt


def extract(archives, output):
    """Stream members to atomic files. Existing files require matching size and source receipt."""
    from depth_project.io import read_json, write_json
    archives, output = Path(archives).expanduser(), Path(output).expanduser()
    files = sorted(archives.glob("train-*.tar")) + sorted(archives.glob("val-*.tar"))
    if not files:
        raise FileNotFoundError(f"No train/val tar archives: {archives}")
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for path in files:
        identity = {"source": str(path.resolve()), "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns}
        receipt = output / ".extraction" / (path.name + ".json")
        claimed = receipt.exists()
        if receipt.exists() and read_json(receipt)["archive"] != identity:
            raise ValueError(f"Archive changed after extraction: {path}")
        if receipt.exists() and read_json(receipt).get("blocked"):
            raise ValueError(f"Resolve the previously reported extraction conflict: {path}")
        # The claim is written before extraction, so interrupted archives can safely resume.
        write_json(receipt, {"archive": identity, "complete": False})
        count = 0
        with tarfile.open(path) as source:
            for member in source:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                        or member.issym() or member.islnk()):
                    raise ValueError(f"Unsafe tar member: {member.name}")
                if member.isdir():
                    continue
                split = "train" if path.name.startswith("train-") else "val"
                if (not member.isfile() or len(name.parts) != 3 or name.parts[0] != split
                        or name.suffix != ".h5" or not name.stem.isdigit()):
                    raise ValueError(f"Unexpected H5 member: {member.name}")
                target = output / str(name)
                if not target.resolve().is_relative_to(output.resolve()) or target.is_symlink():
                    raise ValueError(f"Extraction escapes output: {target}")
                if target.exists():
                    if not claimed or target.stat().st_size != member.size:
                        write_json(receipt, {"archive": identity, "blocked": True})
                        raise ValueError(f"Conflicting extracted file: {target}")
                else:
                    import shutil
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".part")
                    if temporary.is_symlink():
                        raise ValueError(f"Extraction temporary file is a symlink: {temporary}")
                    with source.extractfile(member) as stream, temporary.open("wb") as dest:
                        shutil.copyfileobj(stream, dest, length=1024 * 1024)
                    if temporary.stat().st_size != member.size:
                        raise OSError(f"Short extraction: {target}")
                    temporary.replace(target)
                count += 1
        counts[split] += count
        write_json(receipt, {"archive": identity, "complete": True, "files": count})
        print(f"{path.name}: {count} H5 files", flush=True)
    print(dict(counts), flush=True)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract",))
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.archives, args.output)


if __name__ == "__main__":
    main()
