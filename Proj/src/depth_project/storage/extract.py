"""Extract verified ZIP files with CRC checking during decompression."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile
import zlib

from depth_project.io import write_json
from depth_project.storage.mount import require_dataset_volume


def existing_crc(path):
    checksum = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum


def extract_archive(archive, output):
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            target = output.joinpath(*relative.parts)
            if any(path.is_symlink() for path in [target, *target.parents]):
                raise ValueError(f"Refusing extraction through symlink: {target}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size or existing_crc(target) != member.CRC:
                    raise FileExistsError(f"Existing data conflicts with ZIP: {target}")
                continue
            partial = target.with_name(target.name + ".extracting")
            with source.open(member) as compressed, partial.open("wb") as destination:
                shutil.copyfileobj(compressed, destination, 8 * 1024 ** 2)
            partial.replace(target)
    write_json(output / ".extraction" / (archive.name + ".json"), {"archive": str(archive), "complete": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_dataset_volume()
    for archive in args.archives:
        extract_archive(archive, args.output)
        print(archive.name)


if __name__ == "__main__":
    main()
