"""Explicit copy, independent full verification, and optional source deletion."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from depth_project.io import read_json, write_json
from depth_project.storage.inventory import inventory
from depth_project.storage.mount import VOLUME, require_dataset_volume


def migrate(source, target, receipt, delete_source=False):
    require_dataset_volume()
    source = source.resolve(strict=True)
    target = target.resolve()
    if not target.is_relative_to(VOLUME) or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Source and target must be separate directories; target must be on the dataset volume")
    if receipt.exists():
        state = read_json(receipt)
        if state["source"] != str(source) or state["target"] != str(target):
            raise ValueError("Receipt belongs to a different migration")
    else:
        entries = inventory(source)
        if target.exists():
            existing = inventory(target)
            if any(entries.get(name) != value for name, value in existing.items()):
                raise FileExistsError("Existing destination contains conflicting data")
        state = {"source": str(source), "target": str(target), "entries": entries, "phase": "copying"}
        write_json(receipt, state)
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-rl", "--checksum", "--partial", "--no-perms", "--no-owner",
                    "--no-group", str(source) + "/", str(target) + "/"], check=True)
    subprocess.run(["sync"], check=True)
    subprocess.run([sys.executable, "-m", "depth_project.storage.verify_copy",
                    str(source), str(target), str(receipt)], check=True)
    state["phase"] = "verified"
    write_json(receipt, state)
    if delete_source:
        shutil.rmtree(source)
        state["phase"] = "source_removed"
        write_json(receipt, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args()
    migrate(args.source, args.target, args.receipt, args.delete_source)


if __name__ == "__main__":
    main()
