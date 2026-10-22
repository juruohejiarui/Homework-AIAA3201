"""Independent-process verification used by the migration command."""
import argparse
from pathlib import Path

from depth_project.io import read_json
from depth_project.storage.inventory import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    expected = read_json(args.manifest)["entries"]
    if inventory(args.target) != expected or inventory(args.source) != expected:
        raise RuntimeError("Copy differs from source or source changed; source must be retained")
    print("Independent full-copy verification passed")


if __name__ == "__main__":
    main()
