"""Download only official TUM validation RGB-D archives, with withheld camera trajectories."""
import argparse
from pathlib import Path
import re

from depth_project.io import read_json
from depth_project.paths import DATASETS, PROJECT
from explore_a.download import tum

DEFAULT_MANIFEST = PROJECT / "storage/tum/validation_plan.json"
DEFAULT_OUTPUT = Path.home() / "Documents/Resources/Datasets/TUM-RGBD/official_validation"


def archive_name(value):
    name = value if value.startswith("rgbd_dataset_") else "rgbd_dataset_" + value
    name = name if name.endswith(".tgz") else name + ".tgz"
    if not re.fullmatch(r"rgbd_dataset_freiburg[123]_[a-z0-9_]+_validation\.tgz", name):
        raise argparse.ArgumentTypeError("Expected freiburgN_NAME_validation[.tgz]")
    return name


def validation_plan(path):
    manifest = read_json(path)
    entries = manifest["files"]
    if not entries or manifest.get("subset") != "validation":
        raise ValueError("Use an official validation-only plan from this command")
    if any(e.get("category") != "validation" or archive_name(e["name"]) != e["name"]
           for e in entries):
        raise ValueError("Plan contains a non-validation archive")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "download", "extract", "status"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", nargs="+", type=archive_name,
                        help="plan only; omit to select all official validation sequences")
    parser.add_argument("--metadata-workers", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or not 1 <= args.metadata_workers <= 32:
        parser.error("jobs: 1-8; metadata-workers: 1-32")
    args.manifest = args.manifest.expanduser().resolve()
    if args.command != "plan" and (args.sequences or args.output != DEFAULT_OUTPUT):
        parser.error("--sequences and --output apply only to plan; reuse its frozen paths")
    if args.command == "plan":
        tum.plan(args.manifest, args.output, workers=args.metadata_workers,
                 subset="validation", names=args.sequences)
        print("Hidden GT means trajectory, not sensor depth; no submission is performed.")
        return
    manifest = validation_plan(args.manifest)
    if args.command == "download":
        tum.download(args.manifest, args.jobs)
    elif args.command == "extract":
        tum.extract(args.manifest)
    else:
        done = sum(tum.complete(tum.archive_target(manifest, e), e) for e in manifest["files"])
        print(f"Archives complete by size: {done}/{len(manifest['files'])}; "
              f"planned {manifest['total_bytes'] / 1e9:.3f} GB")
        print("Output:", manifest["output"])


if __name__ == "__main__":
    main()
