"""TUM plan/download/extract are separate actions, always explicitly started by the user."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tarfile

from depth_project.io import read_json, write_json
from depth_project.paths import DATASETS, PROJECT
from explore_a.download.tum_links import PAGE, crawl, get_size

DEFAULT_MANIFEST = PROJECT / "storage/tum/plan.json"


def plan(path, output, include_all=False, workers=8):
    if path.exists():
        raise ValueError("Plan exists; reuse it or select a different --manifest")
    entries = crawl()
    selected = entries if include_all else [e for e in entries if e["category"] == "ordinary"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        selected = list(pool.map(get_size, selected))
    result = {"schema": 1, "source": PAGE, "all_discovered": entries,
              "output": str(output.expanduser().resolve()), "files": selected,
              "total_bytes": sum(e["bytes"] for e in selected), "official_checksums": None}
    write_json(path, result)
    print(f"Discovered {len(entries)} archives; selected {len(selected)}; "
          f"{result['total_bytes'] / 1e9:.3f} GB. No archive payload downloaded.")


def archive_target(manifest, entry):
    name = entry["name"]
    if Path(name).name != name or not name.endswith(".tgz"):
        raise ValueError(f"Unsafe archive name: {name}")
    return Path(manifest["output"]) / "archives" / name


def complete(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] \
        and not path.with_name(path.name + ".aria2").exists()


def download(path, jobs):
    manifest = read_json(path)
    queue = []
    for entry in manifest["files"]:
        target = archive_target(manifest, entry)
        if complete(target, entry):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        queue.append(f"{entry['url']}\n  dir={target.parent}\n  out={target.name}\n")
    if queue:
        request = path.with_suffix(".aria2.txt")
        request.write_text("".join(queue))
        subprocess.run(["aria2c", "--continue=true", "--auto-file-renaming=false",
                        "--allow-overwrite=false", "--check-certificate=true", "--split=4",
                        "--max-connection-per-server=4", f"--max-concurrent-downloads={jobs}",
                        "--file-allocation=none", "--max-tries=10", "--retry-wait=5",
                        "--connect-timeout=30", "--timeout=60", "--input-file=" + str(request)],
                       check=True)
    for entry in manifest["files"]:
        if not complete(archive_target(manifest, entry), entry):
            raise ValueError(f"Incomplete archive: {entry['name']}")
    print("Download sizes match. Official digests were not supplied; no official hash claim.")


def extract(path):
    manifest = read_json(path)
    root = Path(manifest["output"]) / "sequences"
    root.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        archive = archive_target(manifest, entry)
        if not complete(archive, entry):
            raise ValueError(f"Finish downloading first: {archive}")
        destination = root / entry["name"].removesuffix(".tgz")
        marker = destination / ".explore_a_extracted.json"
        if marker.exists():
            continue
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                target = (root / member.name).resolve()
                safe_type = member.isfile() or member.isdir()
                if not target.is_relative_to(destination.resolve()) or not safe_type:
                    raise ValueError(f"Unsafe/unexpected tar member: {member.name}")
            source.extractall(root, filter="data")
        write_json(marker, {"archive": entry["name"], "bytes": entry["bytes"]})
        print("Extracted", entry["name"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "download", "extract"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DATASETS / "TUM-RGBD")
    parser.add_argument("--all", action="store_true",
                        help="include validation and calibration archives")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--metadata-workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or not 1 <= args.metadata_workers <= 32:
        parser.error("jobs: 1-8; metadata workers: 1-32")
    args.manifest = args.manifest.expanduser().resolve()
    if args.command != "plan" and (args.all or args.output != DATASETS / "TUM-RGBD"):
        parser.error("--all and --output apply only to plan")
    if args.command == "plan":
        plan(args.manifest, args.output, args.all, args.metadata_workers)
    elif args.command == "download":
        download(args.manifest, args.jobs)
    else:
        extract(args.manifest)


if __name__ == "__main__":
    main()
