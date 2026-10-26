"""User-run full NYUv2 Raw plan, MD5-verified download, verification and extraction."""
import argparse
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

from depth_project.io import read_json, write_json
from depth_project.paths import DATASETS, PROJECT
from explore_a.download.nyu_integrity import (
    complete, record_verified, target_path, verified_receipt, verify_file,
)
from explore_a.download.nyu_links import PAGE, discover

DEFAULT_MANIFEST = PROJECT / "storage/nyuv2/plan.json"


def plan(path, output, workers):
    if path.exists():
        raise ValueError("Plan exists; reuse it or select a new --manifest path")
    entries = discover(workers)
    metadata = path.with_suffix(".metadata")
    metadata.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        text = entry.pop("official_md5_text")
        (metadata / (entry["name"].removesuffix(".zip") + ".md5")).write_text(text)
    (metadata / "official_md5sums.txt").write_text("".join(
        f"{entry['md5']}  {entry['name']}\n" for entry in entries))
    result = {"schema": 1, "source": PAGE, "output": str(output.expanduser().resolve()),
              "files": entries, "total_bytes": sum(entry["bytes"] for entry in entries)}
    write_json(path, result)
    print(f"Full NYUv2 Raw: {len(entries)} ZIPs; {result['total_bytes'] / 1e9:.3f} GB; "
          f"official MD5 coverage {len(entries)}/{len(entries)}. No ZIP payload downloaded.")


def receipts_for(path):
    receipt_path = path.with_suffix(".verified.json")
    return receipt_path, read_json(receipt_path) if receipt_path.exists() else {}


def download(path, jobs):
    data = read_json(path)
    receipt_path, receipts = receipts_for(path)
    queued, lines = [], []
    for entry in data["files"]:
        target = target_path(data, entry)
        if complete(target, entry):
            # Unlike the legacy size-only skip, preexisting complete ZIPs need an MD5 receipt.
            verify_file(target, entry, receipts)
            write_json(receipt_path, receipts)
            print("Verified/reused", entry["name"], flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        queued.append(entry)
        lines.append(f"{entry['url']}\n  dir={target.parent}\n  out={target.name}\n"
                     f"  checksum=md5={entry['md5']}\n")
    if queued:
        request = path.with_suffix(".aria2.txt")
        request.write_text("".join(lines))
        subprocess.run(["aria2c", "--continue=true", "--check-integrity=true",
                        "--check-certificate=true", "--auto-file-renaming=false",
                        "--allow-overwrite=false", "--file-allocation=none", "--split=4",
                        "--max-connection-per-server=4", f"--max-concurrent-downloads={jobs}",
                        "--max-tries=10", "--retry-wait=5", "--connect-timeout=30",
                        "--timeout=60", "--input-file=" + str(request)], check=True)
        for entry in queued:
            target = target_path(data, entry)
            if not complete(target, entry):
                raise ValueError(f"aria2 did not finish: {target}")
            record_verified(target, entry, receipts, "aria2-official-md5")
            write_json(receipt_path, receipts)
    print("All selected Raw ZIPs passed official MD5 verification. Archives retained.")


def verify(path):
    data = read_json(path)
    receipt_path, receipts = receipts_for(path)
    for entry in data["files"]:
        verify_file(target_path(data, entry), entry, receipts, recheck=True)
        write_json(receipt_path, receipts)
        print("Official MD5 passed:", entry["name"], flush=True)


def extract_zip(archive, root):
    """Stream ZIP members to atomic targets; never extract links or escape the Raw root."""
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            member = Path(entry.filename)
            target = root / member
            if member.is_absolute() or ".." in member.parts \
                    or not target.resolve().is_relative_to(root.resolve()) \
                    or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Unsafe Raw ZIP path: {entry.filename}")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == entry.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extract-part")
            with source.open(entry) as reader, temporary.open("wb") as writer:
                while block := reader.read(1024 * 1024):
                    writer.write(block)
            temporary.replace(target)


def extract(path):
    data = read_json(path)
    receipt_path, receipts = receipts_for(path)
    root = Path(data["output"]) / "raw/scenes"
    for entry in data["files"]:
        archive = target_path(data, entry)
        if not verified_receipt(archive, entry, receipts):
            verify_file(archive, entry, receipts)
            write_json(receipt_path, receipts)
        marker = root / ".extracted" / (entry["name"] + ".json")
        expected = {"name": entry["name"], "official_md5": entry["md5"]}
        if marker.exists() and read_json(marker) == expected:
            continue
        extract_zip(archive, root)
        write_json(marker, expected)
        print("Extracted", entry["name"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "download", "verify", "extract"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DATASETS / "nyu_depth_v2")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--metadata-workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or not 1 <= args.metadata_workers <= 32:
        parser.error("jobs: 1-8; metadata workers: 1-32")
    args.manifest = args.manifest.expanduser().resolve()
    if args.command != "plan" and args.output != DATASETS / "nyu_depth_v2":
        parser.error("--output is a plan option; subsequent commands read the saved plan")
    if args.command == "plan":
        plan(args.manifest, args.output, args.metadata_workers)
    elif args.command == "download":
        download(args.manifest, args.jobs)
    elif args.command == "verify":
        verify(args.manifest)
    else:
        extract(args.manifest)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
