"""Download the 13 public-GT ETH3D DSLR scenes used for 454-image depth testing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from urllib.parse import urljoin
from urllib.request import Request, urlopen

PAGE = "https://www.eth3d.net/datasets"
COUNTS = dict(zip(
    ("courtyard", "delivery_area", "electro", "facade", "kicker", "meadow",
     "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains"),
    (38, 44, 45, 76, 31, 15, 26, 14, 38, 31, 31, 23, 42)))
PROJECT = Path(__file__).resolve().parents[3]
MANIFEST = PROJECT / "storage/eth3d/high_res_454_plan.json"
OUTPUT = Path.home() / "Documents/Resources/Datasets/ETH3D/high_res_multiview_454"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href", "")
        if tag == "a" and href.endswith(".7z"):
            self.links[PurePosixPath(href).name] = urljoin(PAGE, href)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".part")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def plan(path, output, workers):
    if path.exists():
        raise ValueError("Plan exists; reuse it, or specify a new --manifest")
    with urlopen(PAGE, timeout=45) as response:
        parser = Links()
        parser.feed(response.read().decode())
    entries = []
    for scene, count in COUNTS.items():
        for kind, suffix in (("rgb_calibration", "dslr_jpg"), ("depth", "dslr_depth")):
            name = f"{scene}_{suffix}.7z"
            entries.append(dict(name=name, url=parser.links[name], scene=scene,
                                images=count, kind=kind))

    def metadata(entry):
        with urlopen(Request(entry["url"], method="HEAD"), timeout=45) as response:
            size = int(response.headers["Content-Length"])
            if size <= 0:
                raise ValueError(f"Invalid archive size: {entry['name']}")
            return dict(entry, bytes=size, etag=response.headers.get("ETag"),
                        last_modified=response.headers.get("Last-Modified"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(metadata, entries))
    result = dict(schema=1, source=PAGE, official_split="high-res multi-view training",
                  purpose="454-image depth evaluation, not continuous video evaluation",
                  image_variant="original distorted JPEG; matches official rendered depth",
                  scenes=COUNTS, images=sum(COUNTS.values()), output=str(output.resolve()),
                  files=entries, total_bytes=sum(e["bytes"] for e in entries),
                  official_checksums=None,
                  checked_at=datetime.now(timezone.utc).isoformat())
    save(path, result)
    status(result)


def load(path):
    result = json.loads(path.read_text())
    expected = {f"{scene}_{suffix}.7z" for scene in COUNTS
                for suffix in ("dslr_jpg", "dslr_depth")}
    if result["scenes"] != COUNTS or {e["name"] for e in result["files"]} != expected:
        raise ValueError("Unexpected ETH3D subset in plan")
    if len(result["files"]) != len(expected):
        raise ValueError("Duplicate archive entries")
    for entry in result["files"]:
        if entry["url"] != f"https://www.eth3d.net/data/{entry['name']}":
            raise ValueError("Unexpected archive URL")
    return result


def complete(path, entry):
    return (path.is_file() and path.stat().st_size == entry["bytes"]
            and not path.with_name(path.name + ".aria2").exists())


def status(manifest):
    archive_root = Path(manifest["output"]) / "archives"
    done = sum(complete(archive_root / e["name"], e) for e in manifest["files"])
    print(f"13 scenes / 454 images / {len(manifest['files'])} archives; complete {done}")
    for kind in ("rgb_calibration", "depth"):
        size = sum(e["bytes"] for e in manifest["files"] if e["kind"] == kind)
        print(f"{kind}: {size:,} bytes = {size/1e9:.6f} GB = {size/2**30:.6f} GiB")
    size = manifest["total_bytes"]
    print(f"Total: {size:,} bytes = {size/1e9:.6f} GB = {size/2**30:.6f} GiB")
    print("Output:", manifest["output"])


def download(manifest, path, jobs):
    root = Path(manifest["output"]) / "archives"
    queue = []
    for entry in manifest["files"]:
        if not complete(root / entry["name"], entry):
            queue.append(f"{entry['url']}\n  dir={root}\n  out={entry['name']}\n")
    if queue:
        root.mkdir(parents=True, exist_ok=True)
        request = path.with_suffix(".aria2.txt")
        request.write_text("".join(queue))
        subprocess.run([
            "aria2c", "--continue=true", "--auto-file-renaming=false",
            "--allow-overwrite=false", "--check-certificate=true", "--split=4",
            "--max-connection-per-server=4", f"--max-concurrent-downloads={jobs}",
            "--file-allocation=none", "--max-tries=10", "--retry-wait=5",
            "--connect-timeout=30", "--timeout=60", "--input-file=" + str(request),
        ], check=True)
    for entry in manifest["files"]:
        if not complete(root / entry["name"], entry):
            raise ValueError(f"Incomplete archive: {entry['name']}")
    print("All archive sizes match the frozen plan; no official digest was supplied.")


def archive_members(executable, path):
    listing = subprocess.run([executable, "l", "-slt", str(path)], check=True,
                             capture_output=True, text=True).stdout
    members = listing.split("----------", 1)[1]
    size = 0
    for block in members.strip().split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        name = fields.get("Path", "")
        candidate = PurePosixPath(name)
        if (not name or candidate.is_absolute() or ".." in candidate.parts
                or "\\" in name or ":" in name
                or any(k in fields for k in ("Symbolic Link", "Hard Link"))):
            raise ValueError(f"Unsafe archive member: {name}")
        size += int(fields.get("Size", 0))
    return size


def extract(manifest):
    root = Path(manifest["output"])
    executable = shutil.which("7z") or shutil.which("7zz")
    if not executable:
        raise FileNotFoundError("Install 7z/7zz first")
    unpacked = 0
    for entry in manifest["files"]:
        archive = root / "archives" / entry["name"]
        if not complete(archive, entry):
            raise ValueError(f"Download first: {archive}")
        unpacked += archive_members(executable, archive)
        marker = root / "extracted" / (entry["name"] + ".json")
        if marker.exists() and json.loads(marker.read_text()) == entry:
            continue
        # 7z verifies archive CRC while extracting. Existing identical members are
        # overwritten on recovery of a partial extraction, without deleting other data.
        subprocess.run([executable, "x", "-y", "-aoa", "-bsp0", str(archive),
                        "-o" + str(root / "data")], check=True)
        save(marker, entry)
    save(root / "extraction_sizes.json", dict(unpacked_bytes=unpacked,
         compressed_bytes=manifest["total_bytes"]))
    print(f"Archive member sizes after extraction: {unpacked:,} bytes")
    verify(manifest)


def verify(manifest):
    root = Path(manifest["output"]) / "data"
    all_images, all_depths = [], []
    for scene, expected in COUNTS.items():
        folder = root / scene
        files = list(folder.rglob("*"))
        images = [p for p in files if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
                  and "dslr_images" in p.parts and "ground_truth_depth" not in p.parts]
        depths = [p for p in files if p.is_file() and "ground_truth_depth" in p.parts]
        if len(images) != expected or len(depths) != expected:
            raise ValueError(f"{scene}: expected {expected} RGB/depth; found {len(images)}/{len(depths)}")
        if {p.name for p in images} != {p.name for p in depths}:
            raise ValueError(f"RGB/depth names differ: {scene}")
        if not list(folder.rglob("cameras.txt")) or not list(folder.rglob("images.txt")):
            raise ValueError(f"Missing calibration: {scene}")
        all_images.extend(images)
        all_depths.extend(depths)
    print(f"VERIFIED {len(COUNTS)} scenes; {len(all_images)} RGB / {len(all_depths)} depth files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "download", "status", "extract", "verify"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="plan only")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--metadata-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or not 1 <= args.metadata_workers <= 16:
        parser.error("jobs must be 1-8; metadata-workers must be 1-16")
    path = args.manifest.expanduser().resolve()
    if args.command == "plan":
        plan(path, args.output.expanduser(), args.metadata_workers)
    else:
        if args.output != OUTPUT:
            parser.error("--output applies to plan only; reuse its frozen output path")
        manifest = load(path)
        if args.command == "download":
            download(manifest, path, args.jobs)
        elif args.command == "extract":
            extract(manifest)
        elif args.command == "verify":
            verify(manifest)
        else:
            status(manifest)


if __name__ == "__main__":
    main()
