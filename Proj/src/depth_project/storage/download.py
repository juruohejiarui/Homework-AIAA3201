"""Download from the frozen NYUv2 catalog, or the verified Bonn HF mirror."""
import argparse
import os
import shutil
import subprocess

from depth_project.io import file_hash, read_json, write_json
from depth_project.paths import DATASETS, PROJECT
from depth_project.storage.mount import require_dataset_volume

BONN_REPO = "geyongtao/bonn"
BONN_REVISION = "383b5f8d3fbb14412a2593d2be45a09268960cea"
BONN_SHA256 = "d2afdc286ecdb02a3e57cb98e3908185b1133a3f79f181970064eb177221a84b"


def download_nyuv2():
    manifest = read_json(PROJECT / "storage/nyuv2_full_download_manifest.json")
    destination = DATASETS / "nyu_depth_v2/archives/raw"
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c"):
        subprocess.run(["aria2c", "-c", "-j", "3", "-x", "4", "-s", "4",
                        "--file-allocation=none", "--auto-file-renaming=false",
                        "-i", str(PROJECT / "storage/nyuv2_raw_aria2.txt")], check=True)
        return
    for archive in manifest["archives"]:
        target = destination / archive["name"]
        receipt = target.with_suffix(".verified.json")
        if receipt.exists() and target.exists() and target.stat().st_size == archive["bytes"]:
            continue
        partial = target.with_name(target.name + ".part")
        if target.exists():
            partial = target
        subprocess.run(["curl", "--fail", "--location", "--continue-at", "-",
                        "--retry", "20", "--retry-all-errors", "--retry-delay", "10",
                        "--output", str(partial), archive["url"]], check=True)
        if file_hash(partial, "md5") != archive["md5"]:
            raise RuntimeError(f"Official checksum mismatch: {partial}")
        if partial != target:
            partial.replace(target)
        write_json(receipt, archive)


def download_bonn():
    destination = DATASETS / "Bonn-RGBD/archives"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "rgbd_bonn_dataset.zip"
    original_receipt = target.with_name(target.name + ".verified.json")
    if target.exists() and original_receipt.exists():
        if read_json(original_receipt).get("sha256") == BONN_SHA256:
            print(f"Reuse verified original archive: {target}")
            return
    os.environ["HF_XET_CACHE"] = str(DATASETS / ".cache/huggingface/xet")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BONN_REPO, target.name, repo_type="dataset", revision=BONN_REVISION,
                           local_dir=destination, cache_dir=DATASETS / ".cache/huggingface/datasets")
    if file_hash(path) != BONN_SHA256:
        raise RuntimeError("Bonn mirror checksum mismatch")
    write_json(original_receipt, {"repo": BONN_REPO, "revision": BONN_REVISION, "sha256": BONN_SHA256})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["nyuv2-raw", "bonn"])
    args = parser.parse_args()
    require_dataset_volume()
    if args.dataset == "nyuv2-raw":
        download_nyuv2()
    else:
        download_bonn()


if __name__ == "__main__":
    main()
