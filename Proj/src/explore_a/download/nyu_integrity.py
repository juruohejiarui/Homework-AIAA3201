"""Official MD5 receipts avoid a second full read after aria2 has verified new downloads."""
import hashlib
from pathlib import Path


def target_path(plan, entry):
    name = entry["name"]
    if Path(name).name != name or not name.endswith(".zip"):
        raise ValueError(f"Unsafe Raw ZIP name: {name}")
    return Path(plan["output"]) / "archives/raw" / name


def complete(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] \
        and not path.with_name(path.name + ".aria2").exists()


def fingerprint(path, entry):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "official_md5": entry["md5"]}


def verified_receipt(path, entry, receipts):
    return complete(path, entry) and receipts.get(entry["name"], {}).get("file") \
        == fingerprint(path, entry)


def record_verified(path, entry, receipts, method):
    receipts[entry["name"]] = {"file": fingerprint(path, entry), "verified_by": method}


def verify_file(path, entry, receipts, recheck=False):
    if not recheck and verified_receipt(path, entry, receipts):
        return
    if not complete(path, entry):
        raise ValueError(f"Raw ZIP is missing, partial or has the wrong size: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != entry["md5"]:
        raise ValueError(f"Official MD5 mismatch: {path}; expected {entry['md5']}, got {digest}")
    record_verified(path, entry, receipts, "python-official-md5")
